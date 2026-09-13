"""
model.py
========
Top-level MathLLM model — ties together the embedding table, positional
encoding, transformer decoder stack, and output projection head.

MODEL OVERVIEW
--------------
The model is a decoder-only GPT-style Transformer initialised from RANDOM
weights (no pretrained weights used anywhere).

Data flow during a forward pass:

    token_ids  (B, T)
        ↓ token embedding lookup
    token_emb  (B, T, d_model)
        ↓ + positional encoding
    x          (B, T, d_model)
        ↓ dropout
    x          (B, T, d_model)
        ↓ transformer decoder (N blocks of attention + FFN)
    hidden     (B, T, d_model)
        ↓ linear output head
    logits     (B, T, vocab_size)

WHAT ARE EMBEDDINGS?
--------------------
A learnable table of shape (vocab_size, d_model).  Token ID 42 simply
looks up row 42 — a dense vector of d_model floats.  During training the
model learns what each vector should represent.

WHAT IS POSITIONAL ENCODING?
-----------------------------
Attention is order-invariant by default.  To give the model a sense of
position we add a learned vector for each position 0 … T-1.
(Alternatively one can use sinusoidal encodings — both work similarly
for small models.)

WEIGHT TYING
------------
We re-use the token embedding matrix as the output projection:
    logits = hidden @ embedding_table.T

This is standard practice (Press & Wolf 2017) — it halves the number of
parameters and typically improves perplexity.

NEXT-TOKEN PREDICTION
---------------------
Given input [t0, t1, t2, …, tN-1] the model outputs logits for each
position.  The target at position i is the next token t_{i+1}.
We shift input and labels by one when computing the loss.

CAUSAL MASKING
--------------
The attention mechanism ensures position i can only attend to positions
0 … i.  This means the model cannot "cheat" by looking at the answer
when predicting the answer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from config import ModelConfig
from transformer import TransformerDecoder


class MathLLM(nn.Module):
    """
    Decoder-only GPT-style language model for mathematical problem solving.

    Parameters
    ----------
    cfg : ModelConfig
        Architecture hyper-parameters.  See config.py for details.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ----------------------------------------------------------------
        # 1. Token embedding table
        #    Maps each token ID to a d_model-dimensional dense vector.
        #    Shape: (vocab_size, d_model)
        # ----------------------------------------------------------------
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ----------------------------------------------------------------
        # 2. Positional embedding table (learned)
        #    One vector per sequence position, 0 … max_seq_len - 1.
        #    Shape: (max_seq_len, d_model)
        # ----------------------------------------------------------------
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # ----------------------------------------------------------------
        # 3. Embedding dropout (applied after token + position sum)
        # ----------------------------------------------------------------
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # ----------------------------------------------------------------
        # 4. Transformer decoder stack (N blocks)
        # ----------------------------------------------------------------
        self.decoder = TransformerDecoder(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
            use_flash=True,
        )

        # ----------------------------------------------------------------
        # 5. Output projection: d_model → vocab_size
        #    No bias — tied to the token embedding table.
        # ----------------------------------------------------------------
        self.output_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # ----------------------------------------------------------------
        # Weight tying: output_head.weight = token_embedding.weight
        # The same matrix is used for both embedding lookup AND logit
        # computation.  Sharing weights halves this large matrix and
        # acts as a regulariser.
        # ----------------------------------------------------------------
        if cfg.tie_weights:
            self.output_head.weight = self.token_embedding.weight

        # ----------------------------------------------------------------
        # Weight initialisation
        # ----------------------------------------------------------------
        self._init_weights()

        # ----------------------------------------------------------------
        # Report parameter count
        # ----------------------------------------------------------------
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MathLLM] Model '{cfg.name}' created.")
        print(f"  Total parameters    : {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Architecture        : {cfg.n_layers} layers × "
              f"d_model={cfg.d_model} × n_heads={cfg.n_heads} × d_ff={cfg.d_ff}")
        print(f"  Vocab size          : {cfg.vocab_size}")
        print(f"  Max sequence length : {cfg.max_seq_len}")

    # -----------------------------------------------------------------------
    # Weight initialisation
    # -----------------------------------------------------------------------

    def _init_weights(self):
        """
        Initialise weights following GPT-2 conventions:
          - Linear layers: normal(mean=0, std=0.02)
          - Embedding layers: normal(mean=0, std=0.02)
          - Residual projection layers: scaled by 1/sqrt(n_layers)
          - LayerNorm: weight=1, bias=0

        The scaled initialisation for residual projections prevents the
        residual stream variance from growing with depth.
        """
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(_init_module)

        # Scale the output projections of attention and FFN by 1/sqrt(2 * n_layers)
        # (two residual paths per block: attention + FFN)
        scale = (2 * self.cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if "out_proj.weight" in name or ("ffn" in name and name.endswith(".weight")
                                              and "net.4" in name):
                p.data.mul_(scale)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the full model.

        Parameters
        ----------
        input_ids : LongTensor of shape (batch, seq_len)
            Token IDs (from the tokenizer).

        labels : LongTensor of shape (batch, seq_len), optional
            Target token IDs for loss computation.  Positions with value
            -100 are ignored by the loss (question tokens, padding).
            If None, only logits are returned (inference mode).

        key_padding_mask : BoolTensor of shape (batch, seq_len), optional
            True at PAD positions.  If None, we infer it from pad_token_id=0.

        Returns
        -------
        dict with keys:
            "logits" : FloatTensor (batch, seq_len, vocab_size)
            "loss"   : scalar FloatTensor — cross-entropy loss (only if labels given)
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        # ---- Step 1: Build position indices [0, 1, 2, …, T-1] -----------
        # Same positions for every item in the batch.
        pos = torch.arange(T, dtype=torch.long, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(B, -1)   # (B, T)

        # ---- Step 2: Compute token + position embeddings -----------------
        tok_emb = self.token_embedding(input_ids)   # (B, T, d_model)
        pos_emb = self.pos_embedding(pos)            # (B, T, d_model)
        x = self.embed_dropout(tok_emb + pos_emb)   # (B, T, d_model)

        # ---- Step 3: Infer padding mask if not provided ------------------
        if key_padding_mask is None:
            # Assume token ID 0 is <PAD>
            key_padding_mask = (input_ids == 0)     # (B, T) bool

        # ---- Step 4: Pass through transformer decoder --------------------
        hidden = self.decoder(x, key_padding_mask=key_padding_mask)
        # hidden : (B, T, d_model)

        # ---- Step 5: Project to vocabulary logits ------------------------
        logits = self.output_head(hidden)   # (B, T, vocab_size)

        # ---- Step 6: Compute loss if labels provided ---------------------
        loss = None
        if labels is not None:
            # Causal LM: at each position i, predict the NEXT token (i+1).
            # Shift: input[0..T-2] → predict label[1..T-1]
            shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
            shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

            # cross_entropy expects (N, C) logits and (N,) labels
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),   # (B*(T-1), V)
                shift_labels.view(-1),                           # (B*(T-1),)
                ignore_index=-100,     # -100 → question / padding tokens
            )

        return {"logits": logits, "loss": loss}

    # -----------------------------------------------------------------------
    # Convenience: parameter count
    # -----------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count."""
        return sum(
            p.numel() for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )

    # -----------------------------------------------------------------------
    # Serialisation helpers
    # -----------------------------------------------------------------------

    def get_config_dict(self) -> Dict:
        """Return model config as a plain dict for saving."""
        return {
            "vocab_size":   self.cfg.vocab_size,
            "max_seq_len":  self.cfg.max_seq_len,
            "d_model":      self.cfg.d_model,
            "n_heads":      self.cfg.n_heads,
            "n_layers":     self.cfg.n_layers,
            "d_ff":         self.cfg.d_ff,
            "dropout":      self.cfg.dropout,
            "weight_decay": self.cfg.weight_decay,
            "tie_weights":  self.cfg.tie_weights,
            "name":         self.cfg.name,
        }

    @classmethod
    def from_config_dict(cls, d: Dict) -> "MathLLM":
        """Reconstruct a MathLLM from a config dict (e.g. loaded from JSON)."""
        cfg = ModelConfig(**d)
        return cls(cfg)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from config import get_model_config

    for preset in ["tiny-1M", "small-2M", "medium-5M"]:
        cfg = get_model_config(preset)
        cfg.vocab_size = 128   # pretend tokenizer returned 128 tokens

        model = MathLLM(cfg)

        # Forward pass
        B, T = 4, 64
        ids    = torch.randint(0, cfg.vocab_size, (B, T))
        labels = torch.randint(0, cfg.vocab_size, (B, T))
        labels[:, :10] = -100   # mask question tokens

        out = model(ids, labels=labels)
        print(f"  logits: {out['logits'].shape}   loss: {out['loss'].item():.4f}")

        # Round-trip serialisation
        cfg_dict = model.get_config_dict()
        model2   = MathLLM.from_config_dict(cfg_dict)
        assert model2.num_parameters() == model.num_parameters()
        print(f"  Config round-trip OK  ({preset})\n")
