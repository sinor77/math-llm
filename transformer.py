"""
transformer.py
==============
Full GPT-style Transformer building blocks:

    TransformerBlock   — one decoder layer (attention + FFN + residuals + norms)
    TransformerDecoder — stack of N TransformerBlocks

ARCHITECTURE OVERVIEW
---------------------
Each TransformerBlock applies two sub-layers with residual connections:

    x = x + Attention( LayerNorm(x) )   ← pre-norm style (more stable)
    x = x + FFN( LayerNorm(x) )

WHY PRE-NORM?
-------------
In the original Transformer paper (Vaswani 2017) LayerNorm was applied
AFTER the residual addition (post-norm).  Modern models (GPT-2 onwards)
apply it BEFORE (pre-norm) because it makes gradient flow more stable at
initialisation and allows training deeper networks without warmup tricks.

FEED-FORWARD NETWORK (FFN)
--------------------------
A two-layer MLP with a GELU activation:
    FFN(x) = W2 · GELU( W1 · x + b1 ) + b2

The inner dimension d_ff is typically 4 × d_model.  This is where the
model does most of its "factual" computation; attention handles routing,
the FFN handles transformation.

GELU vs ReLU
------------
GELU (Gaussian Error Linear Unit) is smoother than ReLU near zero and
empirically works better for language models.

RESIDUAL CONNECTIONS
--------------------
Every sub-layer adds its input back to its output:  x_out = x_in + f(x_in)
This means gradients can flow directly backwards through identity paths,
solving the vanishing-gradient problem in deep networks.
"""

import torch
import torch.nn as nn
from typing import Optional

from attention import CausalSelfAttention


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network used inside each Transformer block.

    Architecture:
        Linear(d_model → d_ff)  →  GELU  →  Linear(d_ff → d_model)  →  Dropout

    Parameters
    ----------
    d_model  : model dimension (input and output size)
    d_ff     : inner hidden dimension (typically 4 × d_model)
    dropout  : dropout probability applied after the second linear layer
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),              # smooth activation — better than ReLU for LMs
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Single Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One GPT-style decoder block.

    Applies (in order):
        1. LayerNorm → CausalSelfAttention → residual add
        2. LayerNorm → FeedForwardNetwork  → residual add

    This is called "pre-norm" layout because normalisation is applied
    BEFORE the sub-layer (not after as in the original paper).

    Parameters
    ----------
    d_model     : embedding/hidden dimension
    n_heads     : number of attention heads
    d_ff        : feed-forward inner dimension
    max_seq_len : maximum sequence length (for pre-allocating causal mask)
    dropout     : dropout rate
    use_flash   : use flash attention if available
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_flash: bool = True,
    ):
        super().__init__()

        # Pre-norm: normalise before attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_flash=use_flash,
        )

        # Pre-norm: normalise before FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                : (batch, seq_len, d_model)
        key_padding_mask : (batch, seq_len) bool — True at padding positions

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)
        """
        # --- Sub-layer 1: Self-attention with residual ---------------------
        # Pre-norm: normalise x, pass through attention, add back to x
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)

        # --- Sub-layer 2: FFN with residual --------------------------------
        x = x + self.ffn(self.norm2(x))

        return x


# ---------------------------------------------------------------------------
# Full Transformer Decoder Stack
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """
    Stack of N TransformerBlock layers forming the decoder trunk.

    Input  : token embeddings already summed with positional encodings
             Shape: (batch, seq_len, d_model)
    Output : contextualised hidden states
             Shape: (batch, seq_len, d_model)

    Parameters
    ----------
    n_layers    : number of decoder blocks
    d_model     : embedding dimension
    n_heads     : attention heads per block
    d_ff        : FFN inner dimension
    max_seq_len : max sequence length
    dropout     : dropout rate
    use_flash   : use flash attention if available
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_flash: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_flash=use_flash,
            )
            for _ in range(n_layers)
        ])

        # Final LayerNorm applied after all blocks (standard for pre-norm models)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pass embeddings through all decoder blocks then apply final norm.

        Parameters
        ----------
        x                : (batch, seq_len, d_model) — embedded input
        key_padding_mask : (batch, seq_len) bool — True at padding positions

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)
        """
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T = 4, 32          # batch=4, seq_len=32
    D, H = 128, 4         # d_model=128, n_heads=4
    FF   = 512            # d_ff=512
    L    = 4              # 4 layers

    decoder = TransformerDecoder(
        n_layers=L, d_model=D, n_heads=H, d_ff=FF,
        max_seq_len=64, dropout=0.0, use_flash=False,
    )

    x    = torch.randn(B, T, D)
    mask = torch.zeros(B, T, dtype=torch.bool)   # no padding
    out  = decoder(x, key_padding_mask=mask)

    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, T, D)

    # Count parameters
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"TransformerDecoder parameters: {n_params:,}")
    print("TransformerDecoder self-test passed.")
