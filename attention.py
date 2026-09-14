"""
attention.py
============
Causal Multi-Head Self-Attention — implemented from scratch using PyTorch
primitives (no nn.MultiheadAttention).

WHAT IS SELF-ATTENTION?
-----------------------
Each token in the sequence is allowed to "look at" all other tokens that
come BEFORE it (causal / autoregressive).  For each position we compute
three vectors:
    Query  (Q) — what am I looking for?
    Key    (K) — what do I contain?
    Value  (V) — what do I actually pass forward?

The attention score between position i and position j is:
    score(i, j) = Q[i] · K[j] / sqrt(d_k)

After masking out future positions (j > i) and applying softmax, we get
attention weights that sum to 1.  The output for position i is the
weighted sum of all Value vectors.

WHY MULTIPLE HEADS?
-------------------
Different heads learn to attend to different kinds of relationships
(e.g. one head tracks numerical magnitude, another tracks operator type).
The outputs of all heads are concatenated and linearly projected back to
the original dimension.

CAUSAL MASK
-----------
Because this is a decoder-only model (like GPT), we must ensure that
when predicting token t, the model can only use tokens 0 … t-1.
We achieve this by setting attention scores at future positions to -inf
BEFORE the softmax, so they become ~0 after softmax.

FLASH-ATTENTION OPTION
-----------------------
If PyTorch >= 2.0 is available, we optionally use
torch.nn.functional.scaled_dot_product_attention which is a fused CUDA
kernel (flash attention).  This is faster and uses less memory.
We keep the manual implementation for educational clarity, selectable
via use_flash=False.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) multi-head self-attention for a decoder-only transformer.

    Parameters
    ----------
    d_model   : total embedding dimension (must be divisible by n_heads)
    n_heads   : number of attention heads
    max_seq_len : maximum sequence length (used to pre-allocate the causal mask)
    dropout   : dropout probability applied to attention weights
    use_flash : use PyTorch's built-in scaled_dot_product_attention (faster on GPU)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_flash: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_k       = d_model // n_heads   # dimension per head
        self.dropout_p = dropout
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        # ---- Linear projections for Q, K, V and the output ---------------
        # We fuse Q, K, V into a single (3 × d_model) → d_model projection
        # for efficiency: one matrix multiply produces all three.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ---- Causal mask: a lower-triangular boolean matrix ---------------
        # Shape: (1, 1, max_seq_len, max_seq_len)
        # True  = allowed to attend
        # False = masked out (future token)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        # Register as a buffer so it moves to the right device with .to(device)
        # but is NOT treated as a learnable parameter.
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, d_model)
            — the token embeddings at the current layer

        key_padding_mask : optional BoolTensor of shape (batch, seq_len)
            — True at positions that are PADDING (should be ignored)

        Returns
        -------
        Tensor of shape (batch, seq_len, d_model)
        """
        B, T, C = x.shape   # batch, time (seq_len), channels (d_model)

        # ---- Step 1: Compute Q, K, V in one matrix multiply ---------------
        # qkv : (B, T, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split into three tensors of shape (B, T, d_model)
        q, k, v = qkv.split(self.d_model, dim=2)

        # ---- Step 2: Reshape for multi-head attention ----------------------
        # New shape: (B, n_heads, T, d_k)
        # We move the head dimension before the time dimension so that
        # matrix multiplications operate per-head in parallel.
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        q = split_heads(q)   # (B, H, T, d_k)
        k = split_heads(k)   # (B, H, T, d_k)
        v = split_heads(v)   # (B, H, T, d_k)

        # ---- Step 3: Compute attention ------------------------------------
        if self.use_flash:
            # PyTorch 2.0+ fused kernel — handles masking internally.
            # is_causal=True applies the causal mask automatically.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            y = self._manual_attention(q, k, v, B, T, key_padding_mask)

        # ---- Step 4: Concatenate heads and project output -----------------
        # y : (B, H, T, d_k) → (B, T, H * d_k) = (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection + dropout
        y = self.resid_dropout(self.out_proj(y))
        return y

    # -----------------------------------------------------------------------
    # Manual attention implementation (for educational purposes)
    # -----------------------------------------------------------------------

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Scaled dot-product attention with a causal mask, computed manually.

        Attention formula:
            Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) · V

        The causal mask ensures score[i, j] = -inf when j > i.
        """
        # ---- Scaled dot-product scores ------------------------------------
        # scores : (B, H, T, T)
        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # ---- Apply causal mask --------------------------------------------
        # causal_mask is lower-triangular: True where j <= i (allowed).
        # We set disallowed positions (j > i) to -inf so softmax → 0.
        causal = self.causal_mask[:, :, :T, :T]   # trim to current seq_len
        scores = scores.masked_fill(~causal, float("-inf"))

        # ---- Apply padding mask (optional) --------------------------------
        # key_padding_mask : (B, T) — True where token is padding
        if key_padding_mask is not None:
            # Expand to (B, 1, 1, T) so it broadcasts over heads and query pos
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        # ---- Softmax + dropout -------------------------------------------
        attn_weights = F.softmax(scores, dim=-1)
        # Replace NaN (can arise if entire row is -inf) with 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # ---- Weighted sum of values --------------------------------------
        # output : (B, H, T, d_k)
        return torch.matmul(attn_weights, v)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D, H = 2, 16, 128, 4

    attn = CausalSelfAttention(d_model=D, n_heads=H, max_seq_len=T,
                                dropout=0.0, use_flash=False)
    x = torch.randn(B, T, D)
    out = attn(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, T, D), "Shape mismatch!"
    print("CausalSelfAttention self-test passed.")

    # Test with Flash Attention if available.
    # IMPORTANT: must compare the SAME weights through both code paths —
    # constructing a second module would draw fresh random weights from
    # the RNG stream and make the "diff" meaningless (any two independently
    # initialised modules will produce wildly different outputs regardless
    # of whether the attention math is correct). We instead flip the
    # use_flash flag on the existing module and copy state_dict across.
    if hasattr(F, "scaled_dot_product_attention"):
        attn_flash = CausalSelfAttention(d_model=D, n_heads=H, max_seq_len=T,
                                          dropout=0.0, use_flash=True)
        attn_flash.load_state_dict(attn.state_dict())
        out_flash = attn_flash(x)
        diff = (out - out_flash).abs().max().item()
        print(f"Flash vs manual max diff (same weights): {diff:.2e}  (should be < 1e-5)")
        assert diff < 1e-4, "Flash and manual attention implementations disagree!"
