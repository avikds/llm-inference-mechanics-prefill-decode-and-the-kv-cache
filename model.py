"""
LLM Inference Mechanics: Prefill, Decode and the KV Cache

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - RMSNorm
import torch
import torch.nn as nn

def rms_norm(x, weight, eps=1e-6):
    # Compute the root-mean-square normalization along the last dimension.
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps

        # Learnable scale parameter, initialized to ones.
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def swiglu(x, w1, w3, w2):
    # SwiGLU(x) = w2(SiLU(w1(x)) * w3(x))
    return w2(torch.nn.functional.silu(w1(x)) * w3(x))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()

        # Input projection and gating projection.
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)

        # Output projection back to model dimension.
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return swiglu(x, self.w1, self.w3, self.w2)

# Step 2 - rope_cache
def rope_cache(max_len, head_dim, base=10000.0):
    # Compute the inverse frequencies:
    # base^(-2i / head_dim), for i = 0, ..., head_dim // 2 - 1.
    half_dim = head_dim // 2
    inv_freq = base ** (
        -2 * torch.arange(half_dim, dtype=torch.float32) / head_dim
    )

    # Position indices: 0, 1, ..., max_len - 1.
    positions = torch.arange(max_len, dtype=torch.float32)

    # Each entry [t, i] contains:
    # t * base^(-2i / head_dim)
    angles = positions[:, None] * inv_freq[None, :]

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin, offset=0):
    # x has shape (B, H, T, head_dim).
    B, H, T, head_dim = x.shape
    half_dim = head_dim // 2

    # Split the final dimension into two halves.
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Select the position rows needed for this sequence.
    c = cos[offset:offset + T].to(dtype=x.dtype, device=x.device)
    s = sin[offset:offset + T].to(dtype=x.dtype, device=x.device)

    # Reshape for broadcasting over batch and heads.
    c = c[None, None, :, :]
    s = s[None, None, :, :]

    # Apply the rotary transformation and concatenate the halves.
    rotated_x1 = x1 * c - x2 * s
    rotated_x2 = x1 * s + x2 * c

    return torch.cat([rotated_x1, rotated_x2], dim=-1)


def rope_is_relative(cos, sin, head_dim, seed=0):
    # Make the random vectors reproducible.
    torch.manual_seed(seed)

    # Draw one random query and one random key vector.
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    # Give them the shape expected by apply_rope:
    # (B, H, T, head_dim).
    q = q.reshape(1, 1, 1, head_dim)
    k = k.reshape(1, 1, 1, head_dim)

    # Rotate at positions 5 and 2.
    q_5 = apply_rope(q, cos, sin, offset=5)[0, 0, 0]
    k_2 = apply_rope(k, cos, sin, offset=2)[0, 0, 0]

    # Rotate the same vectors at positions 15 and 12.
    q_15 = apply_rope(q, cos, sin, offset=15)[0, 0, 0]
    k_12 = apply_rope(k, cos, sin, offset=12)[0, 0, 0]

    # RoPE should preserve the dot product when both positions
    # are shifted by the same amount, since their relative offset
    # remains unchanged.
    dot_1 = torch.dot(q_5, k_2)
    dot_2 = torch.dot(q_15, k_12)

    return torch.allclose(dot_1, dot_2, atol=1e-4, rtol=0.0)

# Step 3 - GQAAttention
class GQAAttention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, max_len=2048):
        super().__init__()

        if d % n_heads != 0:
            raise ValueError("d must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.d = d
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_len = max_len

        # Dimension of each attention head.
        self.head_dim = d // n_heads

        # Bias-free projections.
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

        # Precompute rotary position embeddings and store them as buffers.
        cos, sin = rope_cache(max_len, self.head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, cache=None):
        B, T, _ = x.shape

        # Determine the number of cached/past tokens.
        if cache is None:
            S_past = 0
            k_past = None
            v_past = None
        else:
            k_past, v_past = cache
            S_past = k_past.shape[2]

        # Project queries, keys, and values.
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to:
        # q -> (B, n_heads, T, head_dim)
        # k,v -> (B, n_kv_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE using absolute sequence positions.
        q = apply_rope(q, self.cos, self.sin, offset=S_past)
        k = apply_rope(k, self.cos, self.sin, offset=S_past)

        # Append the newly generated keys and values to the existing cache.
        if cache is None:
            k_all = k
            v_all = v
        else:
            k_all = torch.cat([k_past, k], dim=2)
            v_all = torch.cat([v_past, v], dim=2)

        # Expand KV heads to the number of query heads.
        # Each KV head is shared by an equal group of query heads.
        repeat_factor = self.n_heads // self.n_kv_heads
        k_attn = k_all.repeat_interleave(repeat_factor, dim=1)
        v_attn = v_all.repeat_interleave(repeat_factor, dim=1)

        # Scaled dot-product attention.
        scores = torch.matmul(q, k_attn.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask using absolute positions.
        # Query i corresponds to absolute position S_past + i,
        # and may attend only to keys at positions <= that position.
        S_total = S_past + T
        q_positions = torch.arange(
            S_past, S_total, device=x.device
        )[:, None]
        k_positions = torch.arange(
            S_total, device=x.device
        )[None, :]

        causal_mask = k_positions <= q_positions

        # Mask future positions before softmax.
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values.
        out = torch.matmul(attn, v_attn)

        # Merge attention heads back into model dimension.
        out = out.transpose(1, 2).contiguous().view(B, T, self.d)

        # Final output projection.
        out = self.wo(out)

        # Return the un-expanded KV cache.
        return out, (k_all, v_all)

