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

# Step 4 - TinyLM
class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, hidden, max_len):
        super().__init__()

        self.norm1 = RMSNorm(d)
        self.attn = GQAAttention(d, n_heads, n_kv_heads, max_len)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, hidden)

    def forward(self, x, cache=None):
        # Pre-norm attention followed by a residual connection.
        attn_out, new_cache = self.attn(self.norm1(x), cache=cache)
        x = x + attn_out

        # Pre-norm SwiGLU MLP followed by a residual connection.
        x = x + self.mlp(self.norm2(x))

        return x, new_cache


class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Keep the original config object so that m.cfg is cfg.
        self.cfg = cfg

        # Token embedding.
        self.tok = nn.Embedding(cfg["vocab"], cfg["d"])

        # Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d=cfg["d"],
                n_heads=cfg["n_heads"],
                n_kv_heads=cfg["n_kv_heads"],
                hidden=cfg["hidden"],
                max_len=cfg["max_len"],
            )
            for _ in range(cfg["n_layers"])
        ])

        # Final normalization and vocabulary projection.
        self.norm = RMSNorm(cfg["d"])
        self.head = nn.Linear(cfg["d"], cfg["vocab"], bias=False)

    def forward(self, idx, cache=None):
        # Convert token IDs to embeddings.
        x = self.tok(idx)

        # Create an empty cache for every block on the first pass.
        if cache is None:
            cache = [None] * len(self.blocks)

        new_cache = []

        # Run each block with its corresponding cache.
        for block, block_cache in zip(self.blocks, cache):
            x, block_cache = block(x, cache=block_cache)
            new_cache.append(block_cache)

        # Final normalization and language-model head.
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_cache


def make_cfg(
    vocab=64,
    d=64,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    hidden=None,
    max_len=2048,
):
    # Default hidden size is four times the model width.
    if hidden is None:
        hidden = 4 * d

    return {
        "vocab": vocab,
        "d": d,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden": hidden,
        "max_len": max_len,
    }

# Step 5 - prefill
@torch.no_grad()
def prefill(model, prompt):
    # prompt is a 1-D tensor of token IDs; add a batch dimension.
    if prompt.dim() != 1:
        raise ValueError("prompt must be a 1-D tensor")

    logits, cache = model(prompt.unsqueeze(0))

    # Return only the logits corresponding to the final prompt token.
    next_logits = logits[0, -1]

    return next_logits, cache


@torch.no_grad()
def decode_step(model, token, cache):
    # Find the model's device so the new token is placed consistently.
    device = next(model.parameters()).device

    # Feed exactly one token with batch and sequence dimensions.
    token_tensor = torch.tensor(
        [[token]],
        dtype=torch.long,
        device=device,
    )

    logits, new_cache = model(token_tensor, cache=cache)

    # Remove batch and sequence dimensions.
    next_logits = logits[0, -1]

    return next_logits, new_cache


@torch.no_grad()
def generate(model, prompt, n, greedy=True, gen=None):
    # Prefill the prompt once to construct the initial KV cache.
    next_logits, cache = prefill(model, prompt)

    tokens = []

    for step in range(n):
        if greedy:
            # Greedy decoding selects the most probable token.
            token = torch.argmax(next_logits).item()
        else:
            # Sample from the probability distribution.
            probs = torch.softmax(next_logits, dim=-1)
            token = torch.multinomial(
                probs,
                num_samples=1,
                generator=gen,
            ).item()

        tokens.append(token)

        # Do not feed the final sampled token back into the model.
        # This keeps the number of model forward calls equal to n:
        # one prefill call + (n - 1) decode calls would otherwise be
        # needed for n generated tokens. Instead, the first generated
        # token is obtained from prefill's logits, and only subsequent
        # tokens require decode passes.
        if step < n - 1:
            next_logits, cache = decode_step(model, token, cache)

    return tokens, n


@torch.no_grad()
def generate_recompute(model, prompt, n):
    # Keep the complete growing sequence and recompute it from scratch
    # at every generation step.
    if prompt.dim() != 1:
        raise ValueError("prompt must be a 1-D tensor")

    device = next(model.parameters()).device
    sequence = prompt.to(device)

    tokens = []

    for _ in range(n):
        # Re-run the entire sequence without using a cache.
        logits, _ = model(sequence.unsqueeze(0))

        # Greedy next-token selection.
        token = torch.argmax(logits[0, -1]).item()
        tokens.append(token)

        # Append the newly generated token to the sequence.
        sequence = torch.cat([
            sequence,
            torch.tensor([token], dtype=torch.long, device=device),
        ])

    return tokens, n

# Step 6 - kv_cache_bytes
def kv_bytes_per_token(cfg, bytes_per_elem):
    # Each token stores one key and one value for every layer.
    head_dim = cfg["d"] // cfg["n_heads"]

    return (
        2
        * cfg["n_layers"]
        * cfg["n_kv_heads"]
        * head_dim
        * bytes_per_elem
    )


def kv_cache_bytes(cfg, seq_len, batch, bytes_per_elem):
    return (
        batch
        * seq_len
        * kv_bytes_per_token(cfg, bytes_per_elem)
    )


def measured_cache_bytes(cache):
    total_bytes = 0

    # Sum the storage used by every key and value tensor.
    for k, v in cache:
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()

    return total_bytes


def max_context(cfg, memory_bytes, batch, bytes_per_elem, weight_bytes):
    # Memory remaining after accounting for model weights.
    available_bytes = max(0, memory_bytes - weight_bytes)

    bytes_per_token = kv_bytes_per_token(cfg, bytes_per_elem)

    if bytes_per_token <= 0 or batch <= 0:
        return 0

    # Explicitly convert to int because inputs such as 80e9
    # and 16e9 are floating-point values.
    return int(
        available_bytes // (batch * bytes_per_token)
    )


def cache_growth(model, prompt, n):
    # Prefill the prompt and measure the initial cache.
    next_logits, cache = prefill(model, prompt)
    growth = [measured_cache_bytes(cache)]

    # Generate n additional tokens using cached decoding.
    for _ in range(n):
        token = torch.argmax(next_logits).item()
        next_logits, cache = decode_step(model, token, cache)
        growth.append(measured_cache_bytes(cache))

    return growth

# Step 7 - flops_per_token
def param_count(cfg):
    d = cfg["d"]
    vocab = cfg["vocab"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    n_kv_heads = cfg["n_kv_heads"]
    hidden = cfg["hidden"]

    head_dim = d // n_heads

    # Token embedding.
    embedding_params = vocab * d

    # Parameters in one Transformer block:
    # wq + wo
    attention_qo = 2 * d * d

    # wk + wv
    attention_kv = 2 * d * n_kv_heads * head_dim

    # w1 + w3 + w2 in SwiGLU
    mlp = 3 * d * hidden

    # norm1 + norm2
    block_norms = 2 * d

    per_block = attention_qo + attention_kv + mlp + block_norms

    # Final RMSNorm and vocabulary projection head.
    final_norm = d
    head = d * vocab

    return (
        embedding_params
        + n_layers * per_block
        + final_norm
        + head
    )


def matmul_params(cfg):
    d = cfg["d"]
    vocab = cfg["vocab"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    n_kv_heads = cfg["n_kv_heads"]
    hidden = cfg["hidden"]

    head_dim = d // n_heads

    # All weights that participate in matrix multiplications:
    # attention projections + SwiGLU projections + LM head.
    per_block = (
        2 * d * d
        + 2 * d * n_kv_heads * head_dim
        + 3 * d * hidden
    )

    head = d * vocab

    return n_layers * per_block + head


def flops_per_token(cfg, context_len):
    d = cfg["d"]
    n_layers = cfg["n_layers"]

    # Projection/MLP matrix multiplications account for
    # 2 FLOPs per parameter (multiply + add).
    matmul_flops = 2 * matmul_params(cfg)

    # Attention score and value multiplications.
    attention_flops = 4 * n_layers * d * context_len

    return matmul_flops + attention_flops


def prefill_flops(cfg, n):
    # During prefill, token position t attends over t positions.
    # Therefore sum the per-token FLOPs for context lengths 1..n.
    return sum(
        flops_per_token(cfg, context_len)
        for context_len in range(1, n + 1)
    )


def decode_flops(cfg, context_len):
    # A decode step attends over the full existing context.
    return flops_per_token(cfg, context_len)

