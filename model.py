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

