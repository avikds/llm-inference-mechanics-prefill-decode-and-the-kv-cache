# LLM Inference Mechanics: Prefill, Decode and the KV Cache

Chapter 2 of Inference Engineering, built to be measured. Assemble a modern decoder block from scratch in PyTorch, RMSNorm, rotary positions, grouped-query attention with a KV cache and a SwiGLU MLP, and stack it into a small language model. Then take it apart the way the chapter does: split generation into prefill and decode and time them, verify that cached decoding reproduces a full recompute, measure the KV cache byte for byte against the formula, count parameters and FLOPs per token analytically and check them against the real model, compute arithmetic intensity for both phases and find the batch size where decode stops being memory-bound, watch attention cost grow with context length, compare the KV budgets of multi-head, grouped-query and multi-query attention, and finish with a roofline report that turns a model configuration and a GPU spec sheet into decode floor, time to first token and inter-token latency estimates. The report reproduces the parameter count of a real 8B model to within a fraction of a percent.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** RMSNorm
- [x] **2.** rope_cache
- [x] **3.** GQAAttention
- [x] **4.** TinyLM
- [x] **5.** prefill

---

Built on Deep-ML.
