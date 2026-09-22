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
- [x] **6.** kv_cache_bytes
- [x] **7.** flops_per_token
- [x] **8.** arithmetic_intensity
- [x] **9.** time_phases
- [x] **10.** attention_cost_vs_length
- [x] **11.** kv_sharing_report
- [x] **12.** inference_report

## Results

```
tiny model: 131392 parameters (analytic) = 131392 (module)
cached vs recomputed greedy decoding: identical=True, passes 16 vs 16; cache after prefill 4096 bytes = formula 4096
prefill of 256 tokens: 6.58 ms (26.0 us per token); one decode step: 0.79 ms; decode step costs 30.7x a prefilled token
decode step time by batch: b=1: 0.70 ms  b=8: 0.88 ms  b=64: 1.85 ms  -> per sequence at 64: 0.029 ms
prefill time vs length: 512: 33.5 ms  1024: 78.3 ms  2048: 335.9 ms; growth exponent 1.662 (attention is half the FLOPs from context 496 on)

Llama-3-8B-shaped configuration on an H100-class GPU (989 TFLOPS BF16, 3.35 TB/s, 80 GB), prompt 1024, batch 1:
  params               8030261248
  weight_gb            16.061
  kv_kb_per_token      128.0
  max_context_batch1   487819
  decode_floor_ms      4.794
  ttft_ms              15.819
  itl_ms               4.834
  decode_intensity     0.96
  decode_bound         memory
  crossover_batch      None
KV sharing at 8192 tokens, batch 1:
  mha  kv heads 32     512 KB/token  cache  4.29 GB  saving 0%
  gqa  kv heads  8     128 KB/token  cache  1.07 GB  saving 75%
  mqa  kv heads  1      16 KB/token  cache  0.13 GB  saving 97%
decode intensity by batch at context 1024: b=1: 1.0  b=8: 7.3  b=64: 40.3  b=512: 93.8  (hardware ratio 295 ops/byte, crossover batch None)
```
