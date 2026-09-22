"""
LLM Inference Mechanics: Prefill, Decode and the KV Cache scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""LLM Inference Mechanics: Prefill, Decode and the KV Cache (Inference Engineering, chapter 2).

Story: build a small decoder-only transformer with grouped-query attention and a
KV cache; split generation into prefill and decode and prove they agree; measure
the cache against its formula; count parameters and FLOPs and check them on a real
8B configuration; compute arithmetic intensity for both phases; watch attention
cost grow with context; compare KV budgets across head-sharing schemes; and print
the roofline report for an 8B model on an H100-class GPU.
"""
import torch


def main() -> None:
    torch.manual_seed(0)
    cfg = make_cfg()
    model = TinyLM(cfg)
    model.eval()
    print(f"tiny model: {param_count(cfg)} parameters (analytic) = {sum(p.numel() for p in model.parameters())} (module)")

    # ---- 1. Prefill and decode agree ----
    prompt = torch.arange(8)
    cached, p1 = generate(model, prompt, 16)
    recomputed, p2 = generate_recompute(model, prompt, 16)
    print(f"cached vs recomputed greedy decoding: identical={cached == recomputed}, passes {p1} vs {p2}; "
          f"cache after prefill {measured_cache_bytes(prefill(model, prompt)[1])} bytes = formula {kv_cache_bytes(cfg, 8, 1, 4)}")

    # ---- 2. Timing the two phases ----
    r = time_phases(model, 256, 1)
    print(f"prefill of 256 tokens: {r['prefill_s'] * 1000:.2f} ms ({r['prefill_per_token_s'] * 1e6:.1f} us per token); "
          f"one decode step: {r['decode_step_s'] * 1000:.2f} ms; decode step costs {r['ratio']:.1f}x a prefilled token")
    tb = time_decode_vs_batch(model, [1, 8, 64], 64)
    print("decode step time by batch: " + "  ".join(f"b={b}: {t * 1000:.2f} ms" for b, t in tb.items()) + f"  -> per sequence at 64: {tb[64] / 64 * 1000:.3f} ms")

    # ---- 3. Attention grows with context ----
    t = attention_cost_vs_length(model, [512, 1024, 2048])
    print(f"prefill time vs length: " + "  ".join(f"{n}: {s * 1000:.1f} ms" for n, s in t.items()) + f"; growth exponent {growth_exponent(t)} "
          f"(attention is half the FLOPs from context {context_where_attention_dominates(cfg)} on)")

    # ---- 4. A real configuration on a real GPU ----
    llama = make_cfg(vocab=128256, d=4096, n_layers=32, n_heads=32, n_kv_heads=8, hidden=14336, max_len=8192)
    hw = {"peak_flops": 989e12, "bandwidth": 3.35e12, "bytes_per_elem": 2, "memory_bytes": 80e9}
    print(f"\nLlama-3-8B-shaped configuration on an H100-class GPU (989 TFLOPS BF16, 3.35 TB/s, 80 GB), prompt 1024, batch 1:")
    for line in format_report(inference_report(llama, hw, 1024, 1)):
        print("  " + line)
    print("KV sharing at 8192 tokens, batch 1:")
    for row in kv_sharing_report(llama, 8192, 1, 2):
        print(f"  {row['name']:4s} kv heads {row['n_kv_heads']:2d}  {row['kv_bytes_per_token'] / 1024:6.0f} KB/token  cache {row['cache_bytes'] / 1e9:5.2f} GB  saving {row['saving_vs_mha']:.0%}")
    print("decode intensity by batch at context 1024: " + "  ".join(f"b={b}: {arithmetic_intensity(llama, b, 1024, 2, 'decode'):.1f}" for b in (1, 8, 64, 512))
          + f"  (hardware ratio {989e12 / 3.35e12:.0f} ops/byte, crossover batch {crossover_batch(llama, 1024, 2, 989e12 / 3.35e12)})")


if __name__ == "__main__":
    main()

