#!/usr/bin/env python3
"""Benchmark: TurboQuant KV-cache compression vs baseline.

Measures inference speed (tok/s), peak memory, and KV-cache size
with and without TurboQuant on a real model.

Usage:
    uv run python bench_turboquant.py [--model MODEL] [--tokens N] [--prompt-len N]
"""
from __future__ import annotations
import argparse, gc, time

def _get_memory_mb() -> float:
    """Current process RSS in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS returns bytes

def _metal_memory_mb() -> tuple[float, float]:
    """Return (active, peak) Metal GPU memory in MB."""
    try:
        import mlx.core as mx
        active = mx.metal.get_active_memory() / (1024**2)
        peak = mx.metal.get_peak_memory() / (1024**2)
        return active, peak
    except Exception:
        return 0.0, 0.0

def _reset_metal_peak():
    try:
        import mlx.core as mx
        mx.metal.reset_peak_memory()
    except Exception:
        pass

def run_benchmark(model_id: str, prompt_len: int, gen_tokens: int, use_turboquant: bool, tq_bits: int = 3):
    """Run a single benchmark pass. Returns dict of metrics."""
    from ppmlx.engine import TextEngine, reset_engine
    import mlx.core as mx

    # Reset singleton + caches
    reset_engine()

    if use_turboquant:
        import ppmlx.turboquant as tq
        from ppmlx.config import KVCacheConfig
        tq._rotation_matrices.clear()
        tq._jl_matrices.clear()
        kv_cfg = KVCacheConfig(quantize="turboquant", bits=tq_bits, qjl=True)
        engine = TextEngine(max_loaded=1, kv_cache=kv_cfg)
    else:
        # Reset the turboquant patch if previously applied
        import ppmlx.engine as eng
        eng._turboquant_patched = False
        engine = TextEngine(max_loaded=1)

    # Build a prompt of roughly prompt_len tokens
    filler = "The quick brown fox jumps over the lazy dog. " * (prompt_len // 10 + 1)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": filler[:prompt_len * 4]},  # ~4 chars/token
    ]

    # Warm up: load model
    print("  Loading model...", end=" ", flush=True)
    lm = engine.load(model_id)
    mx.eval(lm.model.parameters())
    gc.collect()
    print("done")

    _reset_metal_peak()
    metal_before, _ = _metal_memory_mb()

    # Generate
    print(f"  Generating {gen_tokens} tokens...", end=" ", flush=True)
    t0 = time.perf_counter()
    result = engine.generate(
        model_id, messages,
        max_tokens=gen_tokens,
        temperature=0.0,
        enable_thinking=False,
        strip_thinking=False,
    )
    t1 = time.perf_counter()
    print("done")

    metal_after, metal_peak = _metal_memory_mb()
    elapsed = t1 - t0

    # Estimate KV-cache size
    kv_bytes = 0
    for cache_obj in engine._prompt_cache._entries.values():
        for layer in cache_obj.cache:
            nb = getattr(layer, "nbytes", None)
            if isinstance(nb, int):
                kv_bytes += nb
            else:
                for attr in ("keys", "values"):
                    arr = getattr(layer, attr, None)
                    if arr is not None:
                        kv_bytes += getattr(arr, "nbytes", 0)

    metrics = {
        "mode": f"TurboQuant {tq_bits}-bit" if use_turboquant else "Baseline (no compression)",
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_per_s": round(result.completion_tokens / elapsed, 1) if elapsed > 0 else 0,
        "metal_active_mb": round(metal_after, 1),
        "metal_peak_mb": round(metal_peak, 1),
        "metal_delta_mb": round(metal_after - metal_before, 1),
        "kv_cache_mb": round(kv_bytes / (1024**2), 2),
        "output_preview": result.text[:120].replace("\n", " "),
    }

    # Cleanup
    engine.unload_all()
    engine.stop_reaper()
    gc.collect()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="TurboQuant benchmark")
    parser.add_argument("--model", default="mlx-community/Qwopus3.5-4B-v3-4bit",
                        help="Model alias or repo ID")
    parser.add_argument("--tokens", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--prompt-len", type=int, default=256, help="Approximate prompt length in tokens")
    parser.add_argument("--bits", type=int, default=3, help="TurboQuant bits (2, 3, 4)")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Prompt: ~{args.prompt_len} tokens, Generate: {args.tokens} tokens")
    print(f"TQ bits: {args.bits}")
    print("=" * 70)

    results = []
    for use_tq in [False, True]:
        label = f"TurboQuant {args.bits}-bit" if use_tq else "Baseline"
        print(f"\n--- {label} ---")
        m = run_benchmark(args.model, args.prompt_len, args.tokens, use_tq, args.bits)
        results.append(m)

    # Print comparison
    print("\n" + "=" * 70)
    print(f"{'Metric':<30s} {'Baseline':>15s} {'TurboQuant':>15s} {'Delta':>10s}")
    print("-" * 70)

    base, tq = results[0], results[1]
    rows = [
        ("Prompt tokens", base["prompt_tokens"], tq["prompt_tokens"], ""),
        ("Completion tokens", base["completion_tokens"], tq["completion_tokens"], ""),
        ("Time (s)", base["elapsed_s"], tq["elapsed_s"],
         f"{(tq['elapsed_s']/base['elapsed_s'] - 1)*100:+.0f}%" if base["elapsed_s"] > 0 else ""),
        ("Speed (tok/s)", base["tok_per_s"], tq["tok_per_s"],
         f"{(tq['tok_per_s']/base['tok_per_s'] - 1)*100:+.0f}%" if base["tok_per_s"] > 0 else ""),
        ("Metal active (MB)", base["metal_active_mb"], tq["metal_active_mb"],
         f"{tq['metal_active_mb'] - base['metal_active_mb']:+.0f}"),
        ("Metal peak (MB)", base["metal_peak_mb"], tq["metal_peak_mb"],
         f"{tq['metal_peak_mb'] - base['metal_peak_mb']:+.0f}"),
        ("KV-cache (MB)", base["kv_cache_mb"], tq["kv_cache_mb"],
         f"{(1 - tq['kv_cache_mb']/base['kv_cache_mb'])*100:.0f}% less" if base["kv_cache_mb"] > 0 else ""),
    ]

    for label, bval, tval, delta in rows:
        print(f"{label:<30s} {str(bval):>15s} {str(tval):>15s} {delta:>10s}")

    print("\nOutput preview (baseline):", base["output_preview"])
    print("Output preview (turboquant):", tq["output_preview"])


if __name__ == "__main__":
    main()
