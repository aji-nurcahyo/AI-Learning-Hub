"""Model Profiling Tool for Edge Inference Readiness.

Measures actual runtime footprint of a PyTorch model:
1. Total and trainable parameter counts.
2. Theoretical FP32 weight footprint vs actual serialized .pt artifact size on disk.
3. Cold-start latency vs warm-start latency (Mean, Median P50, P95 ms).
4. Peak Python memory allocation during single-sample inference (tracemalloc).
5. Hardware context documentation (OS, CPU count, PyTorch threads).
"""

import os
import platform
import tempfile
import time
import tracemalloc
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from edge_ai_foundations import evaluate_accuracy, train_baseline_model


def get_serialized_size_kb(model: nn.Module) -> float:
    """Saves model (state_dict or scripted/pickled) to measure exact byte size."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model, tmp_path)
        size_bytes = os.path.getsize(tmp_path)
        return round(size_bytes / 1024.0, 2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def profile_cpu_latency(model: nn.Module, sample_input: torch.Tensor, warmup: int = 50, n_runs: int = 500) -> Dict[str, float]:
    model.eval()
    latencies_ms = []

    with torch.no_grad():
        # Cold call
        t_c0 = time.perf_counter()
        _ = model(sample_input)
        cold_ms = (time.perf_counter() - t_c0) * 1000.0

        # Warm-up passes
        for _ in range(warmup):
            _ = model(sample_input)

        # Timed passes
        t_wall0 = time.perf_counter()
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(sample_input)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        wall_s = time.perf_counter() - t_wall0

    arr = np.array(latencies_ms)
    return {
        "cold_ms": round(cold_ms, 4),
        "mean_ms": round(float(np.mean(arr)), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "throughput_rps": round(float(n_runs / max(1e-9, wall_s)), 1),
    }


def profile_inference_peak_memory_kb(model: nn.Module, sample_input: torch.Tensor) -> float:
    """Measures Python heap peak allocation during forward pass via tracemalloc."""
    model.eval()
    tracemalloc.start()
    with torch.no_grad():
        _ = model(sample_input)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak_bytes / 1024.0, 2)


def main():
    print("=" * 75)
    print("CHAPTER 37: BASELINE FP32 MODEL PROFILING ON CPU")
    print("=" * 75)

    model, (_, _, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    sample = X_test[:1]

    param_count = sum(p.numel() for p in model.parameters())
    fp32_weight_kb = round((param_count * 4) / 1024.0, 2)
    disk_size_kb = get_serialized_size_kb(model)
    acc = evaluate_accuracy(model, X_test, y_test)
    lat = profile_cpu_latency(model, sample, warmup=50, n_runs=500)
    peak_kb = profile_inference_peak_memory_kb(model, sample)

    print("Hardware & Runtime Context:")
    print(f"- OS / Platform   : {platform.platform()}")
    print(f"- Logical CPUs    : {os.cpu_count()} | PyTorch Threads: {torch.get_num_threads()}")
    print(f"- PyTorch Version : {torch.__version__}\n")

    print("Baseline Model Footprint & Latency:")
    print(f"- Parameter Count       : {param_count:,}")
    print(f"- Raw FP32 Weight RAM   : {fp32_weight_kb:.2f} KB (4 bytes/param)")
    print(f"- Serialized .pt Size   : {disk_size_kb:.2f} KB")
    print(f"- Tracemalloc Peak Heap : {peak_kb:.2f} KB (single-sample forward)")
    print(f"- Test Accuracy         : {acc * 100:.2f}%")
    print(f"- Cold Latency          : {lat['cold_ms']:.4f} ms")
    print(f"- Warm Mean / P50 / P95 : {lat['mean_ms']:.4f} ms / {lat['p50_ms']:.4f} ms / {lat['p95_ms']:.4f} ms")
    print(f"- Throughput            : {lat['throughput_rps']} req/s")
    print("\n[SUCCESS] Baseline model profiling completed.")


if __name__ == "__main__":
    main()

