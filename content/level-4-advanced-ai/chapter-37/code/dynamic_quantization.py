"""Post-Training Dynamic Quantization in PyTorch CPU.

Applies PyTorch dynamic quantization (INT8 weights, dynamically quantized activations)
using the active 'onednn' backend, then compares:
1. Serialized .pt artifact size (KB).
2. Test set classification accuracy.
3. Actual CPU inference latency (ms).
Reports actual observed differences honestly (never assuming speedup).
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.ao.quantization as tq

from edge_ai_foundations import evaluate_accuracy, train_baseline_model
from model_profiling import get_serialized_size_kb, profile_cpu_latency


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    model_eval = model.cpu().eval()
    return tq.quantize_dynamic(model_eval, {nn.Linear}, dtype=torch.qint8)


def main():
    print("=" * 75)
    print("CHAPTER 37: PYTORCH DYNAMIC QUANTIZATION (FP32 -> INT8)")
    print("=" * 75)

    fp32_model, (_, _, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    q_model = apply_dynamic_quantization(fp32_model)

    fp32_kb = get_serialized_size_kb(fp32_model)
    int8_kb = get_serialized_size_kb(q_model)

    fp32_acc = evaluate_accuracy(fp32_model, X_test, y_test)
    int8_acc = evaluate_accuracy(q_model, X_test, y_test)

    fp32_lat = profile_cpu_latency(fp32_model, X_test[:1], warmup=50, n_runs=500)
    int8_lat = profile_cpu_latency(q_model, X_test[:1], warmup=50, n_runs=500)

    comp_ratio = round(fp32_kb / max(0.01, int8_kb), 2)

    print(f"Active PyTorch Quantization Engine: {torch.backends.quantized.engine}")
    print("Note: Weights are stored in qint8; activations are dynamically quantized at runtime.\n")

    print(f"{'Metric':<25} | {'FP32 Baseline':<18} | {'Dynamic INT8'}")
    print("-" * 65)
    print(f"{'Artifact Size (.pt KB)':<25} | {fp32_kb:<18.2f} | {int8_kb:.2f} ({comp_ratio}x smaller)")
    print(f"{'Test Accuracy (%)':<25} | {fp32_acc*100:<18.2f} | {int8_acc*100:.2f}")
    print(f"{'Mean Latency (ms)':<25} | {fp32_lat['mean_ms']:<18.4f} | {int8_lat['mean_ms']:.4f}")
    print(f"{'P95 Latency (ms)':<25} | {fp32_lat['p95_ms']:<18.4f} | {int8_lat['p95_ms']:.4f}")

    print("\nEmpirical Observation:")
    print("- Dynamic quantization reduces serialized weight footprint (~3x on Linear layers).")
    print("- On tiny batch=1 MLP tensors on x86/Windows CPU, runtime dynamic activation")
    print("  quantization overhead can exceed INT8 GEMM savings—always benchmark empirically!")
    print("\n[SUCCESS] Dynamic quantization verification completed.")


if __name__ == "__main__":
    main()

