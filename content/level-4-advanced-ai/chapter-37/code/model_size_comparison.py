"""Artifact Disk Size and Compression Ratio Comparison.

Compares serialized .pt artifact sizes across all Chapter 37 optimization techniques,
and demonstrates why sparse/compressed storage (gzip) shrinks unstructured pruned weights
whereas raw dense .pt does not.
"""

import gzip
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import torch

from edge_ai_foundations import train_baseline_model
from model_profiling import get_serialized_size_kb
from dynamic_quantization import apply_dynamic_quantization
from static_quantization_demo import build_static_quantized_model
from magnitude_pruning import apply_unstructured_pruning, remove_pruning_reparametrization
from structured_pruning_demo import apply_structured_l1_pruning
from knowledge_distillation import train_student_distilled


def get_gzip_size_kb(model: torch.nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        raw_path = tmp.name
    gz_path = raw_path + ".gz"
    try:
        torch.save(model, raw_path)
        with open(raw_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.writelines(f_in)
        return round(os.path.getsize(gz_path) / 1024.0, 2)
    finally:
        for p in (raw_path, gz_path):
            if os.path.exists(p):
                os.remove(p)


def main():
    print("=" * 75)
    print("CHAPTER 37: MODEL ARTIFACT SIZE & COMPRESSION RATIO AUDIT")
    print("=" * 75)

    fp32, (X_train, y_train, _, _) = train_baseline_model(epochs=25, seed=42)
    dyn_q = apply_dynamic_quantization(fp32)
    stat_q = build_static_quantized_model(fp32, X_train[:250])

    unstruct = apply_unstructured_pruning(fp32, amount=0.70)
    remove_pruning_reparametrization(unstruct)

    struct = apply_structured_l1_pruning(fp32, keep_ratio=0.50)
    distilled = train_student_distilled(fp32, X_train, y_train, epochs=25, seed=42)

    models = [
        ("1. FP32 Baseline", fp32),
        ("2. Dynamic INT8", dyn_q),
        ("3. Static INT8", stat_q),
        ("4. Unstructured Pruned (70%)", unstruct),
        ("5. Structured Pruned (50% ch)", struct),
        ("6. Distilled Student", distilled),
    ]

    base_kb = get_serialized_size_kb(fp32)
    print(f"{'Model Variant':<30} | {'Raw .pt (KB)':<14} | {'Ratio vs Base':<14} | {'Gzipped (KB)'}")
    print("-" * 75)

    for name, m in models:
        raw_kb = get_serialized_size_kb(m)
        gz_kb = get_gzip_size_kb(m)
        ratio = round(base_kb / max(0.01, raw_kb), 2)
        print(f"{name:<30} | {raw_kb:<14.2f} | {ratio:<13.2f}x | {gz_kb:.2f}")

    print("\nTakeaway: Unstructured pruning yields 1.00x raw disk compression until entropy/sparse")
    print("compression (gzip: drops drastically due to 70% repeated zero bits) or structured pruning is applied.")
    print("\n[SUCCESS] Model artifact size audit completed.")


if __name__ == "__main__":
    main()

