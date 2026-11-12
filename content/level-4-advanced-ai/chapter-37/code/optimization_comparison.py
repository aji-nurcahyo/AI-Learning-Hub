"""Unified Edge Optimization Scorecard & Trade-Off Visualization.

Evaluates and plots all key trade-offs:
- FP32 Baseline
- Dynamic INT8
- Static INT8
- Unstructured Pruned + FT (70%)
- Structured Pruned + FT (50%)
- Distilled Student (1.1K params)

Saves 3-panel trade-off visualization to experiments/edge_optimization_tradeoffs.png.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_ai_foundations import evaluate_accuracy, train_baseline_model
from model_profiling import get_serialized_size_kb, profile_cpu_latency
from dynamic_quantization import apply_dynamic_quantization
from static_quantization_demo import build_static_quantized_model
from magnitude_pruning import apply_unstructured_pruning, remove_pruning_reparametrization
from structured_pruning_demo import apply_structured_l1_pruning
from knowledge_distillation import train_student_distilled


def main():
    print("=" * 75)
    print("CHAPTER 37: UNIFIED EDGE OPTIMIZATION STRATEGY COMPARISON")
    print("=" * 75)

    fp32, (X_train, y_train, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    dyn_q = apply_dynamic_quantization(fp32)
    stat_q = build_static_quantized_model(fp32, X_train[:250])

    # Unstructured pruned + 5-epoch FT
    unstruct = apply_unstructured_pruning(fp32, amount=0.70)
    unstruct.train()
    opt = optim.Adam(unstruct.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(5):
        opt.zero_grad()
        crit(unstruct(X_train), y_train).backward()
        opt.step()
    unstruct.eval()
    remove_pruning_reparametrization(unstruct)

    # Structured pruned + 10-epoch FT
    struct = apply_structured_l1_pruning(fp32, keep_ratio=0.50)
    struct.train()
    opt_s = optim.Adam(struct.parameters(), lr=1e-3)
    for _ in range(10):
        opt_s.zero_grad()
        crit(struct(X_train), y_train).backward()
        opt_s.step()
    struct.eval()

    # Distilled student
    distilled = train_student_distilled(fp32, X_train, y_train, epochs=25, temperature=3.0, alpha=0.6, seed=42)

    candidates = [
        ("FP32 Baseline", fp32),
        ("Dynamic INT8", dyn_q),
        ("Static INT8", stat_q),
        ("Unstruct Pruned", unstruct),
        ("Struct Pruned", struct),
        ("Distilled Student", distilled),
    ]

    summary_rows = []
    print(f"{'Strategy':<18} | {'Acc (%)':<8} | {'Size (KB)':<10} | {'Mean Lat (ms)':<14} | {'P95 (ms)'}")
    print("-" * 70)

    for name, m in candidates:
        acc = round(evaluate_accuracy(m, X_test, y_test) * 100, 2)
        kb = get_serialized_size_kb(m)
        lat = profile_cpu_latency(m, X_test[:1], warmup=50, n_runs=400)
        summary_rows.append((name, acc, kb, lat["mean_ms"], lat["p95_ms"]))
        print(f"{name:<18} | {acc:<8.2f} | {kb:<10.2f} | {lat['mean_ms']:<14.4f} | {lat['p95_ms']:.4f}")

    # Plot 3-Panel Trade-Off Chart
    names = [r[0] for r in summary_rows]
    accs = [r[1] for r in summary_rows]
    sizes = [r[2] for r in summary_rows]
    lats = [r[3] for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = ["steelblue", "teal", "seagreen", "orange", "purple", "crimson"]

    axes[0].barh(names, sizes, color=colors)
    axes[0].set_title("Artifact Size (.pt KB) - Lower is Better")
    axes[0].set_xlabel("Size (KB)")
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(names, accs, color=colors)
    axes[1].set_title("Test Accuracy (%) - Higher is Better")
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_xlim(50, 100)
    axes[1].invert_yaxis()
    axes[1].grid(True, axis="x", alpha=0.3)

    axes[2].scatter(lats, accs, c=colors, s=120, edgecolors="black")
    for n, l, a in zip(names, lats, accs):
        axes[2].annotate(n, (l, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    axes[2].set_title("CPU Latency (ms) vs Accuracy (%)")
    axes[2].set_xlabel("Mean Single-Sample Latency (ms) [Windows CPU]")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_ylim(min(accs) - 5, 100)
    axes[2].grid(True, alpha=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.abspath(os.path.join(script_dir, "..", "experiments", "edge_optimization_tradeoffs.png"))
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close(fig)

    print(f"\nSaved trade-off chart to: {out_img}")
    print("[SUCCESS] Unified optimization comparison completed.")


if __name__ == "__main__":
    main()

