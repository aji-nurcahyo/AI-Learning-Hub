"""Sparsity vs Accuracy Sensitivity Curve Analysis.

Sweeps unstructured L1 pruning sparsity from 0% to 95%, comparing:
1. Zero-shot post-pruning accuracy (before fine-tuning).
2. Fine-tuned recovery accuracy (5 epochs mask-preserving FT).
Saves plot to experiments/pruning_sparsity_curve.png.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_ai_foundations import evaluate_accuracy, train_baseline_model
from magnitude_pruning import apply_unstructured_pruning, measure_sparsity, remove_pruning_reparametrization


def main():
    print("=" * 75)
    print("CHAPTER 37: PRUNING SPARSITY VS ACCURACY SENSITIVITY CURVE")
    print("=" * 75)

    fp32_model, (X_train, y_train, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    sparsities = [0.0, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]

    zero_shot_accs = []
    finetuned_accs = []
    actual_sparsities = []

    print(f"{'Target Sparsity':<16} | {'Actual %':<10} | {'Zero-Shot Acc':<15} | {'Post-FT Acc'}")
    print("-" * 62)

    for sp in sparsities:
        if sp == 0.0:
            acc0 = evaluate_accuracy(fp32_model, X_test, y_test) * 100
            actual_sparsities.append(0.0)
            zero_shot_accs.append(acc0)
            finetuned_accs.append(acc0)
            print(f"{'0%':<16} | {'0.00%':<10} | {acc0:<14.2f}% | {acc0:.2f}%")
            continue

        pm = apply_unstructured_pruning(fp32_model, amount=sp)
        act_sp = measure_sparsity(pm)
        acc_zs = evaluate_accuracy(pm, X_test, y_test) * 100

        pm.train()
        opt = optim.Adam(pm.parameters(), lr=5e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(5):
            opt.zero_grad()
            crit(pm(X_train), y_train).backward()
            opt.step()
        pm.eval()
        remove_pruning_reparametrization(pm)
        acc_ft = evaluate_accuracy(pm, X_test, y_test) * 100

        actual_sparsities.append(act_sp)
        zero_shot_accs.append(acc_zs)
        finetuned_accs.append(acc_ft)
        print(f"{int(sp*100):<15}% | {act_sp:<9.2f}% | {acc_zs:<14.2f}% | {acc_ft:.2f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(actual_sparsities, zero_shot_accs, marker="o", linestyle="--", color="orange", label="Zero-Shot Post-Prune")
    ax.plot(actual_sparsities, finetuned_accs, marker="s", linewidth=2, color="blue", label="After 5-Epoch Fine-Tuning")
    ax.set_title("Unstructured L1 Pruning: Sparsity vs Accuracy (EdgeSensorMLP)")
    ax.set_xlabel("Actual Zero-Weight Sparsity (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.abspath(os.path.join(script_dir, "..", "experiments", "pruning_sparsity_curve.png"))
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close(fig)

    print(f"\nSaved sparsity curve to: {out_img}")
    print("[SUCCESS] Pruning sparsity sweep completed.")


if __name__ == "__main__":
    main()

