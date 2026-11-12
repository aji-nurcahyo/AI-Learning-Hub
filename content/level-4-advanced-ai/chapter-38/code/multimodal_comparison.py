"""Multimodal Fusion Comparison: Clean vs Missing Modality Resilience.

Compares three fusion strategies on unseen test pairs (N=96):
1. Early Fusion (Feature Concatenation).
2. Late Fusion (Weighted Decision Ensemble).
3. Cross-Attention Fusion (Q=Text Tokens, K/V=Image Spatial Patches).

Evaluates three operating conditions: Both Present, Missing Text, Missing Image.
Saves grouped comparison chart to experiments/multimodal_fusion_comparison.png.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from early_fusion import train_and_evaluate_early_fusion
from late_fusion import train_and_evaluate_late_fusion
from cross_attention_fusion import train_and_evaluate_cross_attention


def main():
    print("=" * 75)
    print("CHAPTER 38: FUSION STRATEGY BENCHMARK & MISSING MODALITY RESILIENCE")
    print("=" * 75)

    _, ef_both, ef_notxt, ef_noimg = train_and_evaluate_early_fusion(epochs=20, seed=42)
    _, lf_both, lf_notxt, lf_noimg = train_and_evaluate_late_fusion(epochs=20, seed=42)
    _, ca_both, ca_notxt, ca_noimg = train_and_evaluate_cross_attention(epochs=20, seed=42)

    rows = [
        ("Early Fusion", ef_both, ef_notxt, ef_noimg),
        ("Late Fusion", lf_both, lf_notxt, lf_noimg),
        ("Cross-Attention", ca_both, ca_notxt, ca_noimg),
    ]

    print(f"{'Fusion Architecture':<20} | {'Both Present':<14} | {'No Text (Img Only)':<19} | {'No Image (Txt Only)'}")
    print("-" * 78)
    for name, b, nt, ni in rows:
        print(f"{name:<20} | {b:<13.2f}% | {nt:<18.2f}% | {ni:.2f}%")

    # Grouped bar chart
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width, [r[1] for r in rows], width, label="Both Modalities", color="steelblue")
    ax.bar(x, [r[2] for r in rows], width, label="Image Only (Missing Text)", color="orange")
    ax.bar(x + width, [r[3] for r in rows], width, label="Text Only (Missing Image)", color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Unseen Test Accuracy (%)")
    ax.set_title("Multimodal Fusion Comparison: Clean vs Missing Modality")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.abspath(os.path.join(script_dir, "..", "experiments", "multimodal_fusion_comparison.png"))
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close(fig)

    print(f"\nSaved comparison chart to: {out_img}")
    print("[SUCCESS] Multimodal fusion benchmark completed.")


if __name__ == "__main__":
    main()

