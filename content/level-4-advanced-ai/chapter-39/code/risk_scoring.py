"""Quantitative Risk Scoring & Heatmap Visualization (Likelihood x Severity)."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai_risk_register import build_sample_risk_register


def compute_risk_reduction_stats(register):
    inherent = [r.inherent_score for r in register]
    residual = [r.residual_score for r in register]
    pct_drop = 100.0 * (sum(inherent) - sum(residual)) / sum(inherent)
    return sum(inherent), sum(residual), pct_drop


def save_risk_matrix_heatmap(register, out_path: str):
    grid = np.zeros((5, 5), dtype=float)
    for l_idx in range(5):
        for s_idx in range(5):
            grid[l_idx, s_idx] = (l_idx + 1) * (s_idx + 1)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    im = ax.imshow(grid, origin="lower", cmap="YlOrRd", vmin=1, vmax=25)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["1\nNegligible", "2\nMinor", "3\nModerate", "4\nMajor", "5\nCritical"])
    ax.set_yticks(range(5))
    ax.set_yticklabels(["1 Rare", "2 Unlikely", "3 Possible", "4 Likely", "5 Almost\nCertain"])
    ax.set_xlabel("Severity (Impact)")
    ax.set_ylabel("Likelihood")
    ax.set_title("Chapter 39: Inherent vs Residual Risk Matrix (5x5)")

    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(int(grid[i, j])), ha="center", va="center", color="#333333", alpha=0.55, fontsize=9)

    for r in register:
        ix, iy = r.severity - 1, r.likelihood - 1
        rx, ry = r.residual_severity - 1, r.residual_likelihood - 1
        ax.scatter(ix, iy, color="#111111", s=80, zorder=5, label="Inherent" if r.risk_id == "RSK-01" else "")
        ax.scatter(rx, ry, color="#0d6efd", marker="D", s=70, zorder=6, label="Residual" if r.risk_id == "RSK-01" else "")
        ax.annotate("", xy=(rx, ry), xytext=(ix, iy), arrowprops=dict(arrowstyle="->", color="#0d6efd", lw=1.4))
        ax.text(ix + 0.08, iy + 0.12, r.risk_id, fontsize=8, fontweight="bold", color="#111111")

    ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax, label="Risk Score (L x S)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 75)
    print("CHAPTER 39: QUANTITATIVE RISK SCORING & MATRIX ANALYSIS")
    print("=" * 75)

    register = build_sample_risk_register()
    total_inh, total_res, pct = compute_risk_reduction_stats(register)

    print(f"Total Inherent Risk Points : {total_inh}")
    print(f"Total Residual Risk Points : {total_res}")
    print(f"Control Risk Reduction     : {pct:.2f}% (Note: residual risk never becomes absolute zero!)\n")

    for r in register:
        delta = r.inherent_score - r.residual_score
        print(f"- {r.risk_id}: Inherent={r.inherent_score:2d} ({r.priority:8s}) -> Residual={r.residual_score:2d} (Reduced by {delta})")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "risk_matrix_heatmap.png")
    save_risk_matrix_heatmap(register, out_path)
    assert os.path.exists(out_path)
    print(f"\nSaved Risk Matrix visualization to: {out_path}")
    print("[SUCCESS] Risk scoring verification passed.")


if __name__ == "__main__":
    main()

