"""Subgroup Model Performance & Threshold Fairness Trade-off Analysis."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from dataset_bias_analysis import generate_synthetic_bias_dataset
from fairness_metrics import evaluate_fairness


def main():
    print("=" * 75)
    print("CHAPTER 39: SUBGROUP PERFORMANCE & THRESHOLD AUDIT")
    print("=" * 75)

    X, y, group, _ = generate_synthetic_bias_dataset(seed=42, n_samples=1200)
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=42, stratify=group)

    # Train classifier WITHOUT explicit group indicator (proxy bias still enters via f_proxy)
    clf = LogisticRegression(random_state=42)
    clf.fit(X[tr_idx], y[tr_idx])

    probs = clf.predict_proba(X[te_idx])[:, 1]
    preds = (probs >= 0.50).astype(int)
    y_te, g_te = y[te_idx], group[te_idx]

    overall_acc = accuracy_score(y_te, preds)
    r0, r1, f_metrics = evaluate_fairness(y_te, preds, g_te)
    acc0 = accuracy_score(y_te[g_te == 0], preds[g_te == 0])
    acc1 = accuracy_score(y_te[g_te == 1], preds[g_te == 1])

    print(f"Overall Test Accuracy      : {overall_acc * 100:.2f}%")
    print(f"Group A (n={int((g_te==0).sum()):3d})           : Acc={acc0*100:.2f}% | Sel={r0['selection_rate']:.3f} | TPR={r0['tpr']:.3f} | FPR={r0['fpr']:.3f}")
    print(f"Group B (n={int((g_te==1).sum()):3d})           : Acc={acc1*100:.2f}% | Sel={r1['selection_rate']:.3f} | TPR={r1['tpr']:.3f} | FPR={r1['fpr']:.3f}")
    print(f"Fairness Disparity Summary : DP Diff={f_metrics['dp_diff']:.3f} | EO Diff={f_metrics['eo_diff']:.3f} | EqOdds={f_metrics['eq_odds_diff']:.3f}")

    # Sweep threshold for Group B to inspect EO vs Overall Accuracy trade-off
    thresholds = np.linspace(0.25, 0.65, 9)
    eo_curve, dp_curve, acc_curve = [], [], []
    for th_b in thresholds:
        adj_preds = np.where(g_te == 0, (probs >= 0.50).astype(int), (probs >= th_b).astype(int))
        _, _, fm = evaluate_fairness(y_te, adj_preds, g_te)
        eo_curve.append(fm["eo_diff"])
        dp_curve.append(fm["dp_diff"])
        acc_curve.append(accuracy_score(y_te, adj_preds))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fairness_metrics_comparison.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(thresholds, acc_curve, marker="o", label="Overall Accuracy")
    ax.plot(thresholds, eo_curve, marker="s", label="Equal Opportunity Diff (|dTPR|)")
    ax.plot(thresholds, dp_curve, marker="^", label="Demographic Parity Diff (|dSel|)")
    ax.axvline(0.50, color="gray", linestyle="--", label="Uniform Threshold (0.50)")
    ax.set_xlabel("Group B Decision Threshold (Group A fixed at 0.50)")
    ax.set_ylabel("Metric Value")
    ax.set_title("Chapter 39: Accuracy vs Fairness Gaps Across Group Thresholds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved Fairness Trade-off Chart to: {out_path}")
    print("[SUCCESS] Subgroup performance analysis passed.")


if __name__ == "__main__":
    main()

