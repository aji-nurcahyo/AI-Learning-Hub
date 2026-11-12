"""Fairness Metrics Calculation & Conflict Verification (DP vs EO vs Equalized Odds).

Includes:
1. A worked hand-verified 2-group confusion matrix demonstrating step-by-step arithmetic
   for Demographic Parity Difference, Equal Opportunity Difference, and Equalized Odds.
2. Demonstration that fairness criteria can conflict when base rates differ.
"""

import numpy as np


def compute_group_rates(y_true: np.ndarray, y_pred: np.ndarray):
    pos_pred_rate = float(np.mean(y_pred == 1))
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = float(np.mean(y_pred[pos_mask] == 1)) if pos_mask.any() else 0.0
    fpr = float(np.mean(y_pred[neg_mask] == 1)) if neg_mask.any() else 0.0
    return {"selection_rate": pos_pred_rate, "tpr": tpr, "fpr": fpr}


def evaluate_fairness(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray):
    r0 = compute_group_rates(y_true[group == 0], y_pred[group == 0])
    r1 = compute_group_rates(y_true[group == 1], y_pred[group == 1])
    dp_diff = abs(r0["selection_rate"] - r1["selection_rate"])
    eo_diff = abs(r0["tpr"] - r1["tpr"])
    fpr_diff = abs(r0["fpr"] - r1["fpr"])
    eq_odds_diff = max(eo_diff, fpr_diff)
    return r0, r1, {"dp_diff": dp_diff, "eo_diff": eo_diff, "fpr_diff": fpr_diff, "eq_odds_diff": eq_odds_diff}


def worked_manual_fairness_example():
    """Exact manual worked verification on 20 synthetic predictions across 2 groups."""
    # Group 0 (n=10, 6 positives, 4 negatives): TP=4, FN=2, FP=1, TN=3 -> Sel=5/10=0.50, TPR=4/6=0.6667, FPR=1/4=0.25
    # Group 1 (n=10, 3 positives, 7 negatives): TP=2, FN=1, FP=1, TN=6 -> Sel=3/10=0.30, TPR=2/3=0.6667, FPR=1/7=0.1429
    y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0,  1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0,  1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    group  = np.array([0]*10 + [1]*10)

    r0, r1, metrics = evaluate_fairness(y_true, y_pred, group)
    assert np.isclose(metrics["eo_diff"], 0.0, atol=1e-6), "Equal Opportunity should be satisfied exactly (TPR=0.6667)"
    assert np.isclose(metrics["dp_diff"], 0.20, atol=1e-6), "Demographic Parity difference should be exactly 0.20"

    print("1. Worked Manual Fairness Metric & Conflict Verification (N=20):")
    print(f"   Group 0: Selection Rate={r0['selection_rate']:.4f} | TPR={r0['tpr']:.4f} | FPR={r0['fpr']:.4f}")
    print(f"   Group 1: Selection Rate={r1['selection_rate']:.4f} | TPR={r1['tpr']:.4f} | FPR={r1['fpr']:.4f}")
    print(f"   - Demographic Parity Diff  (|Sel_0 - Sel_1|) : {metrics['dp_diff']:.4f} (NOT equal because base rates differ 60% vs 30%)")
    print(f"   - Equal Opportunity Diff   (|TPR_0 - TPR_1|) : {metrics['eo_diff']:.4f} (SATISFIED perfectly at 0.6667 vs 0.6667!)")
    print(f"   - Equalized Odds Gap       (max |dTPR|,|dFPR|): {metrics['eq_odds_diff']:.4f}")
    print("\n   Takeaway: Equal Opportunity can be 0.0000 while Demographic Parity simultaneously reads 0.2000!")
    print("   No single fairness metric is universally valid across all contexts.")


def main():
    print("=" * 78)
    print("CHAPTER 39: FAIRNESS METRICS & IMPOSSIBILITY TRADE-OFF VERIFICATION")
    print("=" * 78)
    worked_manual_fairness_example()
    print("\n[SUCCESS] Fairness metric verification complete.")


if __name__ == "__main__":
    main()

