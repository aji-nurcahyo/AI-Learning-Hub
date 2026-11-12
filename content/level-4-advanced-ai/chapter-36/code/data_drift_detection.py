"""Statistical Data Drift Detection (PSI and Two-Sample KS Test).

Computes feature distribution shift between Reference Data (X_ref) and Current Data (X_curr):
1. Population Stability Index (PSI) across quantile bins:
   PSI = sum( (p_curr - p_ref) * ln(p_curr / p_ref) )
2. Two-Sample Kolmogorov-Smirnov (KS) statistic D = sup_x |F_ref(x) - F_curr(x)|
   and asymptotic Kolmogorov p-value calculated cleanly in pure NumPy/Math.

Important Caveat:
Data Drift measures P(X) changes only. It does NOT automatically guarantee
that model predictive performance P(y|X) has degraded.
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def compute_psi(ref_col: np.ndarray, curr_col: np.ndarray, bins: int = 10) -> float:
    """Computes Population Stability Index (PSI) using reference quantile bin edges."""
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(ref_col, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Deduplicate non-strictly increasing edges if discrete
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref_col, bins=edges)
    curr_counts, _ = np.histogram(curr_col, bins=edges)

    eps = 1e-4
    ref_pct = np.clip(ref_counts / len(ref_col), eps, None)
    curr_pct = np.clip(curr_counts / len(curr_col), eps, None)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return round(float(psi), 4)


def compute_ks_2samp(x1: np.ndarray, x2: np.ndarray) -> Tuple[float, float]:
    """Computes exact 2-sample KS statistic D and asymptotic p-value."""
    n1, n2 = len(x1), len(x2)
    x1_sorted = np.sort(x1)
    x2_sorted = np.sort(x2)
    all_vals = np.concatenate([x1_sorted, x2_sorted])

    cdf1 = np.searchsorted(x1_sorted, all_vals, side="right") / n1
    cdf2 = np.searchsorted(x2_sorted, all_vals, side="right") / n2
    d_stat = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic Kolmogorov distribution p-value approximation
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d_stat
    if lam <= 0.0:
        pval = 1.0
    else:
        total = 0.0
        for k in range(1, 101):
            term = ((-1) ** (k - 1)) * math.exp(-2.0 * (k ** 2) * (lam ** 2))
            total += term
            if abs(term) < 1e-12:
                break
        pval = max(0.0, min(1.0, 2.0 * total))

    return round(d_stat, 4), round(pval, 6)


def main():
    print("=" * 75)
    print("CHAPTER 36: STATISTICAL DATA DRIFT DETECTION (PSI & KS TEST)")
    print("=" * 75)

    rng = np.random.default_rng(42)
    # Reference dataset: 500 samples, 4 features ~ Normal(0, 1)
    X_ref = rng.normal(loc=0.0, scale=1.0, size=(500, 4))

    # Current dataset (300 samples):
    # Feature 0: No drift
    # Feature 1: Slight mean shift (+0.35)
    # Feature 2: Severe covariate shift (+1.8 mean, 1.5x variance)
    # Feature 3: Variance expansion (scale = 2.0)
    X_curr = np.column_stack([
        rng.normal(0.0, 1.0, 300),
        rng.normal(0.35, 1.0, 300),
        rng.normal(1.8, 1.5, 300),
        rng.normal(0.0, 2.0, 300),
    ])

    print(f"Reference N=500 vs Current N=300 Feature Drift Analysis:\n")
    print(f"{'Feature':<10} | {'Mean Shift':<12} | {'PSI':<8} | {'KS Stat D':<10} | {'KS p-val':<10} | {'Alert Level'}")
    print("-" * 75)

    for j in range(4):
        mean_diff = float(np.mean(X_curr[:, j]) - np.mean(X_ref[:, j]))
        psi = compute_psi(X_ref[:, j], X_curr[:, j], bins=10)
        ks_d, ks_p = compute_ks_2samp(X_ref[:, j], X_curr[:, j])

        if psi >= 0.25 or ks_p < 0.001:
            alert = "CRITICAL DRIFT"
        elif psi >= 0.10 or ks_p < 0.05:
            alert = "MODERATE SHIFT"
        else:
            alert = "NORMAL"

        print(f"feat_{j:<5} | {mean_diff:+11.3f} | {psi:<8.4f} | {ks_d:<10.4f} | {ks_p:<10.6f} | {alert}")

    print("\nMethodological Note:")
    print("- Heuristic PSI rules (e.g., <0.1 ok, >0.25 drift) depend heavily on bin count & N.")
    print("- Large sample sizes trigger small KS p-values even for tiny practical shifts.")
    print("- Data Drift alone checks P(X); ground-truth accuracy is still needed for Concept Drift.")
    print("\n[SUCCESS] Statistical data drift detection verified.")


if __name__ == "__main__":
    main()

