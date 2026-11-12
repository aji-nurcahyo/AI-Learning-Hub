"""Synthetic Dataset Bias Analysis (Representation, Sampling & Measurement Noise).

Generates an educational synthetic dataset (`Group A` vs `Group B`) with deliberate
sampling imbalance and noisy proxy measurement to illustrate bias auditing safely
without making conclusions about any real-world human population.
"""

import numpy as np


def generate_synthetic_bias_dataset(seed: int = 42, n_samples: int = 1000):
    """Generates synthetic tabular features with controlled group imbalance & proxy bias."""
    rng = np.random.default_rng(seed)

    # Protected attribute: 0 = Synthetic Group A (75% majority), 1 = Synthetic Group B (25% minority)
    group = rng.choice([0, 1], size=n_samples, p=[0.75, 0.25])

    # Latent qualification score ~ N(0, 1) identically distributed for both groups
    true_merit = rng.normal(0.0, 1.0, size=n_samples)

    # Observed feature 1 (objective score): slightly correlated with merit + noise
    f_score = true_merit + rng.normal(0.0, 0.35, size=n_samples)

    # Observed feature 2 (biased proxy measurement): Group B has systematic measurement shift -0.6
    f_proxy = true_merit - 0.6 * group + rng.normal(0.0, 0.45, size=n_samples)

    # Historical/Label bias: baseline positive selection threshold slightly higher for Group B in training logs
    noisy_decision = true_merit + rng.normal(0.0, 0.3, size=n_samples) - 0.45 * group
    y = (noisy_decision > 0.2).astype(int)

    X = np.column_stack([f_score, f_proxy])
    return X, y, group, true_merit


def main():
    print("=" * 75)
    print("CHAPTER 39: SYNTHETIC DATASET BIAS ANALYSIS")
    print("=" * 75)
    print("Notice: Synthetic Group A & Group B are purely mathematical constructs.\n")

    X, y, group, _ = generate_synthetic_bias_dataset(seed=42, n_samples=1000)

    for g_id, g_name in [(0, "Group A (Majority)"), (1, "Group B (Minority)")]:
        mask = group == g_id
        count = int(mask.sum())
        share = 100.0 * count / len(group)
        pos_rate = 100.0 * y[mask].mean()
        mean_score = float(X[mask, 0].mean())
        mean_proxy = float(X[mask, 1].mean())
        print(
            f"{g_name:20s} | n={count:3d} ({share:5.1f}%) | "
            f"Positive Label Rate={pos_rate:5.2f}% | "
            f"Mean Score={mean_score:+.3f} | Mean Biased Proxy={mean_proxy:+.3f}"
        )

    rep_ratio = (group == 1).sum() / (group == 0).sum()
    label_gap = float(y[group == 0].mean() - y[group == 1].mean())
    print(f"\nRepresentation Ratio (B / A) : {rep_ratio:.3f}")
    print(f"Positive Label Rate Gap      : {label_gap * 100:.2f} percentage points (due to injected label shift)")
    assert abs(X[group == 0, 0].mean() - X[group == 1, 0].mean()) < 0.15
    assert abs(X[group == 0, 1].mean() - X[group == 1, 1].mean()) > 0.35
    print("\n[SUCCESS] Dataset bias audit verified representation & measurement disparity.")


if __name__ == "__main__":
    main()

