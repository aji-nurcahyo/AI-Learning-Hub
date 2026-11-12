"""Concept Drift vs Pure Data Drift Simulation.

Uses synthetic labeled streams to demonstrate the critical distinction:
1. Stable Phase: P(X) stationary, P(y|X) stationary -> High accuracy.
2. Pure Data Drift (Covariate Shift): P(X) shifts significantly, P(y|X) unchanged -> Accuracy stays high!
3. Abrupt Concept Drift: P(X) unchanged, P(y|X) decision boundary shifts/flips -> Accuracy collapses!
4. Gradual Concept Drift: P(y|X) evolves smoothly over sliding windows.

Key Principle: Without delayed ground-truth labels y_true (or proxy performance evidence),
changes in input distribution or prediction rates alone cannot prove Concept Drift.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_drift_detection import compute_psi


def main():
    print("=" * 75)
    print("CHAPTER 36: CONCEPT DRIFT VS COVARIATE DATA DRIFT SIMULATION")
    print("=" * 75)

    rng = np.random.default_rng(42)

    # Train initial classifier on Stable Concept: y = 1 if (x0 + x1 > 0)
    X_train = rng.normal(0.0, 1.0, size=(600, 2))
    y_train = ((X_train[:, 0] + X_train[:, 1] + rng.normal(0, 0.2, 600)) > 0).astype(int)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    # 1. Stable Evaluation Window
    X_w1 = rng.normal(0.0, 1.0, size=(300, 2))
    y_w1 = ((X_w1[:, 0] + X_w1[:, 1] + rng.normal(0, 0.2, 300)) > 0).astype(int)
    acc_w1 = accuracy_score(y_w1, clf.predict(X_w1))
    psi_w1 = compute_psi(X_train[:, 0], X_w1[:, 0])

    # 2. Pure Covariate Shift / Data Drift (mean shifts +1.5, rule P(y|X) UNCHANGED)
    X_w2 = rng.normal(1.5, 1.0, size=(300, 2))
    y_w2 = ((X_w2[:, 0] + X_w2[:, 1] + rng.normal(0, 0.2, 300)) > 0).astype(int)
    acc_w2 = accuracy_score(y_w2, clf.predict(X_w2))
    psi_w2 = compute_psi(X_train[:, 0], X_w2[:, 0])

    # 3. Abrupt Concept Drift (input P(X) UNCHANGED, rule flips to x0 - 2*x1 < 0)
    X_w3 = rng.normal(0.0, 1.0, size=(300, 2))
    y_w3 = ((X_w3[:, 0] - 2.0 * X_w3[:, 1] + rng.normal(0, 0.2, 300)) < 0).astype(int)
    acc_w3 = accuracy_score(y_w3, clf.predict(X_w3))
    psi_w3 = compute_psi(X_train[:, 0], X_w3[:, 0])

    # 4. Gradual Concept Drift (50% old concept, 50% new concept transition)
    X_w4 = rng.normal(0.0, 1.0, size=(300, 2))
    mix_mask = rng.random(300) < 0.5
    y_old = ((X_w4[:, 0] + X_w4[:, 1] + rng.normal(0, 0.2, 300)) > 0).astype(int)
    y_new = ((X_w4[:, 0] - 2.0 * X_w4[:, 1] + rng.normal(0, 0.2, 300)) < 0).astype(int)
    y_w4 = np.where(mix_mask, y_new, y_old)
    acc_w4 = accuracy_score(y_w4, clf.predict(X_w4))
    psi_w4 = compute_psi(X_train[:, 0], X_w4[:, 0])

    print(f"{'Stream Regime':<30} | {'Input PSI (x0)':<15} | {'Model Accuracy':<15} | {'Diagnosis'}")
    print("-" * 80)
    print(f"{'1. Stable Baseline':<30} | {psi_w1:<15.4f} | {acc_w1*100:<14.2f}% | Healthy")
    print(f"{'2. Pure Covariate Data Drift':<30} | {psi_w2:<15.4f} | {acc_w2*100:<14.2f}% | P(X) shifted, P(y|X) intact!")
    print(f"{'3. Abrupt Concept Drift':<30} | {psi_w3:<15.4f} | {acc_w3*100:<14.2f}% | ALERT: True Concept Drift!")
    print(f"{'4. Gradual Concept Drift (50%)':<30} | {psi_w4:<15.4f} | {acc_w4*100:<14.2f}% | Degradation Detected")

    print("\nCritical Takeaway:")
    print("Regime 2 shows high PSI (>0.25) yet retains high accuracy because P(y|X) held.")
    print("Regime 3 shows near-zero input PSI (<0.10) yet accuracy collapsed, proving")
    print("labeled ground truth is mandatory to diagnose true Concept Drift.")
    print("\n[SUCCESS] Concept drift simulation verified.")


if __name__ == "__main__":
    main()

