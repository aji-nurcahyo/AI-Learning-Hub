"""Observability & Monitoring Dashboard Generator.

Aggregates actual simulation telemetry across 5 operational windows:
1. Feature Data Drift (PSI per window).
2. Concept Drift Rolling Accuracy (when ground truth arrives).
3. Inference Latency Distribution histogram (ms).
Saves visual dashboard to experiments/monitoring_dashboard.png.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_drift_detection import compute_psi


def main():
    print("=" * 75)
    print("CHAPTER 36: MLOPS OBSERVABILITY & MONITORING DASHBOARD")
    print("=" * 75)

    rng = np.random.default_rng(42)
    X_ref = rng.normal(0.0, 1.0, size=(500, 2))
    y_ref = ((X_ref[:, 0] + X_ref[:, 1] + rng.normal(0, 0.25, 500)) > 0).astype(int)

    model = LogisticRegression(random_state=42).fit(X_ref, y_ref)

    # Simulate 5 time windows: W1-W2 stable, W3 covariate drift, W4-W5 concept drift
    window_names = ["W1 (Base)", "W2 (Norm)", "W3 (CovShift)", "W4 (Concept1)", "W5 (Concept2)"]
    psi_vals = []
    acc_vals = []

    for idx in range(5):
        if idx < 2:
            Xw = rng.normal(0.0, 1.0, size=(200, 2))
            yw = ((Xw[:, 0] + Xw[:, 1] + rng.normal(0, 0.25, 200)) > 0).astype(int)
        elif idx == 2:
            # Pure Covariate Shift: mean +1.4, same decision boundary
            Xw = rng.normal(1.4, 1.0, size=(200, 2))
            yw = ((Xw[:, 0] + Xw[:, 1] + rng.normal(0, 0.25, 200)) > 0).astype(int)
        elif idx == 3:
            # Partial Concept Drift
            Xw = rng.normal(0.5, 1.0, size=(200, 2))
            yw = ((Xw[:, 0] - 0.8 * Xw[:, 1] + rng.normal(0, 0.25, 200)) > 0).astype(int)
        else:
            # Severe Concept Drift
            Xw = rng.normal(0.5, 1.0, size=(200, 2))
            yw = ((Xw[:, 0] + Xw[:, 1] + rng.normal(0, 0.25, 200)) < 0).astype(int)

        psi_vals.append(compute_psi(X_ref[:, 0], Xw[:, 0]))
        acc_vals.append(round(float(accuracy_score(yw, model.predict(Xw))) * 100, 1))

    # Measure 300 in-memory latencies (ms)
    latencies = []
    sample = X_ref[:1]
    for _ in range(300):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    lat_arr = np.array(latencies)
    p95_ms = float(np.percentile(lat_arr, 95))

    print(f"{'Window':<15} | {'PSI (x0)':<10} | {'Accuracy (%)':<14} | {'Status'}")
    print("-" * 65)
    for w, p, a in zip(window_names, psi_vals, acc_vals):
        status = "CRITICAL ALERT" if a < 75.0 else ("DRIFT WARNING" if p >= 0.25 else "OK")
        print(f"{w:<15} | {p:<10.4f} | {a:<14.1f} | {status}")
    print(f"\nInference P95 Latency: {p95_ms:.3f} ms (Mean: {np.mean(lat_arr):.3f} ms)")

    # Plot 3-panel monitoring dashboard
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].bar(window_names, psi_vals, color=["steelblue", "steelblue", "orange", "crimson", "crimson"])
    axes[0].axhline(0.25, color="red", linestyle="--", label="Alert Threshold (0.25)")
    axes[0].set_title("Feature Data Drift (PSI)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(window_names, acc_vals, marker="o", linewidth=2, color="darkgreen")
    axes[1].axhline(80.0, color="red", linestyle="--", label="SLA Min (80%)")
    axes[1].set_title("Rolling Accuracy (Concept Drift)")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(lat_arr, bins=25, color="teal", alpha=0.8)
    axes[2].axvline(p95_ms, color="red", linestyle="--", label=f"P95: {p95_ms:.3f} ms")
    axes[2].set_title("CPU Inference Latency (ms)")
    axes[2].set_xlabel("Latency (ms)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.abspath(os.path.join(script_dir, "..", "experiments", "monitoring_dashboard.png"))
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close(fig)

    print(f"Saved Monitoring Dashboard plot to: {out_img}")
    print("[SUCCESS] Monitoring dashboard generated cleanly.")


if __name__ == "__main__":
    main()

