"""Post-Deployment Monitoring Simulation & PSI / KS Drift Assessment."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CapstoneConfig
from dataset_generator import generate_sensor_telemetry
from preprocessing import prepare_splits
from evaluation_and_error_analysis import load_verified_bundle
from baseline_model import evaluate_classifier


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(expected, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.maximum(exp_counts / len(expected), 1e-4)
    act_pct = np.maximum(act_counts / len(actual), 1e-4)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def ks_statistic_2samp(x1: np.ndarray, x2: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov D statistic in pure NumPy."""
    data_all = np.sort(np.concatenate([x1, x2]))
    cdf1 = np.searchsorted(np.sort(x1), data_all, side="right") / len(x1)
    cdf2 = np.searchsorted(np.sort(x2), data_all, side="right") / len(x2)
    return float(np.max(np.abs(cdf1 - cdf2)))


def run_drift_simulation(cfg: CapstoneConfig = CapstoneConfig()) -> dict:
    data = prepare_splits(cfg)
    bundle = load_verified_bundle(cfg)
    model = bundle["model"]

    # Stream A: Nominal telemetry (seed=101, zero shift)
    X_nom_raw, y_nom = generate_sensor_telemetry(n_samples=300, seed=101, temp_shift=0.0, vib_shift=0.0)
    # Stream B: Gradual heating + bearing wear drift (temp +9.5C, vib +1.1 mm/s)
    X_drift_raw, y_drift = generate_sensor_telemetry(n_samples=300, seed=202, temp_shift=9.5, vib_shift=1.1)

    streams = {"Nominal Stream": (X_nom_raw, y_nom), "Drifted Stream (+9.5C/+1.1mm/s)": (X_drift_raw, y_drift)}
    report = {}

    for name, (Xr, yr) in streams.items():
        Xs = bundle["scaler"].transform(Xr)
        metrics = evaluate_classifier(model, Xs, yr)
        psi_per_feat = {
            feat: round(compute_psi(data.X_train_raw[:, idx], Xr[:, idx]), 4)
            for idx, feat in enumerate(cfg.feature_names)
        }
        ks_temp = round(ks_statistic_2samp(data.X_train_raw[:, 0], Xr[:, 0]), 4)
        max_psi = max(psi_per_feat.values())
        report[name] = {
            "metrics": metrics,
            "psi_per_feature": psi_per_feat,
            "max_psi": max_psi,
            "ks_temp_stat": ks_temp,
            "drift_alert": max_psi > cfg.psi_alert_threshold,
        }

    # Save visual PSI comparison chart
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    feats = list(cfg.feature_names)
    x_pos = np.arange(len(feats))
    nom_vals = [report["Nominal Stream"]["psi_per_feature"][f] for f in feats]
    drf_vals = [report["Drifted Stream (+9.5C/+1.1mm/s)"]["psi_per_feature"][f] for f in feats]
    ax.bar(x_pos - 0.18, nom_vals, width=0.35, label="Nominal Stream", color="#16a34a")
    ax.bar(x_pos + 0.18, drf_vals, width=0.35, label="Drifted Stream (+9.5C)", color="#dc2626")
    ax.axhline(cfg.psi_alert_threshold, color="#111111", linestyle="--", label=f"Alert Threshold (PSI={cfg.psi_alert_threshold})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feats, rotation=20, ha="right")
    ax.set_ylabel("Population Stability Index (PSI)")
    ax.set_title("Chapter 40 Capstone: Sensor Drift Monitoring (PSI per Feature)")
    ax.legend()
    fig.tight_layout()
    chart_path = os.path.join(cfg.exp_dir, "capstone_drift_monitoring.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    report["chart_path"] = chart_path
    return report


def main():
    rep = run_drift_simulation()
    print("=" * 80)
    print("CHAPTER 40 CAPSTONE: POST-DEPLOYMENT MONITORING & DRIFT AUDIT")
    print("=" * 80)
    for k, v in rep.items():
        if k == "chart_path":
            continue
        print(
            f"- {k:32s} | Acc={v['metrics']['accuracy']*100:6.2f}% | "
            f"MaxPSI={v['max_psi']:.4f} | KS(Temp)={v['ks_temp_stat']:.4f} | Alert={v['drift_alert']}"
        )


if __name__ == "__main__":
    main()
