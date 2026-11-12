"""Prediction Distribution and Confidence Monitoring.

Tracks changes in model outputs (Prediction Drift) when ground-truth labels are delayed:
1. Positive class prediction ratio shift vs baseline training distribution.
2. Mean and low-confidence (near 0.50 uncertainty) fraction monitoring.
3. KS test applied directly to predicted probability outputs P(y=1|X).
"""

import numpy as np
from reproducible_training import train_and_save_pipeline
from model_artifact_validation import safe_load_trusted_artifact
from data_drift_detection import compute_ks_2samp


def main():
    print("=" * 75)
    print("CHAPTER 36: PREDICTION DRIFT & CONFIDENCE MONITORING")
    print("=" * 75)

    res = train_and_save_pipeline("logistic_regression", C=1.0, seed=42, artifact_filename="pred_mon.pkl")
    model = safe_load_trusted_artifact(res["artifact_path"], res["checksum_sha256"])

    # Load reference test split
    import os
    data_path = os.path.abspath(os.path.join(os.path.dirname(res["artifact_path"]), "..", "data", "reference_split.npz"))
    ref_npz = np.load(data_path)
    X_ref = ref_npz["X_test"]

    ref_probs = model.predict_proba(X_ref)[:, 1]
    ref_preds = model.predict(X_ref)
    ref_pos_rate = float(np.mean(ref_preds))
    ref_uncertain = float(np.mean((ref_probs >= 0.4) & (ref_probs <= 0.6)))

    # Simulate drifted incoming live stream (shift features)
    rng = np.random.default_rng(99)
    X_live = X_ref.copy()
    X_live[:, 0] += 1.5
    X_live[:, 2] -= 1.2

    live_probs = model.predict_proba(X_live)[:, 1]
    live_preds = model.predict(X_live)
    live_pos_rate = float(np.mean(live_preds))
    live_uncertain = float(np.mean((live_probs >= 0.4) & (live_probs <= 0.6)))

    ks_d, ks_p = compute_ks_2samp(ref_probs, live_probs)

    print("1. Reference Window Prediction Telemetry:")
    print(f"   - Positive Class Rate : {ref_pos_rate*100:.2f}%")
    print(f"   - Uncertain Band Ratio: {ref_uncertain*100:.2f}% (prob in [0.40, 0.60])")

    print("\n2. Live Drifted Stream Prediction Telemetry:")
    print(f"   - Positive Class Rate : {live_pos_rate*100:.2f}% (Delta: {(live_pos_rate - ref_pos_rate)*100:+.2f}%)")
    print(f"   - Uncertain Band Ratio: {live_uncertain*100:.2f}%")
    print(f"   - Prob KS Test vs Ref : D = {ks_d:.4f}, p-value = {ks_p:.6f}")

    if ks_p < 0.01:
        print("\n[ALERT] Significant Prediction Distribution Drift detected (p < 0.01)!")
    else:
        print("\n[OK] Prediction distribution remains within baseline bounds.")

    print("[SUCCESS] Prediction monitoring verified.")


if __name__ == "__main__":
    main()

