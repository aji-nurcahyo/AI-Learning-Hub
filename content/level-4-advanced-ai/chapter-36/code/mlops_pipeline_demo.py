"""End-to-End Local MLOps Pipeline Demonstration.

Unifies all core components of Chapter 36 into one reproducible execution flow:
1. Reproducible Training Pipeline (zero leakage, fixed seed).
2. SQLite Experiment Tracking & Candidate Logging.
3. Model Versioning & Registry Staging (human-approved).
4. Cryptographic Artifact Verification before loading.
5. Localhost (127.0.0.1) REST API Verification & Shutdown.
6. Drift Detection (PSI / KS) over synthetic evaluation window.
"""

import json
import os
import urllib.request
import numpy as np

from reproducible_training import train_and_save_pipeline
from experiment_tracking import LocalExperimentTracker
from local_model_registry import LocalModelRegistry
from model_artifact_validation import safe_load_trusted_artifact
from model_serving_api import start_local_serving_server
from data_drift_detection import compute_psi, compute_ks_2samp


def main():
    print("=" * 75)
    print("CHAPTER 36: COMPLETE END-TO-END LOCAL MLOPS PIPELINE")
    print("=" * 75)

    # 1. Train Reproducibly
    print("[Stage 1/5] Running leak-free training pipeline...")
    res = train_and_save_pipeline("random_forest", n_estimators=40, seed=42, artifact_filename="e2e_model_v1.pkl")
    print(f"            Dataset: {res['dataset_version']} | AUC={res['metrics']['roc_auc']:.4f} | SHA256={res['checksum_sha256'][:12]}...")

    # 2. Log Experiment
    print("[Stage 2/5] Logging run to SQLite Experiment Tracker...")
    tracker = LocalExperimentTracker()
    run_id = tracker.log_run(
        "e2e_mlops_demo", res["dataset_version"], res["hyperparams"],
        res["metrics"], res["artifact_path"], res["checksum_sha256"], res["env_metadata"]
    )
    print(f"            Logged Run ID: {run_id}")

    # 3. Register & Stage Model
    print("[Stage 3/5] Registering and promoting model in Local Registry...")
    registry = LocalModelRegistry()
    registry.register_model(
        "e2e_telemetry_clf", "v1.0.0", res["dataset_version"], "RandomForestClassifier",
        res["hyperparams"], res["metrics"], res["artifact_path"], res["checksum_sha256"], "CANDIDATE"
    )
    registry.promote_model("e2e_telemetry_clf", "v1.0.0", "VALIDATED")
    registry.promote_model("e2e_telemetry_clf", "v1.0.0", "STAGED_LOCAL_SERVING", human_approved=True)
    staged = registry.get_active_local_model("e2e_telemetry_clf")
    _ = safe_load_trusted_artifact(staged["artifact_path"], staged["checksum_sha256"])
    print(f"            Active Staged Version: {staged['version']} (Checksum Verified)")

    # 4. Serve Locally on 127.0.0.1 & Test Inference
    print("[Stage 4/5] Starting 127.0.0.1 REST API for live inference check...")
    server, port, _ = start_local_serving_server(port=0)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/predict",
            data=json.dumps({"features": [0.4, -0.9, 0.7, -0.2, 0.1, 0.5]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            api_out = json.loads(resp.read().decode("utf-8"))
        print(f"            API Response (127.0.0.1:{port}): Pred={api_out['prediction']}, Prob={api_out['probability_positive']}, Latency={api_out['latency_ms']}ms")
    finally:
        server.shutdown()
        server.server_close()

    # 5. Monitor Feature Drift
    print("[Stage 5/5] Checking feature drift vs reference baseline...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(res["artifact_path"]), "..", "data", "reference_split.npz"))
    X_ref = np.load(data_path)["X_test"]
    rng = np.random.default_rng(77)
    X_curr = X_ref + rng.normal(0.1, 0.05, size=X_ref.shape)
    psi_f0 = compute_psi(X_ref[:, 0], X_curr[:, 0])
    ks_d, ks_p = compute_ks_2samp(X_ref[:, 0], X_curr[:, 0])
    print(f"            Feature 0 Monitoring -> PSI={psi_f0:.4f}, KS D={ks_d:.4f} (p={ks_p:.4f})")

    print("\n[SUCCESS] End-to-end local MLOps pipeline executed cleanly.")


if __name__ == "__main__":
    main()

