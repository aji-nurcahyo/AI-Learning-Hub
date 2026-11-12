"""Controlled Model Rollback Simulation with Human Governance.

Demonstrates safe rollback workflow when monitoring triggers an SLA violation:
1. Registers stable baseline model v1.0.0 and stages it for local serving.
2. Trains & promotes new candidate v1.1.0 to local serving.
3. Simulates monitoring alert: v1.1.0 accuracy drops below SLA (<80%).
4. Blocks automatic unapproved rollback (fail-safe governance).
5. Executes explicit human-approved rollback back to v1.0.0 with checksum verification.
"""

import json
import os
from reproducible_training import train_and_save_pipeline
from local_model_registry import LocalModelRegistry
from model_artifact_validation import safe_load_trusted_artifact


def main():
    print("=" * 75)
    print("CHAPTER 36: CONTROLLED MODEL ROLLBACK SIMULATION")
    print("=" * 75)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    rb_db = os.path.abspath(os.path.join(script_dir, "..", "experiments", "rollback_test_registry.db"))
    if os.path.exists(rb_db):
        os.remove(rb_db)

    registry = LocalModelRegistry(db_path=rb_db)
    model_id = "telemetry_anomaly_cls"

    # 1. Stage stable v1.0.0
    res_v1 = train_and_save_pipeline("logistic_regression", C=1.0, seed=42, artifact_filename="rb_v1.pkl")
    registry.register_model(
        model_id, "v1.0.0", res_v1["dataset_version"], "LogisticRegression",
        res_v1["hyperparams"], res_v1["metrics"], res_v1["artifact_path"], res_v1["checksum_sha256"], "CANDIDATE"
    )
    registry.promote_model(model_id, "v1.0.0", "VALIDATED")
    registry.promote_model(model_id, "v1.0.0", "STAGED_LOCAL_SERVING", human_approved=True)
    print(f"[1] Baseline v1.0.0 staged locally (AUC={res_v1['metrics']['roc_auc']}).")

    # 2. Promote candidate v1.1.0
    res_v2 = train_and_save_pipeline("random_forest", n_estimators=30, seed=42, artifact_filename="rb_v2.pkl")
    registry.register_model(
        model_id, "v1.1.0", res_v2["dataset_version"], "RandomForest",
        res_v2["hyperparams"], res_v2["metrics"], res_v2["artifact_path"], res_v2["checksum_sha256"], "CANDIDATE"
    )
    registry.promote_model(model_id, "v1.1.0", "VALIDATED")
    registry.promote_model(model_id, "v1.1.0", "STAGED_LOCAL_SERVING", human_approved=True)
    active = registry.get_active_local_model(model_id)
    print(f"[2] Upgraded active local model to {active['version']}.")

    # 3. Simulate Monitoring Alert triggering rollback recommendation
    simulated_live_acc = 0.68  # Below 0.80 SLA!
    print(f"\n[3] MONITORING ALERT: Model v1.1.0 rolling accuracy dropped to {simulated_live_acc*100:.1f}% (SLA >= 80%).")
    print("    Recommending rollback to validated version v1.0.0.")

    # 4. Enforce Human Approval Gate on rollback
    try:
        registry.promote_model(model_id, "v1.0.0", "STAGED_LOCAL_SERVING", human_approved=False)
    except PermissionError as e:
        print(f"[4] Governance Gate Intercept: {e}")

    # 5. Execute Human-Approved Rollback
    registry.promote_model(model_id, "v1.0.0", "STAGED_LOCAL_SERVING", human_approved=True)
    restored = registry.get_active_local_model(model_id)
    # Verify artifact checksum before restoring inference
    restored_model = safe_load_trusted_artifact(restored["artifact_path"], restored["checksum_sha256"])

    print(f"[5] ROLLBACK EXECUTED (Human Approved): Active version restored to {restored['version']}.")
    print(f"    Checksum Verified: {restored['checksum_sha256'][:16]}... | Loaded: {type(restored_model).__name__}")

    assert restored["version"] == "v1.0.0"
    print("\n[SUCCESS] Controlled model rollback simulation verified.")


if __name__ == "__main__":
    main()

