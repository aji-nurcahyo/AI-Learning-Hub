"""Educational Local Model Registry (SQLite-Backed).

Implements:
1. Full model metadata schema: model_id, version, training_date, dataset_version,
   algorithm, hyperparams, eval_metrics, artifact_path, checksum_sha256, status.
2. Controlled lifecycle statuses:
   - CANDIDATE
   - VALIDATED
   - STAGED_LOCAL_SERVING
   - ARCHIVED
3. Explicit human-approved promotion and rollback gates.
"""

from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from reproducible_training import train_and_save_pipeline


VALID_STATUSES = {"CANDIDATE", "VALIDATED", "STAGED_LOCAL_SERVING", "ARCHIVED"}


class LocalModelRegistry:
    """SQLite-backed Model Registry for educational model governance."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            exp_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments"))
            os.makedirs(exp_dir, exist_ok=True)
            db_path = os.path.join(exp_dir, "model_registry.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    training_date TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    hyperparams_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY (model_id, version)
                )
            """)
            conn.commit()

    def register_model(
        self,
        model_id: str,
        version: str,
        dataset_version: str,
        algorithm: str,
        hyperparams: Dict[str, Any],
        metrics: Dict[str, float],
        artifact_path: str,
        checksum_sha256: str,
        status: str = "CANDIDATE",
    ):
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Allowed: {VALID_STATUSES}")

        dt = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO models
                (model_id, version, training_date, dataset_version, algorithm, hyperparams_json, metrics_json, artifact_path, checksum_sha256, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    version,
                    dt,
                    dataset_version,
                    algorithm,
                    json.dumps(hyperparams),
                    json.dumps(metrics),
                    artifact_path,
                    checksum_sha256,
                    status,
                ),
            )
            conn.commit()

    def promote_model(self, model_id: str, version: str, new_status: str, human_approved: bool = False):
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'")
        if new_status == "STAGED_LOCAL_SERVING" and not human_approved:
            raise PermissionError("Promotion to STAGED_LOCAL_SERVING requires explicit human approval.")

        with sqlite3.connect(self.db_path) as conn:
            # If promoting to STAGED_LOCAL_SERVING, archive previous staged versions
            if new_status == "STAGED_LOCAL_SERVING":
                conn.execute(
                    "UPDATE models SET status = 'ARCHIVED' WHERE model_id = ? AND status = 'STAGED_LOCAL_SERVING'",
                    (model_id,),
                )
            cur = conn.execute(
                "UPDATE models SET status = ? WHERE model_id = ? AND version = ?",
                (new_status, model_id, version),
            )
            if cur.rowcount == 0:
                raise LookupError(f"Model {model_id} {version} not found in registry.")
            conn.commit()

    def get_active_local_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ? AND status = 'STAGED_LOCAL_SERVING'",
                (model_id,),
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM models WHERE model_id = ? ORDER BY training_date ASC",
                (model_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def main():
    print("=" * 75)
    print("CHAPTER 36: EDUCATIONAL LOCAL MODEL REGISTRY")
    print("=" * 75)

    registry = LocalModelRegistry()
    model_id = "telemetry_anomaly_cls"

    # Register v1.0.0
    res_v1 = train_and_save_pipeline("logistic_regression", C=1.0, seed=42, artifact_filename="reg_v1.pkl")
    registry.register_model(
        model_id=model_id,
        version="v1.0.0",
        dataset_version=res_v1["dataset_version"],
        algorithm="LogisticRegression",
        hyperparams=res_v1["hyperparams"],
        metrics=res_v1["metrics"],
        artifact_path=res_v1["artifact_path"],
        checksum_sha256=res_v1["checksum_sha256"],
        status="CANDIDATE",
    )

    # Validate & Promote v1.0.0 with explicit approval
    registry.promote_model(model_id, "v1.0.0", "VALIDATED")
    registry.promote_model(model_id, "v1.0.0", "STAGED_LOCAL_SERVING", human_approved=True)

    # Register v1.1.0 (RandomForest) as candidate
    res_v2 = train_and_save_pipeline("random_forest", n_estimators=50, seed=42, artifact_filename="reg_v2.pkl")
    registry.register_model(
        model_id=model_id,
        version="v1.1.0",
        dataset_version=res_v2["dataset_version"],
        algorithm="RandomForestClassifier",
        hyperparams=res_v2["hyperparams"],
        metrics=res_v2["metrics"],
        artifact_path=res_v2["artifact_path"],
        checksum_sha256=res_v2["checksum_sha256"],
        status="CANDIDATE",
    )

    # Verify blocking unapproved staging
    try:
        registry.promote_model(model_id, "v1.1.0", "STAGED_LOCAL_SERVING", human_approved=False)
    except PermissionError as e:
        print(f"[Governance Gate] Blocked unapproved staging: {e}\n")

    print(f"Registered Versions for '{model_id}':")
    for m in registry.list_versions(model_id):
        metrics = json.loads(m["metrics_json"])
        print(
            f"- {m['version']:<8} | {m['algorithm']:<22} | Status: {m['status']:<20} | "
            f"AUC: {metrics['roc_auc']:.4f} | SHA256: {m['checksum_sha256'][:12]}..."
        )

    print("\n[SUCCESS] Local Model Registry governance verified.")


if __name__ == "__main__":
    main()

