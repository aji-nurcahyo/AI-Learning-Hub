"""Lightweight Local Experiment Tracker (SQLite + JSON).

Implements local experiment management without requiring external servers:
- Tracks Experiment Name, Run ID, Parameters, Metrics, Dataset Version,
  Code Provenance, Artifact Path, and Checksum.
- Persists runs cleanly to SQLite database in experiments/mlops_tracker.db.
"""

from datetime import datetime, timezone
import json
import os
import sqlite3
import uuid
from typing import Any, Dict, List

from reproducible_training import train_and_save_pipeline


class LocalExperimentTracker:
    """SQLite-backed local experiment tracker."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            exp_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments"))
            os.makedirs(exp_dir, exist_ok=True)
            db_path = os.path.join(exp_dir, "mlops_tracker.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    env_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_run(
        self,
        experiment_name: str,
        dataset_version: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifact_path: str,
        checksum: str,
        env_metadata: Dict[str, str],
    ) -> str:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiment_runs
                (run_id, experiment_name, timestamp_utc, dataset_version, params_json, metrics_json, artifact_path, checksum_sha256, env_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_name,
                    ts,
                    dataset_version,
                    json.dumps(params),
                    json.dumps(metrics),
                    artifact_path,
                    checksum,
                    json.dumps(env_metadata),
                ),
            )
            conn.commit()
        return run_id

    def get_runs(self, experiment_name: str = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if experiment_name:
                rows = conn.execute(
                    "SELECT * FROM experiment_runs WHERE experiment_name = ? ORDER BY timestamp_utc DESC",
                    (experiment_name,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM experiment_runs ORDER BY timestamp_utc DESC").fetchall()

        results = []
        for r in rows:
            results.append({
                "run_id": r["run_id"],
                "experiment_name": r["experiment_name"],
                "timestamp": r["timestamp_utc"],
                "dataset_version": r["dataset_version"],
                "params": json.loads(r["params_json"]),
                "metrics": json.loads(r["metrics_json"]),
                "artifact_path": r["artifact_path"],
                "checksum_sha256": r["checksum_sha256"],
            })
        return results


def main():
    print("=" * 75)
    print("CHAPTER 36: LOCAL EXPERIMENT TRACKING (SQLITE BACKED)")
    print("=" * 75)

    tracker = LocalExperimentTracker()

    # Run 1: Logistic Regression
    res = train_and_save_pipeline(model_type="logistic_regression", C=1.0, seed=42, artifact_filename="model_logreg_c1.pkl")
    run_id = tracker.log_run(
        experiment_name="telemetry_anomaly_cls",
        dataset_version=res["dataset_version"],
        params=res["hyperparams"],
        metrics=res["metrics"],
        artifact_path=res["artifact_path"],
        checksum=res["checksum_sha256"],
        env_metadata=res["env_metadata"],
    )

    print(f"Logged Run ID   : {run_id}")
    print(f"Experiment      : telemetry_anomaly_cls")
    print(f"Parameters      : {res['hyperparams']}")
    print(f"Metrics         : {res['metrics']}")
    print(f"Tracker DB Path : {tracker.db_path}")

    runs = tracker.get_runs("telemetry_anomaly_cls")
    print(f"\nTotal logged runs in DB: {len(runs)}")
    print("[SUCCESS] Local experiment tracking operational.")


if __name__ == "__main__":
    main()

