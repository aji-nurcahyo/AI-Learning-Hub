"""Experiment Comparison and Best Candidate Selection.

Runs 4 distinct hyperparameter/model configurations, logs them to the local
SQLite experiment tracker, compares validation F1 and ROC-AUC, and selects
the winning candidate reproducibly.
"""

from reproducible_training import train_and_save_pipeline
from experiment_tracking import LocalExperimentTracker


def main():
    print("=" * 75)
    print("CHAPTER 36: EXPERIMENT COMPARISON & CANDIDATE SELECTION")
    print("=" * 75)

    tracker = LocalExperimentTracker()
    exp_name = "telemetry_benchmark_v1"

    configs = [
        ("logistic_regression", {"C": 0.1}, "cand_lr_01.pkl"),
        ("logistic_regression", {"C": 1.0}, "cand_lr_10.pkl"),
        ("random_forest", {"n_estimators": 20}, "cand_rf_20.pkl"),
        ("random_forest", {"n_estimators": 80}, "cand_rf_80.pkl"),
    ]

    logged = []
    for m_type, kw, fname in configs:
        res = train_and_save_pipeline(model_type=m_type, seed=42, artifact_filename=fname, **kw)
        run_id = tracker.log_run(
            experiment_name=exp_name,
            dataset_version=res["dataset_version"],
            params=res["hyperparams"],
            metrics=res["metrics"],
            artifact_path=res["artifact_path"],
            checksum=res["checksum_sha256"],
            env_metadata=res["env_metadata"],
        )
        logged.append((run_id, res))

    # Rank by ROC-AUC descending, then F1 descending
    logged.sort(key=lambda item: (item[1]["metrics"]["roc_auc"], item[1]["metrics"]["f1_score"]), reverse=True)

    print(f"Leaderboard for Experiment '{exp_name}':\n")
    print(f"{'Rank':<5} | {'Run ID':<12} | {'Model & Params':<32} | {'Accuracy':<9} | {'F1':<8} | {'ROC-AUC'}")
    print("-" * 80)

    for rank, (rid, r) in enumerate(logged, 1):
        hp = r["hyperparams"]
        if hp["model_type"] == "logistic_regression":
            desc = f"LogReg (C={hp['C']})"
        else:
            desc = f"RandomForest (n={hp['n_estimators']})"
        m = r["metrics"]
        print(f"{rank:<5} | {rid:<12} | {desc:<32} | {m['accuracy']:<9.4f} | {m['f1_score']:<8.4f} | {m['roc_auc']:.4f}")

    best_id, best_res = logged[0]
    print(f"\nSelected Best Candidate: {best_id} (AUC={best_res['metrics']['roc_auc']:.4f}, File={best_res['artifact_path']})")
    print("[SUCCESS] Experiment comparison completed.")


if __name__ == "__main__":
    main()

