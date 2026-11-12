"""Unified End-to-End Orchestrator for Chapter 40 Capstone Reference Project."""

import json
import os
import time

from config import CapstoneConfig
from dataset_generator import save_data_dictionary
from preprocessing import prepare_splits
from baseline_model import evaluate_classifier
from candidate_models import compare_and_save_best
from evaluation_and_error_analysis import run_final_test_evaluation
from local_inference import predict_sensor_reading
from monitoring_and_drift import run_drift_simulation
from responsible_ai_assessment import build_capstone_risk_register, generate_capstone_model_card
from sklearn.dummy import DummyClassifier


def main():
    t0 = time.perf_counter()
    cfg = CapstoneConfig()
    print("=" * 80)
    print("CHAPTER 40 CAPSTONE: END-TO-END INTELLIGENT SENSOR MONITORING PIPELINE")
    print("=" * 80)

    dict_path = save_data_dictionary(cfg)
    data = prepare_splits(cfg)
    dummy = DummyClassifier(strategy="most_frequent").fit(data.X_train, data.y_train)
    dummy_val = evaluate_classifier(dummy, data.X_val, data.y_val)

    cand_summary = compare_and_save_best(cfg)
    eval_summary = run_final_test_evaluation(cfg)
    drift_summary = run_drift_simulation(cfg)
    risk_register = build_capstone_risk_register()
    model_card = generate_capstone_model_card(eval_summary, cfg)

    sample_inf = predict_sensor_reading({
        "temperature_c": 75.0, "vibration_rms": 3.8, "current_amp": 20.0,
        "acoustic_db": 77.0, "rpm_deviation": -50.0, "voltage_ripple_v": 2.2
    }, cfg=cfg)

    duration_s = round(time.perf_counter() - t0, 3)
    final_report = {
        "project": cfg.project_name,
        "execution_time_seconds": duration_s,
        "data_dictionary_path": dict_path,
        "splits": {"train": int(len(data.y_train)), "val": int(len(data.y_val)), "test": int(len(data.y_test))},
        "dummy_baseline_val": dummy_val,
        "candidate_comparison": cand_summary["candidates"],
        "selected_model": eval_summary["model_name"],
        "sha256_prefix": cand_summary["sha256"][:16],
        "heldout_test_metrics": eval_summary["test_metrics"],
        "confusion_matrix": eval_summary["confusion_matrix"],
        "feature_importance": eval_summary["feature_importance"],
        "error_count_on_test": eval_summary["error_count"],
        "sample_local_inference": sample_inf,
        "drift_monitoring": {k: v for k, v in drift_summary.items() if k != "chart_path"},
        "risk_register_items": len(risk_register),
        "model_card_schema": model_card["schema_version"],
    }

    report_path = os.path.join(cfg.exp_dir, "capstone_report_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"Selected Model    : {eval_summary['model_name']} (Val-selected; SHA-256={cand_summary['sha256'][:16]}...)")
    print(f"Dummy Baseline    : Val Acc={dummy_val['accuracy']*100:.2f}%, F1={dummy_val['f1']:.4f}")
    tm = eval_summary["test_metrics"]
    print(f"Held-Out Test     : Acc={tm['accuracy']*100:.2f}%, F1={tm['f1']:.4f}, ROC-AUC={tm['roc_auc']:.4f} (Errors={eval_summary['error_count']}/300)")
    print(f"Drift Audit       : Nominal MaxPSI={drift_summary['Nominal Stream']['max_psi']:.4f} (Alert={drift_summary['Nominal Stream']['drift_alert']}) | "
          f"Drifted MaxPSI={drift_summary['Drifted Stream (+9.5C/+1.1mm/s)']['max_psi']:.4f} (Alert={drift_summary['Drifted Stream (+9.5C/+1.1mm/s)']['drift_alert']})")
    print(f"Risk & Model Card : {len(risk_register)} Risks Logged | Model Card Saved")
    print(f"Total Runtime     : {duration_s}s on local CPU")
    print(f"Saved Report JSON : {report_path}")


if __name__ == "__main__":
    main()

