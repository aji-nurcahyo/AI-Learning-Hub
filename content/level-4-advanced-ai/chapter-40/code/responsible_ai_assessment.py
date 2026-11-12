"""Capstone Risk Assessment, Risk Register & Standardized Model Card Generation."""

import json
import os
from datetime import date

from config import CapstoneConfig


def build_capstone_risk_register() -> list:
    return [
        {
            "risk_id": "CAP-R1",
            "hazard": "Sensor calibration drift causing silent false negatives",
            "nist_function": "Measure / Manage",
            "likelihood": 4,
            "severity": 4,
            "mitigation": "Scheduled PSI (>0.25) & KS monitoring + human maintenance review",
            "residual_score": 4,
        },
        {
            "risk_id": "CAP-R2",
            "hazard": "Corrupted telemetry payload (NaN/Inf/out-of-bounds)",
            "nist_function": "Map / Manage",
            "likelihood": 4,
            "severity": 5,
            "mitigation": "Strict input schema allowlist & type/range validation before inference",
            "residual_score": 3,
        },
        {
            "risk_id": "CAP-R3",
            "hazard": "Pickle deserialization tampering on model artifact",
            "nist_function": "Govern / Manage",
            "likelihood": 2,
            "severity": 5,
            "mitigation": "Mandatory SHA-256 integrity verification prior to pickle.loads()",
            "residual_score": 2,
        },
        {
            "risk_id": "CAP-R4",
            "hazard": "Over-reliance on synthetic training proxy in real machinery",
            "nist_function": "Govern / Map",
            "likelihood": 4,
            "severity": 5,
            "mitigation": "Explicit Prohibited Use notice + mandatory Human-in-the-Loop gate",
            "residual_score": 6,
        },
    ]


def generate_capstone_model_card(eval_summary: dict, cfg: CapstoneConfig = CapstoneConfig()) -> dict:
    card = {
        "schema_version": "capstone-model-card-v1",
        "date": str(date(2026, 11, 11)),
        "model_name": eval_summary["model_name"],
        "project": cfg.project_name,
        "intended_use": "Educational reference capstone demonstrating end-to-end sensor anomaly classification on CPU.",
        "prohibited_use": [
            "Autonomous physical equipment shutdown without certified hardware safety interlocks.",
            "Deployment on uncalibrated industrial motors without domain validation.",
        ],
        "evaluation_metrics": {k: round(v, 4) for k, v in eval_summary["test_metrics"].items()},
        "confusion_matrix": eval_summary["confusion_matrix"],
        "feature_importance": eval_summary["feature_importance"],
        "risk_register_count": len(build_capstone_risk_register()),
        "certification_disclaimer": "Academic reference artifact ONLY; NOT regulatory compliance or industrial safety certification.",
    }
    os.makedirs(cfg.exp_dir, exist_ok=True)
    out_path = os.path.join(cfg.exp_dir, "capstone_model_card.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    return card


def main():
    reg = build_capstone_risk_register()
    print("=" * 75)
    print("CHAPTER 40 CAPSTONE: RISK REGISTER & RESPONSIBLE AI SUMMARY")
    print("=" * 75)
    for r in reg:
        print(f"- {r['risk_id']} [{r['nist_function']:16s}] Inh={r['likelihood']*r['severity']:2d} -> Res={r['residual_score']:2d} | {r['hazard']}")


if __name__ == "__main__":
    main()

