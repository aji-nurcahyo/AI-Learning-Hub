"""Standardized Model Card Generator (Mitchell et al., 2019 Structure)."""

import json
import os
from datetime import date


def build_model_card(
    model_name: str,
    version: str,
    intended_use: str,
    prohibited_uses: list,
    overall_accuracy: float,
    subgroup_metrics: dict,
    limitations: list,
) -> dict:
    return {
        "schema_version": "model-card-edu-v1",
        "generated_date": str(date(2026, 11, 11)),
        "model_details": {
            "name": model_name,
            "version": version,
            "architecture": "LogisticRegression / RandomForest (CPU Reference)",
            "owner": "ITERA AI Learning Hub Coursework Team",
            "license": "CC-BY-4.0 (Educational Use Only)",
        },
        "intended_use": {
            "primary_domain": intended_use,
            "out_of_scope_and_prohibited": prohibited_uses,
        },
        "datasets_and_evaluation": {
            "dataset_type": "Controlled Synthetic Tabular Corpus (zero personal data)",
            "overall_accuracy": round(overall_accuracy, 4),
            "subgroup_breakdown": subgroup_metrics,
        },
        "ethical_and_safety_considerations": {
            "human_oversight": "Mandatory human review required for uncertain or adverse decisions.",
            "regulatory_status": "Educational artifact only; NOT formally certified under ISO/IEC 42001 or statutory audits.",
            "limitations": limitations,
        },
    }


def main():
    print("=" * 75)
    print("CHAPTER 39: MODEL CARD GENERATOR")
    print("=" * 75)

    card = build_model_card(
        model_name="SyntheticScreeningBaseline",
        version="1.0.0",
        intended_use="Demonstrating group fairness & robustness diagnostics in academic workshops.",
        prohibited_uses=[
            "Real-world scholarship, hiring, medical, or credit eligibility decisions.",
            "Deployment without domain validation and human oversight.",
        ],
        overall_accuracy=0.8389,
        subgroup_metrics={
            "Group A (Synthetic Majority)": {"accuracy": 0.859, "tpr": 0.802, "fpr": 0.114},
            "Group B (Synthetic Minority)": {"accuracy": 0.778, "tpr": 0.562, "fpr": 0.145},
        },
        limitations=[
            "Trained on synthetic distributions containing injected proxy measurement shift.",
            "Subgroup accuracy disparity observed; requires mitigation before any practical trial.",
        ],
    )

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_model_card.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    assert "intended_use" in card and "out_of_scope_and_prohibited" in card["intended_use"]
    print(json.dumps(card, indent=2))
    print(f"\nSaved Model Card JSON to: {out_path}")
    print("[SUCCESS] Model Card generated and schema verified.")


if __name__ == "__main__":
    main()

