"""AI Risk Register Demo Mapped to NIST AI RMF 1.0 (Govern, Map, Measure, Manage).

NOTE: Risk likelihood/severity scores shown here are pedagogical synthetic examples
for coursework illustration, NOT an organizational audit result.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class AIRiskItem:
    risk_id: str
    title: str
    nist_function: str
    source: str
    likelihood: int       # 1 (Rare) .. 5 (Almost Certain)
    severity: int         # 1 (Negligible) .. 5 (Critical)
    mitigation: str
    owner: str
    residual_likelihood: int
    residual_severity: int

    @property
    def inherent_score(self) -> int:
        return self.likelihood * self.severity

    @property
    def residual_score(self) -> int:
        return self.residual_likelihood * self.residual_severity

    @property
    def priority(self) -> str:
        score = self.inherent_score
        if score >= 15:
            return "CRITICAL"
        if score >= 9:
            return "HIGH"
        if score >= 4:
            return "MEDIUM"
        return "LOW"


def build_sample_risk_register() -> List[AIRiskItem]:
    """Builds a worked educational Risk Register for an Academic RAG + Screening AI."""
    return [
        AIRiskItem(
            risk_id="RSK-01",
            title="Indirect Prompt Injection via Retrieved Document",
            nist_function="Map / Manage",
            source="Untrusted Corpus Text",
            likelihood=4,
            severity=5,
            mitigation="XML delimiter isolation + instruction/data separation + schema check",
            owner="Security Lead",
            residual_likelihood=2,
            residual_severity=3,
        ),
        AIRiskItem(
            risk_id="RSK-02",
            title="Representation Bias Causing Subgroup TPR Disparity",
            nist_function="Measure",
            source="Historical Skewed Sampling",
            likelihood=4,
            severity=4,
            mitigation="Subgroup fairness audit (EO/DP) + threshold review + Human Oversight",
            owner="ML Data Steward",
            residual_likelihood=2,
            residual_severity=3,
        ),
        AIRiskItem(
            risk_id="RSK-03",
            title="Unauthorized State-Changing Tool Call (Excessive Agency)",
            nist_function="Govern / Manage",
            source="Unconstrained Agent Policy",
            likelihood=3,
            severity=5,
            mitigation="RBAC allowlist + mandatory Human Approval Gate + tamper-evident audit log",
            owner="Agent Platform Owner",
            residual_likelihood=1,
            residual_severity=3,
        ),
        AIRiskItem(
            risk_id="RSK-04",
            title="PII Exposure via Quasi-Identifier Re-Identification",
            nist_function="Map / Measure",
            source="Auxiliary Attribute Linkage",
            likelihood=3,
            severity=4,
            mitigation="Data minimization + k-anonymization generalization + access control",
            owner="Privacy Officer",
            residual_likelihood=2,
            residual_severity=2,
        ),
        AIRiskItem(
            risk_id="RSK-05",
            title="Silent Concept Drift / OOD Confidence Mis-Calibration",
            nist_function="Measure / Manage",
            source="Production Distribution Shift",
            likelihood=4,
            severity=3,
            mitigation="PSI/KS drift monitoring + confidence uncertainty escalation fallback",
            owner="MLOps Engineer",
            residual_likelihood=2,
            residual_severity=2,
        ),
    ]


def main():
    print("=" * 80)
    print("CHAPTER 39: WORKED AI RISK REGISTER (NIST AI RMF 1.0 ALIGNED)")
    print("=" * 80)
    print("Disclaimer: Synthetic academic worked example (not a real org audit).\n")

    register = build_sample_risk_register()
    assert len(register) >= 5
    for r in register:
        assert 1 <= r.likelihood <= 5 and 1 <= r.severity <= 5
        assert r.residual_score <= r.inherent_score

    header = f"{'ID':6s} | {'NIST RMF':15s} | {'Priority':8s} | {'Inh':3s} | {'Res':3s} | {'Owner':19s} | Title"
    print(header)
    print("-" * 105)
    for r in register:
        print(
            f"{r.risk_id:6s} | {r.nist_function:15s} | {r.priority:8s} | "
            f"{r.inherent_score:3d} | {r.residual_score:3d} | {r.owner:19s} | {r.title}"
        )

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "risk_register_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) | {"inherent_score": r.inherent_score, "residual_score": r.residual_score, "priority": r.priority} for r in register], f, indent=2)

    print(f"\nSaved worked register artifact to: {out_path}")
    print("[SUCCESS] AI Risk Register validated.")


if __name__ == "__main__":
    main()

