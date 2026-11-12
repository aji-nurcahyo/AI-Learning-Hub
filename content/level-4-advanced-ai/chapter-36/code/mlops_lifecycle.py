"""MLOps Lifecycle Overview and Stage Contracts.

Demonstrates the structural separation between:
1. ML Development Environment (EDA, Prototyping).
2. Automated Reproducible Training Pipeline.
3. Model Governance & Registry.
4. Local Model Serving & Request Validation.
5. Production Observability & Drift Monitoring.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class LifecycleStage:
    stage_id: int
    name: str
    primary_owner: str
    artifact_output: str
    governance_gate: str


def get_mlops_stages() -> List[LifecycleStage]:
    return [
        LifecycleStage(1, "Data Ingestion & Validation", "Data Engineering", "Versioned Dataset Snapshot", "Schema & Null Checks"),
        LifecycleStage(2, "Reproducible Training Pipeline", "ML Engineering", "Trained Model Artifact + Checksum", "Zero Data Leakage Audit"),
        LifecycleStage(3, "Experiment Tracking", "ML Science", "Params, Metrics & Run Logs", "Baseline Comparison"),
        LifecycleStage(4, "Model Versioning & Registry", "MLOps Platform", "Immutable Versioned Entry", "Checksum & Validation Pass"),
        LifecycleStage(5, "Local Packaging & Serving API", "Backend / Serving", "127.0.0.1 HTTP REST Service", "Input Schema Validation"),
        LifecycleStage(6, "Drift & Latency Monitoring", "SRE / Observability", "PSI, KS, Accuracy, P95 Metrics", "Threshold Alerting"),
        LifecycleStage(7, "Human-Approved Retraining/Rollback", "Governance Team", "Updated Active Model Version", "Explicit Human Approval Gate"),
    ]


def main():
    print("=" * 75)
    print("CHAPTER 36: END-TO-END MLOPS LIFECYCLE STAGES")
    print("=" * 75)

    stages = get_mlops_stages()
    for st in stages:
        print(f"[Stage {st.stage_id}] {st.name}")
        print(f"   Owner          : {st.primary_owner}")
        print(f"   Artifact Output: {st.artifact_output}")
        print(f"   Governance Gate: {st.governance_gate}\n")

    print("Important Principle: Having a local REST API respond with HTTP 200 does NOT make")
    print("a system production-ready. Production readiness requires reproducibility, artifact")
    print("integrity verification, drift monitoring, latency SLAs, and safe rollback controls.")
    print("\n[SUCCESS] MLOps lifecycle contract verified.")


if __name__ == "__main__":
    main()

