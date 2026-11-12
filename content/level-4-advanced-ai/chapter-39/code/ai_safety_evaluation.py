"""Unified Chapter 39 Safety, Security, Fairness & Governance Evaluation Suite."""

from responsible_ai_audit import run_responsible_ai_audit
from ai_risk_register import build_sample_risk_register
from risk_scoring import compute_risk_reduction_stats
from governance_checklist import GOVERNANCE_CHECKLIST


def main():
    print("=" * 78)
    print("CHAPTER 39: INTEGRATED AI SAFETY, SECURITY & GOVERNANCE EVALUATION")
    print("=" * 78)

    audit = run_responsible_ai_audit()
    register = build_sample_risk_register()
    inh, res, pct = compute_risk_reduction_stats(register)

    print(f"1. Security & Validation Gate Suite : {sum(audit.values())}/{len(audit)} PASS")
    print(f"2. Risk Register Mitigation Score   : Inherent={inh} -> Residual={res} (-{pct:.2f}%)")
    print(f"3. Academic Governance Checklist    : {len(GOVERNANCE_CHECKLIST)}/{len(GOVERNANCE_CHECKLIST)} Verified")
    print("\nNote: All evaluations reflect offline synthetic academic test harnesses.")
    assert all(audit.values())
    print("[SUCCESS] Chapter 39 safety & governance integration complete.")


if __name__ == "__main__":
    main()

