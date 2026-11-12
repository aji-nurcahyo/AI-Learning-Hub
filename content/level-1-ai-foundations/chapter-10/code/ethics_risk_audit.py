"""
Chapter 10: AI Foundations Capstone
Script 5: ethics_risk_audit.py
Fokus: Audit Etika, Keadilan Rute (Fairness), Privasi Data (UU PDP No. 27/2022),
       dan Human-in-the-Loop Override Protocol pada Sistem Evakuasi Cerdas (ICER).

Author: Tim Pengembang AI Learning Hub
Lisensi: MIT
"""

import os
import csv
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def compute_evacuation_fairness(
    standard_cost: float,
    wheelchair_cost: float,
    max_acceptable_disparity_ratio: float = 1.35
) -> Dict[str, Any]:
    """
    Menghitung metrik keadilan evakuasi (Demographic Parity / Route Disparity Ratio).
    Memastikan penyandang disabilitas mobilitas tidak mengalami waktu evakuasi
    yang melebihi batas keselamatan kritis.
    """
    disparity_ratio = wheelchair_cost / standard_cost if standard_cost > 0 else 1.0
    is_equitable = disparity_ratio <= max_acceptable_disparity_ratio

    return {
        "standard_cost_seconds": round(standard_cost, 2),
        "wheelchair_cost_seconds": round(wheelchair_cost, 2),
        "disparity_ratio": round(disparity_ratio, 3),
        "disparity_threshold": max_acceptable_disparity_ratio,
        "is_equitable": is_equitable,
        "fairness_status": "FAIR" if is_equitable else "EQUITY_VIOLATION_DETECTED"
    }

def audit_pdp_compliance(system_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Melakukan verifikasi kepatuhan terhadap UU Pelindungan Data Pribadi (UU PDP No. 27/2022)
    khususnya prinsip data minimisation, purpose limitation, dan storage limitation.
    """
    findings = []
    compliant = True

    # Check 1: Storage limitation - Telemetri lokasi harus ephemeral (tidak disimpan permanen)
    if system_specs.get("telemetry_retention_minutes", 0) > 60:
        findings.append("PELANGGARAN: Retensi telemetri lokasi > 60 menit melanggar asas storage limitation.")
        compliant = False
    else:
        findings.append("LULUS: Telemetri lokasi bersifat ephemeral (retensi <= 60 menit).")

    # Check 2: Data minimisation - Tidak menyimpan identitas pribadi (PII) mahasiswa/dosen
    if system_specs.get("stores_pii_identity", True):
        findings.append("PELANGGARAN: Sistem menyimpan NIK/NIM langsung pada node rute.")
        compliant = False
    else:
        findings.append("LULUS: Pseudonimisasi/Agregasi kepadatan digunakan (tanpa PII langsung).")

    # Check 3: Human-in-the-Loop - Harus ada mekanisme override manual oleh komandan satpam/damkar
    if not system_specs.get("has_human_override_switch", False):
        findings.append("PELANGGARAN: Sistem tidak memiliki mekanisme override manual Human-in-the-Loop.")
        compliant = False
    else:
        findings.append("LULUS: Mekanisme override darurat petugas keselamatan aktif.")

    return {
        "pdp_compliant": compliant,
        "audit_findings": findings
    }

def export_icer_risk_register(output_csv_path: str) -> None:
    """
    Mengekspor Risk Register keselamatan & etika sistem ICER ke format CSV
    sesuai kerangka kerja NIST AI RMF 1.0 (Govern, Map, Measure, Manage).
    """
    risks = [
        {
            "Risk_ID": "RISK-01",
            "Category": "Physical Safety",
            "Description": "Sensor api gagal deteksi asap di koridor sempit",
            "Likelihood": "Low",
            "Impact": "Critical",
            "Mitigation_Strategy": "Redundansi multi-sensor (suhu + CO + visual) & fallback manual",
            "NIST_Function": "Measure"
        },
        {
            "Risk_ID": "RISK-02",
            "Category": "Algorithmic Bias / Fairness",
            "Description": "Rute evakuasi kursi roda terhalang tangga tanpa jalur alternatif",
            "Likelihood": "Medium",
            "Impact": "High",
            "Mitigation_Strategy": "Constraint ramp check pada Knowledge Base sebelum rute dihitung",
            "NIST_Function": "Map"
        },
        {
            "Risk_ID": "RISK-03",
            "Category": "Privacy & UU PDP",
            "Description": "Data pergerakan civitas akademika bocor atau dilacak tanpa izin",
            "Likelihood": "Medium",
            "Impact": "High",
            "Mitigation_Strategy": "Agregasi spasial node tanpa merekam identitas perangkat/NIM",
            "NIST_Function": "Govern"
        },
        {
            "Risk_ID": "RISK-04",
            "Category": "Systemic / Robustness",
            "Description": "Koneksi jaringan putus saat bencana gempa/kebakaran",
            "Likelihood": "High",
            "Impact": "Critical",
            "Mitigation_Strategy": "Edge computing lokal di tiap gedung dengan signage LED otonom",
            "NIST_Function": "Manage"
        }
    ]

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(risks[0].keys()))
        writer.writeheader()
        writer.writerows(risks)

    print(f"[SUCCESS] Risk Register berhasil diekspor ke: {output_csv_path}")

def run_ethics_audit():
    print("=" * 60)
    print("CHAPTER 10: ETHICS & RESPONSIBLE AI AUDIT (ICER SYSTEM)")
    print("=" * 60)

    # 1. Audit Fairness Evakuasi
    # Asumsi baseline: Jalur standar 14.0 unit waktu, Jalur kursi roda 16.0 unit waktu
    fairness = compute_evacuation_fairness(standard_cost=14.0, wheelchair_cost=16.0)
    print("\n--- 1. AUDIT DEMOGRAPHIC PARITY & DISABILITY ROUTING ---")
    print(f"Biaya Rute Evakuasi Reguler   : {fairness['standard_cost_seconds']} detik/langkah")
    print(f"Biaya Rute Kursi Roda          : {fairness['wheelchair_cost_seconds']} detik/langkah")
    print(f"Rasio Disparitas (Disparity)   : {fairness['disparity_ratio']} (Batas aman <= {fairness['disparity_threshold']})")
    print(f"Status Keadilan (Fairness)     : {fairness['fairness_status']}")

    # 2. Audit Kepatuhan Regulasi PDP (UU PDP 27/2022)
    print("\n--- 2. AUDIT PRIVASI & REGULASI UU PDP NO. 27/2022 ---")
    specs = {
        "telemetry_retention_minutes": 15,
        "stores_pii_identity": False,
        "has_human_override_switch": True
    }
    pdp_audit = audit_pdp_compliance(specs)
    print(f"Status Kepatuhan UU PDP        : {'MEMENUHI SYARAT' if pdp_audit['pdp_compliant'] else 'TIDAK PATUH'}")
    for item in pdp_audit["audit_findings"]:
        print(f"  * {item}")

    # 3. Export Risk Register
    csv_out = os.path.join(os.path.dirname(__file__), "data", "icer_risk_register.csv")
    print("\n--- 3. PENYUSUNAN NIST AI RMF RISK REGISTER ---")
    export_icer_risk_register(csv_out)
    print("=" * 60)

if __name__ == "__main__":
    run_ethics_audit()

