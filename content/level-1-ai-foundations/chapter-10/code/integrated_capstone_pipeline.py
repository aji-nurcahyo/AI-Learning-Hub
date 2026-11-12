"""
Chapter 10: AI Foundations Capstone
Script 6: integrated_capstone_pipeline.py
Fokus: Pipeline Terpadu Sistem Evakuasi Kampus Cerdas (ICER)
       Menghubungkan:
       1. Spesifikasi PEAS & Perumusan Masalah (Ch 01 & Ch 03)
       2. Telemetri Sensor & Pembersihan Data (Ch 07 & Ch 08)
       3. Representasi Pengetahuan & Forward Chaining (Ch 05)
       4. Pencarian Rute Optimal A* vs BFS (Ch 04)
       5. Audit Etika Keadilan Rute & Kepatuhan UU PDP (Ch 09)

Author: Tim Pengembang AI Learning Hub
Lisensi: MIT
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Impor modul lokal Chapter 10
from problem_formulation import CampusSearchProblem, print_peas_specification
from knowledge_rules import CampusKnowledgeBase
from sensor_data_pipeline import generate_campus_telemetry_raw, clean_telemetry_data
from evacuation_routing import a_star_search, bfs_search as breadth_first_search
from ethics_risk_audit import compute_evacuation_fairness, audit_pdp_compliance, export_icer_risk_register


def run_capstone_pipeline():
    print("=" * 75)
    print("AI FOUNDATIONS CAPSTONE: INTEGRATED CAMPUS EVACUATION ROUTER (ICER)")
    print("Level 1 Capstone Project — Artificial Intelligence Learning Hub")
    print("=" * 75)

    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # FASE 1: SPESIFIKASI PEAS & PERUMUSAN MASALAH
    # -------------------------------------------------------------
    print("\n[FASE 1] Memverifikasi Spesifikasi PEAS & Perumusan Masalah...")
    print_peas_specification()

    # -------------------------------------------------------------
    # FASE 2: TELEMETRI SENSOR & DATA PREPARATION (CH 07 & 08)
    # -------------------------------------------------------------
    print("\n[FASE 2] Menjalankan Pipeline Telemetri Sensor & Pembersihan Data...")
    raw_csv = data_dir / "campus_telemetry_raw.csv"
    clean_csv = data_dir / "campus_telemetry_clean.csv"

    if not raw_csv.exists():
        df_raw = generate_campus_telemetry_raw(raw_csv)
    else:
        df_raw = pd.read_csv(raw_csv)

    clean_df = clean_telemetry_data(df_raw, clean_csv)
    null_count = int(df_raw[["temp_c", "smoke_ppm"]].isna().sum().sum())
    high_temp_count = int((clean_df["temp_c"] >= 55.0).sum())

    print(f"  * Total simpul telemetri : {len(clean_df)}")
    print(f"  * Nilai hilang terimputasi: {null_count} sel")
    print(f"  * Titik suhu tinggi (>55C): {high_temp_count} simpul")

    # -------------------------------------------------------------
    # FASE 3: REPRESENTASI PENGETAHUAN & FORWARD CHAINING (CH 05)
    # -------------------------------------------------------------
    print("\n[FASE 3] Mengaktifkan Mesin Inferensi Rule-Based Knowledge Base...")
    sensor_records = []
    for _, row in clean_df.iterrows():
        sensor_records.append({
            "coord": (int(row["grid_x"]), int(row["grid_y"])),
            "smoke_ppm": float(row["smoke_ppm"]),
            "temp_c": float(row["temp_c"]),
            "has_ramp": bool(row["has_ramp"])
        })

    # Evaluasi mode reguler
    kb_std = CampusKnowledgeBase()
    blocked_std, penalties_std = kb_std.infer_safety_status(sensor_records, wheelchair_mode=False)

    # Evaluasi mode kursi roda (disabilitas mobilitas)
    kb_wc = CampusKnowledgeBase()
    blocked_wc, penalties_wc = kb_wc.infer_safety_status(sensor_records, wheelchair_mode=True)

    print(f"  * Zona Terblokir Bahaya (Reguler)   : {len(blocked_std)} simpul -> {sorted(list(blocked_std))}")
    print(f"  * Zona Terblokir Bahaya (Kursi Roda): {len(blocked_wc)} simpul (termasuk tangga non-ram)")
    print(f"  * Zona Penalti Asap Tebal          : {len(penalties_std)} simpul")

    # -------------------------------------------------------------
    # FASE 4: OPTIMISASI RUTE EVAKUASI DENGAN A* SEARCH (CH 04)
    # -------------------------------------------------------------
    print("\n[FASE 4] Menjalankan Algoritma Pencarian Rute (A* vs BFS)...")
    start_pos = (1, 1)
    safe_goals = {(9, 9), (0, 9)}

    # 4a. Rute Standar
    prob_std = CampusSearchProblem(
        grid_size=(10, 10),
        initial_state=start_pos,
        goals=safe_goals,
        blocked_zones=blocked_std,
        hazard_penalties=penalties_std
    )
    path_astar_std, cost_astar_std, nodes_astar_std = a_star_search(prob_std)
    path_bfs_std, cost_bfs_std, nodes_bfs_std = breadth_first_search(prob_std)

    # 4b. Rute Aksesibel Kursi Roda
    prob_wc = CampusSearchProblem(
        grid_size=(10, 10),
        initial_state=start_pos,
        goals=safe_goals,
        blocked_zones=blocked_wc,
        hazard_penalties=penalties_wc
    )
    path_astar_wc, cost_astar_wc, nodes_astar_wc = a_star_search(prob_wc)

    print("  --- Ringkasan Rute Standar (Reguler) ---")
    print(f"  * A* Path Length   : {len(path_astar_std)} simpul | Total Path Cost: {cost_astar_std:.2f}")
    print(f"  * A* Simpul Dieksplorasi : {nodes_astar_std} simpul")
    print(f"  * BFS Simpul Dieksplorasi: {nodes_bfs_std} simpul")
    reduction = ((nodes_bfs_std - nodes_astar_std) / nodes_bfs_std) * 100
    print(f"  * Pengurangan Eksplorasi A* vs BFS: {reduction:.1f}%")

    print("\n  --- Ringkasan Rute Ramah Disabilitas (Kursi Roda) ---")
    print(f"  * A* Path Length   : {len(path_astar_wc)} simpul | Total Path Cost: {cost_astar_wc:.2f}")
    print(f"  * A* Simpul Dieksplorasi : {nodes_astar_wc} simpul")

    # -------------------------------------------------------------
    # FASE 5: AUDIT KEADILAN (FAIRNESS) & RESPONSIBLE AI (CH 09)
    # -------------------------------------------------------------
    print("\n[FASE 5] Menjalankan Audit Keadilan Algoritmik & Kepatuhan Etika...")
    fairness = compute_evacuation_fairness(
        standard_cost=cost_astar_std,
        wheelchair_cost=cost_astar_wc,
        max_acceptable_disparity_ratio=1.35
    )
    print(f"  * Biaya Jalur Reguler        : {fairness['standard_cost_seconds']} detik/langkah")
    print(f"  * Biaya Jalur Kursi Roda     : {fairness['wheelchair_cost_seconds']} detik/langkah")
    print(f"  * Rasio Disparitas Rute      : {fairness['disparity_ratio']} (Ambang batas <= {fairness['disparity_threshold']})")
    print(f"  * Status Keadilan (Fairness) : {fairness['fairness_status']}")

    pdp_specs = {
        "telemetry_retention_minutes": 15,
        "stores_pii_identity": False,
        "has_human_override_switch": True
    }
    pdp_audit = audit_pdp_compliance(pdp_specs)
    print(f"  * Kepatuhan UU PDP No. 27/2022 : {'MEMENUHI SYARAT' if pdp_audit['pdp_compliant'] else 'TIDAK PATUH'}")

    risk_csv = data_dir / "icer_risk_register.csv"
    export_icer_risk_register(str(risk_csv))

    print("\n" + "=" * 75)
    print("[SUCCESS] PIPELINE CAPSTONE SELESAI SECARA MENYELURUH DENGAN STATUS VALID!")
    print("=" * 75)


if __name__ == "__main__":
    run_capstone_pipeline()
