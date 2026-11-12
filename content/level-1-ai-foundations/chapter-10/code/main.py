"""
Smart Sensor Data Quality Monitor - Main Pipeline Runner
Chapter 10: AI Foundations Capstone Reference Implementation

Menjalankan alur pemantauan kualitas data sensor end-to-end:
1. Ingesti Data Telemetri Sintetis
2. Validasi Skema & Batas Fisika
3. Analisis Statistik Deskriptif & Pagar Tukey
4. Mesin Inferensi Rule-Based & Pembuatan Alert
5. Visualisasi Dashboard Mutu Sensor
6. Pelaporan & Batasan Sistem
"""

import sys
from pathlib import Path
import pandas as pd

from dataset_generator import generate_synthetic_sensor_data
from validator import validate_schema, validate_sensor_records
from statistics_analyzer import compute_sensor_statistics
from rule_engine import DataQualityRuleEngine
from visualization import render_quality_dashboard


def run_sensor_quality_pipeline():
    print("=" * 75)
    print("AI FOUNDATIONS CAPSTONE: SMART SENSOR DATA QUALITY MONITOR")
    print("Reference Implementation — Level 1 AI Foundations Capstone")
    print("=" * 75)
    
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = data_dir / "sensor_telemetry_raw.csv"
    dashboard_png = data_dir / "sensor_quality_dashboard.png"
    
    # -------------------------------------------------------------
    # FASE 1: INGESTI DATA TELEMETRI
    # -------------------------------------------------------------
    print("\n[FASE 1] Ingesti Data Telemetri Sensor...")
    if not raw_csv.exists():
        df_raw = generate_synthetic_sensor_data(raw_csv, n_records=120, seed=42)
    else:
        df_raw = pd.read_csv(raw_csv)
    print(f"  * Total rekaman dimuat   : {len(df_raw)} baris")
    print(f"  * Sensor teridentifikasi : {df_raw['sensor_id'].unique().tolist()}")
    
    # -------------------------------------------------------------
    # FASE 2: VALIDASI SKEMA & BATAS FISIKA
    # -------------------------------------------------------------
    print("\n[FASE 2] Validasi Skema & Batas Fisika Pengukuran...")
    is_valid_schema, schema_msg = validate_schema(df_raw)
    if not is_valid_schema:
        print(f"  [ERROR] {schema_msg}")
        sys.exit(1)
        
    df_validated, val_report = validate_sensor_records(df_raw)
    print(f"  * Skema data            : VALID")
    print(f"  * Rekaman Valid         : {val_report['valid_records_count']} baris")
    print(f"  * Missing Values (NaN)  : {val_report['missing_count']} baris")
    print(f"  * Pelanggaran Batas Fisik: {val_report['invalid_bounds_count']} baris")
    for inv in val_report["invalid_details"]:
        print(f"    - [{inv['timestamp']}] {inv['sensor_id']}: {inv['value']} -> {inv['reason']}")
        
    # -------------------------------------------------------------
    # FASE 3: ANALISIS STATISTIK & PAGAR TUKEY (IQR)
    # -------------------------------------------------------------
    print("\n[FASE 3] Analisis Statistik Deskriptif & Pagar Tukey...")
    stats = compute_sensor_statistics(df_validated)
    for sid, st in stats.items():
        print(f"  * {sid} ({st['unit']}):")
        print(f"    Mean={st['mean']}, Median={st['median']}, Std={st['std']}")
        print(f"    IQR={st['iqr']} -> Pagar Tukey: [{st['lower_fence']}, {st['upper_fence']}]")
        print(f"    Outlier Terdeteksi: {st['outlier_count']} titik")
        
    # -------------------------------------------------------------
    # FASE 4: MESIN INFERENSI RULE-BASED & PERINGATAN KUALITAS
    # -------------------------------------------------------------
    print("\n[FASE 4] Evaluasi Aturan Inferensi & Pembangkitan Alert...")
    engine = DataQualityRuleEngine(stats)
    alerts = engine.evaluate_records(df_validated)
    dqi_scores = engine.calculate_data_quality_index(df_validated)
    
    print(f"  * Total Alert Terpicu   : {len(alerts)} peringatan")
    print(f"  * Skor Data Quality Index (DQI):")
    for sid, score in dqi_scores.items():
        status = "EXCELLENT" if score >= 85 else "ACCEPTABLE" if score >= 70 else "POOR"
        print(f"    - {sid}: {score:.2f}/100 [{status}]")
        
    print("\n  --- Sampel Peringatan Teratas ---")
    for a in alerts[:4]:
        print(f"  * [{a['severity']}] {a['rule_id']}: {a['title']}")
        print(f"    Penjelasan : {a['explanation']}")
        print(f"    Rekomendasi: {a['recommended_action']}")
        
    # -------------------------------------------------------------
    # FASE 5: VISUALISASI DASHBOARD MUTU SENSOR
    # -------------------------------------------------------------
    print("\n[FASE 5] Merender Dashboard Visualisasi Kualitas Data...")
    render_quality_dashboard(df_validated, stats, dqi_scores, dashboard_png)
    print(f"  * Dashboard berhasil diekspor ke: {dashboard_png}")
    
    # -------------------------------------------------------------
    # FASE 6: BATASAN SISTEM & RESPONSIBLE AI
    # -------------------------------------------------------------
    print("\n[FASE 6] Batasan Sistem & Pengawasan Manusia (Human-in-the-Loop)...")
    print("  1. Outlier statistik (RULE-03) tidak otomatis berarti sensor rusak fisik;")
    print("     bisa mencerminkan beban kerja puncak sesaat pada mesin.")
    print("  2. Seluruh alert bersifat rekomendasi pendukung keputusan (Advisory Tool).")
    print("  3. Keputusan penggantian atau pemutusan sensor tetap berada di tangan teknisi manusia.")
    print("=" * 75)
    print("[SUCCESS] PIPELINE SMART SENSOR MONITOR SELESAI SECARA MENYELURUH DENGAN VALID!")
    print("=" * 75)


if __name__ == "__main__":
    run_sensor_quality_pipeline()

