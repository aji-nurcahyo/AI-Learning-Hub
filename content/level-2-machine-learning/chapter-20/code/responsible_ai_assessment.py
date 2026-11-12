"""
Chapter 20 - Machine Learning Capstone
Script 8: Responsible AI & Model Governance Assessment
Mengaudit keadilan performa antar subkelompok mesin (pump units)
dan menetapkan batas operasional aman (operational boundary guidelines).
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from preprocessing_pipeline import build_preprocessing_pipeline

DATA_PATH = Path(__file__).parent / "data" / "predictive_maintenance_telemetry.csv"

def run_responsible_ai():
    print("=" * 80)
    print("CAPSTONE TAHAP 8: RESPONSIBLE AI, FAIRNESS & GOVERNANCE AUDIT")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "vibration_rms_g", "bearing_temperature_c", "rotational_speed_rpm",
        "hydraulic_pressure_bar", "motor_current_amp", "oil_acidity_tan"
    ]
    X = df[feature_cols].values
    y = df['machine_failure'].values
    machine_ids = df['machine_id'].values

    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    machines_test = machine_ids[idx_test]

    model = Pipeline([
        ('prep', build_preprocessing_pipeline()),
        ('rf', RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=2,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 1. Audit Keadilan Antar Subgrup Mesin (Machine Units)
    print("\n[1] Audit Disparitas Performa Antar Unit Mesin (PUMP_01 s/d PUMP_12):")
    audit_df = pd.DataFrame({
        'machine_id': machines_test,
        'y_true': y_test,
        'y_pred': y_pred
    })

    print(f"{'Machine ID':<15} {'Total Sampel':<15} {'Akurasi':<12} {'F1-Score':<10}")
    print("-" * 55)

    subgroup_f1 = []
    for m_id, group in audit_df.groupby('machine_id'):
        acc = accuracy_score(group['y_true'], group['y_pred'])
        if len(group['y_true'].unique()) > 1:
            f1 = f1_score(group['y_true'], group['y_pred'])
        else:
            f1 = 1.0 if acc == 1.0 else 0.0
        subgroup_f1.append(f1)
        print(f"{m_id:<15} {len(group):<15} {acc*100:>7.2f}%     {f1:>7.4f}")

    print("=" * 55)
    f1_disparity = max(subgroup_f1) - min(subgroup_f1)
    print(f"Disparitas F1 Maksimum Antar Unit: {f1_disparity:.4f}")
    if f1_disparity < 0.30:
        print("STATUS AUDIT UNIT: LOLOS - Model beroperasi stabil di seluruh unit pompa.")
    else:
        print("STATUS AUDIT UNIT: PERHATIAN - Terdapat variasi pada unit tertentu.")

    # 2. Model Governance & Operational Boundary
    print("\n[2] Model Card & Batas Operasional Aman (Governance Guidelines):")
    print("  - Lingkup Penggunaan: Deteksi dini degradasi kompresor sentrifugal industri.")
    print("  - Batas Larangan    : Dilarang digunakan pada instalasi berisiko kritis tanpa human-in-the-loop.")
    print("  - Protokol Intervensi: Jika probabilitas > 0.35, jadwalkan verifikasi getaran manual.")

if __name__ == "__main__":
    run_responsible_ai()
