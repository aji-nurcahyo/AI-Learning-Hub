"""
Chapter 20 - Machine Learning Capstone
Script 6: Error Analysis & Operational Diagnostics
Menganalisis kegagalan prediksi (False Negatives & False Positives),
mengaudit profil sampel yang salah terprediksi, dan mengoptimalkan decision threshold
berdasarkan trade-off biaya operasional industri.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from preprocessing_pipeline import build_preprocessing_pipeline

DATA_PATH = Path(__file__).parent / "data" / "predictive_maintenance_telemetry.csv"

def run_diagnostics():
    print("=" * 80)
    print("CAPSTONE TAHAP 6: ERROR ANALYSIS & DECISION THRESHOLD TUNING")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "vibration_rms_g", "bearing_temperature_c", "rotational_speed_rpm",
        "hydraulic_pressure_bar", "motor_current_amp", "oil_acidity_tan"
    ]
    X = df[feature_cols]
    y = df['machine_failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ('prep', build_preprocessing_pipeline()),
        ('rf', RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_split=2,
            random_state=42
        ))
    ])
    model.fit(X_train.values, y_train.values)

    probs = model.predict_proba(X_test.values)[:, 1]
    default_preds = (probs >= 0.50).astype(int)

    test_analysis_df = X_test.copy()
    test_analysis_df['actual'] = y_test.values
    test_analysis_df['predicted_default'] = default_preds
    test_analysis_df['failure_prob'] = probs
    test_analysis_df['error_type'] = 'Correct'
    test_analysis_df.loc[(test_analysis_df['actual'] == 1) & (test_analysis_df['predicted_default'] == 0), 'error_type'] = 'False Negative (FN)'
    test_analysis_df.loc[(test_analysis_df['actual'] == 0) & (test_analysis_df['predicted_default'] == 1), 'error_type'] = 'False Positive (FP)'

    fn_samples = test_analysis_df[test_analysis_df['error_type'] == 'False Negative (FN)']
    fp_samples = test_analysis_df[test_analysis_df['error_type'] == 'False Positive (FP)']

    print(f"Jumlah Sampel Uji          : {len(test_analysis_df)}")
    print(f"Jumlah False Negatives (FN): {len(fn_samples)}")
    print(f"Jumlah False Positives (FP): {len(fp_samples)}")

    if len(fn_samples) > 0:
        print("\nProfil Rata-rata False Negative vs Kasus Failure Aktual:")
        print(f"  Vibrasi RMS FN     : {fn_samples['vibration_rms_g'].mean():.2f} g  (vs Avg Failure: {df[df['machine_failure']==1]['vibration_rms_g'].mean():.2f} g)")
        print(f"  Suhu Bearing FN    : {fn_samples['bearing_temperature_c'].mean():.2f} C  (vs Avg Failure: {df[df['machine_failure']==1]['bearing_temperature_c'].mean():.2f} C)")
        print(f"  Peluang Prediksi FN: {fn_samples['failure_prob'].mean():.4f}")

    # Simulasi Decision Threshold Optimization
    print("\nSIMULASI OPTIMASI THRESHOLD BIAYA OPERASIONAL:")
    print("Cost Matrix: Biaya FN = $5,000 | Biaya FP = $200")
    print(f"{'Threshold':<12} {'Recall':<12} {'Presisi':<12} {'FN':<8} {'FP':<8} {'Total Cost ($)':<15}")
    print("-" * 70)

    best_threshold = 0.5
    min_cost = float('inf')

    threshold_candidates = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    for th in threshold_candidates:
        preds = (probs >= th).astype(int)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = (fn * 5000) + (fp * 200)

        if cost < min_cost:
            min_cost = cost
            best_threshold = th

        print(f"{th:<12.2f} {rec*100:>7.2f}%     {prec*100:>7.2f}%     {fn:<8} {fp:<8} ${cost:>10,}")

    print("=" * 70)
    print(f"REKOMENDASI OPERASIONAL:")
    print(f"Threshold optimal adalah {best_threshold:.2f} dengan total risiko kerugian ${min_cost:,}")

if __name__ == "__main__":
    run_diagnostics()
