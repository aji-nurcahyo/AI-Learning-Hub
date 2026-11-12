"""
Chapter 20 - Machine Learning Capstone
Script 7: Explainability & Interpretability Analysis
Menghitung Permutation Feature Importance dan Partial Dependence Plots (PDP)
untuk memberikan transparansi penalaran model bagi teknisi dan manajemen industri.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from preprocessing_pipeline import build_preprocessing_pipeline

DATA_PATH = Path(__file__).parent / "data" / "predictive_maintenance_telemetry.csv"

def run_explainability():
    print("=" * 80)
    print("CAPSTONE TAHAP 7: MODEL EXPLAINABILITY & ENGINEERING INTERPRETABILITY")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "vibration_rms_g", "bearing_temperature_c", "rotational_speed_rpm",
        "hydraulic_pressure_bar", "motor_current_amp", "oil_acidity_tan"
    ]
    X = df[feature_cols].values
    y = df['machine_failure'].values

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
    model.fit(X_train, y_train)

    # 1. Permutation Feature Importance pada Test Set
    print("\n[1] Permutation Importance pada Test Set (5 pengacakan berulang):")
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, scoring='f1'
    )

    perm_df = pd.DataFrame({
        'Fitur': feature_cols,
        'Mean_Importance': perm_result.importances_mean,
        'Std_Dev': perm_result.importances_std
    }).sort_values(by='Mean_Importance', ascending=False)

    print(f"{'Fitur':<28} {'Mean Drop in F1':<18} {'Std Dev':<10}")
    print("-" * 60)
    for _, row in perm_df.iterrows():
        print(f"{row['Fitur']:<28} {row['Mean_Importance']:>12.4f}       ± {row['Std_Dev']:.4f}")
    print("=" * 60)

    # 2. Analisis Partial Dependence pada Fitur Fisik Dominan
    top_feature_idx = feature_cols.index(perm_df.iloc[0]['Fitur'])
    top_feature_name = perm_df.iloc[0]['Fitur']
    print(f"\n[2] Partial Dependence Profile untuk Fitur Kunci: '{top_feature_name}'")

    pdp_results = partial_dependence(
        model, X_test, features=[top_feature_idx], grid_resolution=10
    )

    grid_values = pdp_results['grid_values'][0]
    avg_predictions = pdp_results['average'][0]

    print(f"{'Nilai Fitur (' + top_feature_name + ')':<30} {'Prediksi Peluang Rusak Rata-rata':<35}")
    print("-" * 68)
    for val, pred in zip(grid_values, avg_predictions):
        bar_len = int(pred * 25)
        bar = "#" * bar_len
        print(f"{val:>15.2f}                {pred:>8.4f}  [{bar:<25}]")
    print("=" * 68)

    print("\nINSIGHT TEKNIK:")
    print(f"Peluang kerusakan meningkat drastis saat '{top_feature_name}' melintasi ambang kritis operasional.")

if __name__ == "__main__":
    run_explainability()
