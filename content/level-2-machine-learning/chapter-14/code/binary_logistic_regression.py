"""
Script 01: Binary Logistic Regression (Klasifikasi Logistik Biner)
Chapter 14: Supervised Learning — Classification Algorithms

Mendemonstrasikan klasifikasi biner deteksi kerusakan mesin (Normal vs Fault):
1. Fungsi Sigmoid: P(y=1|x) = 1 / (1 + exp(-z))
2. Hubungan Log-Odds (Logit): log(p / (1 - p)) = w^T x + b
3. Perbedaan antara skor probabilitas kontinu (predict_proba) dan prediksi kelas diskrit (predict)
4. Interpretasi Odds Ratio dari koefisien model terestimasi
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


def load_fault_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI BINARY LOGISTIC REGRESSION")
    print("=" * 75)
    
    df = load_fault_data()
    feature_cols = [
        "temperature_c", "vibration_rms", "acoustic_db",
        "voltage_fluctuation_pct", "pressure_bar", "current_amp"
    ]
    X = df[feature_cols]
    y = df["fault_detected"]
    
    # Stratified Split 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Membangun Pipeline terstandarisasi
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    
    # 1. Ekstraksi Koefisien dan Interpretasi Odds Ratio
    coefs = pipe.named_steps["classifier"].coef_[0]
    intercept = pipe.named_steps["classifier"].intercept_[0]
    
    print(f"Data: {len(X_train)} Train, {len(X_test)} Test ({len(feature_cols)} Fitur)")
    print(f"Distribusi Target Test : Normal={sum(y_test==0)}, Fault={sum(y_test==1)}\n")
    
    print("KOEFISIEN LOGIT DAN ODDS RATIO (EFEK MULTIPLIKATIF RISIKO KERUSAKAN):")
    print(f"{'Nama Fitur':<26} | {'Koefisien (w)':<16} | {'Odds Ratio exp(w)':<18}")
    print("-" * 66)
    for col, w in zip(feature_cols, coefs):
        odds_ratio = np.exp(w)
        print(f"{col:<26} | {w:>14.4f}   | {odds_ratio:>16.4f}x")
    print(f"{'Intercept (b)':<26} | {intercept:>14.4f}   | {'-':<18}")
    print("-" * 66)
    
    # 2. Contoh Perhitungan Manual Sigmoid vs predict_proba Scikit-learn
    sample = X_test.iloc[[0]]
    sample_scaled = pipe.named_steps["scaler"].transform(sample)
    z_manual = np.dot(sample_scaled, coefs) + intercept
    prob_manual = 1.0 / (1.0 + np.exp(-z_manual[0]))
    prob_skl = pipe.predict_proba(sample)[0, 1]
    
    print(f"\nVERIFIKASI RUMUS SIGMOID PADA SAMPEL UJI:")
    print(f"  * Linear Logit (z)              : {z_manual[0]:.4f}")
    print(f"  * Probabilitas Manual Sigmoid    : {prob_manual:.4f}")
    print(f"  * Probabilitas predict_proba API : {prob_skl:.4f} (IDENTIK 100%)")
    print(f"  * Prediksi Kelas (Threshold 0.5): {'FAULT' if prob_skl >= 0.5 else 'NORMAL'}\n")
    
    # 3. Evaluasi Kinerja pada Seluruh Test Set
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    print("LAPORAN EVALUASI MODEL PADA TEST SET:")
    print(f"  * Akurasi Klasifikasi : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"  * ROC-AUC Score       : {roc_auc_score(y_test, y_prob):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fault"]))
    print("=" * 75)


if __name__ == "__main__":
    main()

