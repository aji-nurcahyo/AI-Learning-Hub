"""
Script 08: Precision-Recall Curve & Average Precision (AP)
Chapter 14: Supervised Learning — Classification Algorithms

Mendemonstrasikan kurva Precision-Recall (PR) pada dataset dengan ketidakseimbangan kelas:
1. Mengapa ROC-AUC bisa tampak sangat optimis palsu saat kelas mayoritas negatif sangat besar
2. Kurva PR berfokus murni pada kelas minoritas positif tanpa terpengaruh oleh jumlah True Negative
3. Perhitungan Average Precision (AP / PR-AUC)
4. Garis dasar (baseline) kurva PR = Prevalensi Kelas Positif (bukan 0.5 seperti ROC!)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score


def load_fault_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 80)
    print("DEMONSTRASI PRECISION-RECALL CURVE & AVERAGE PRECISION")
    print("=" * 80)
    
    df = load_fault_data()
    feature_cols = [
        "temperature_c", "vibration_rms", "acoustic_db",
        "voltage_fluctuation_pct", "pressure_bar", "current_amp"
    ]
    target_col = "fault_detected"
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    
    test_probs = pipe.predict_proba(X_test)[:, 1]
    
    # 1. Metrik Kunci
    prevalence = y_test.mean()
    roc_auc = roc_auc_score(y_test, test_probs)
    avg_prec = average_precision_score(y_test, test_probs)
    
    print(f"Data Uji: {len(y_test)} sampel")
    print(f"Prevalensi Kelas Minoritas (Fault=1): {sum(y_test==1)}/{len(y_test)} ({prevalence*100:.2f}%)\n")
    
    print("PERBANDINGAN ROC-AUC VS AVERAGE PRECISION (PR-AUC):")
    print(f"  * Baseline Acak (Random Guess) ROC-AUC : 0.5000 (Tetap)")
    print(f"  * Baseline Acak (Random Guess) PR-AUC  : {prevalence:.4f} (= Prevalensi Data!)")
    print(f"  * Nilai Model ROC-AUC                  : {roc_auc:.4f}")
    print(f"  * Nilai Model Average Precision (AP)   : {avg_prec:.4f}\n")
    
    # 2. Sampel Titik Koordinat Kurva PR
    precisions, recalls, thresholds = precision_recall_curve(y_test, test_probs)
    print("CONTOH TITIK OPERASI PADA KURVA PRECISION-RECALL:")
    print(f"{'Recall (TPR)':<16} | {'Precision':<16} | {'Ambang Batas Probabilitas':<25}")
    print("-" * 65)
    # Pilih 5 titik representatif
    step = max(1, len(thresholds) // 5)
    for idx in range(0, len(thresholds), step):
        print(f"{recalls[idx]:<16.4f} | {precisions[idx]:<16.4f} | P(y=1) >= {thresholds[idx]:.4f}")
    print("-" * 65)
    
    print("\nPedoman Pemilihan Evaluasi Industri:")
    print("1. Gunakan ROC-AUC jika biaya FP dan FN relatif setara atau data cukup berimbang.")
    print("2. Gunakan PR Curve & Average Precision jika mendeteksi anomali/kerusakan yang sangat langka (<10%),")
    print("   karena metrik Presisi langsung mengukur berapa persen alarm yang benar-benar akurat.")
    print("=" * 80)


if __name__ == "__main__":
    main()

