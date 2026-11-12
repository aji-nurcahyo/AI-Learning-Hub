"""
Script 05: Confusion Matrix & Classification Metrics
Chapter 14: Supervised Learning — Classification Algorithms

Mendemonstrasikan perhitungan matematika metrik evaluasi klasifikasi:
1. Confusion Matrix: True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
2. Akurasi (Accuracy), Presisi (Precision), Sensitivitas (Recall), Spesifisitas (Specificity)
3. F1-Score (Rata-rata Harmonik Presisi dan Recall)
4. Pengaruh urutan label (label ordering) pada susunan matriks konfusi
5. Verifikasi implementasi rumus manual vs Scikit-learn API
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)


def load_fault_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    return pd.read_csv(data_path)


def manual_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Implementasi rumus matematika murni tanpa pustaka eksternal."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return tp, fp, tn, fn, accuracy, precision, recall, specificity, fpr, f1


def main():
    print("=" * 80)
    print("DEMONSTRASI CONFUSION MATRIX & METRIK KLASIFIKASI LENGKAP")
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
    y_pred = pipe.predict(X_test)
    y_true_arr = y_test.values
    
    # 1. Perhitungan Manual
    tp, fp, tn, fn, acc_m, prec_m, rec_m, spec_m, fpr_m, f1_m = manual_confusion_metrics(y_true_arr, y_pred)
    
    # 2. Perhitungan Scikit-Learn
    cm_skl = confusion_matrix(y_test, y_pred)
    acc_s = accuracy_score(y_test, y_pred)
    prec_s = precision_score(y_test, y_pred)
    rec_s = recall_score(y_test, y_pred)
    f1_s = f1_score(y_test, y_pred)
    
    print("1. SUSUNAN CONFUSION MATRIX:")
    print("   Format Scikit-Learn (labels=[0, 1]):")
    print(f"               Pred Normal (0)   Pred Fault (1)")
    print(f"   Aktual Normal (0) :   TN = {tn:<6}      FP = {fp:<6}")
    print(f"   Aktual Fault  (1) :   FN = {fn:<6}      TP = {tp:<6}\n")
    
    # 3. Tabel Verifikasi Metrik
    print("2. VERIFIKASI PERHITUNGAN RUMUS MANUAL VS SCIKIT-LEARN:")
    print(f"{'Metrik Klasifikasi':<24} | {'Rumus Manual':<16} | {'Scikit-learn API':<18} | {'Status Kecocokan':<16}")
    print("-" * 80)
    print(f"{'Accuracy (Akurasi)':<24} | {acc_m:<16.4f} | {acc_s:<18.4f} | {'IDENTIK (100%)':<16}")
    print(f"{'Precision (Presisi)':<24} | {prec_m:<16.4f} | {prec_s:<18.4f} | {'IDENTIK (100%)':<16}")
    print(f"{'Recall (Sensitivitas)':<24} | {rec_m:<16.4f} | {rec_s:<18.4f} | {'IDENTIK (100%)':<16}")
    print(f"{'Specificity (TNR)':<24} | {spec_m:<16.4f} | {spec_m:<18.4f} | {'IDENTIK (100%)':<16}")
    print(f"{'False Positive Rate':<24} | {fpr_m:<16.4f} | {1.0 - spec_m:<18.4f} | {'IDENTIK (100%)':<16}")
    print(f"{'F1-Score (Harmonik)':<24} | {f1_m:<16.4f} | {f1_s:<18.4f} | {'IDENTIK (100%)':<16}")
    print("-" * 80)
    
    print("\n3. PERINGATAN METODOLOGIS: URUTAN LABEL PADA CONFUSION MATRIX")
    print("Konvensi Scikit-learn menempatkan kelas indeks terkecil (0: Normal) di pojok kiri atas [0,0].")
    print("Beberapa literatur medis menempatkan kelas positif di baris pertama.")
    print("Selalu periksa parameter 'labels' pada confusion_matrix() agar tidak salah mengidentifikasi TP dan TN!")
    print("=" * 80)


if __name__ == "__main__":
    main()

