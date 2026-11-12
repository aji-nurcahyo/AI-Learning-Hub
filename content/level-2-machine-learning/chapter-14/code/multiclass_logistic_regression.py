"""
Script 02: Multiclass Logistic Regression (Softmax vs One-vs-Rest)
Chapter 14: Supervised Learning — Classification Algorithms

Mendemonstrasikan klasifikasi multi-kelas diagnosis jenis kerusakan industri:
1. Membedakan strategi Multinomial Softmax vs One-vs-Rest (OvR)
2. Fungsi Softmax: P(y=k|x) = exp(z_k) / sum_j exp(z_j)
3. Evaluasi Confusion Matrix multi-kelas 4x4
4. Perbedaan krusial antara Macro Average dan Weighted Average
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


def load_fault_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 80)
    print("DEMONSTRASI MULTICLASS LOGISTIC REGRESSION (DIAGNOSIS 4 KELAS KERUSAKAN)")
    print("=" * 80)
    
    df = load_fault_data()
    feature_cols = [
        "temperature_c", "vibration_rms", "acoustic_db",
        "voltage_fluctuation_pct", "pressure_bar", "current_amp"
    ]
    target_col = "fault_type_id"
    class_names = ["NORMAL", "BEARING_FAULT", "OVERHEATING", "ELECTRICAL_SURGE"]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Data Latih: {len(X_train)} sampel | Data Uji: {len(X_test)} sampel")
    print(f"Sebaran Kelas Uji: {dict(zip(class_names, np.bincount(y_test)))}\n")
    
    # 1. Multinomial Softmax Model
    multi_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])
    multi_pipe.fit(X_train, y_train)
    y_pred_multi = multi_pipe.predict(X_test)
    y_prob_multi = multi_pipe.predict_proba(X_test)
    
    # 2. Confusion Matrix 4x4
    cm = confusion_matrix(y_test, y_pred_multi)
    print("CONFUSION MATRIX 4x4 (Baris: Aktual, Kolom: Prediksi):")
    cm_df = pd.DataFrame(cm, index=[f"True: {c}" for c in class_names], columns=[f"Pred: {c}" for c in class_names])
    print(cm_df.to_string())
    print()
    
    # 3. Laporan Klasifikasi & Analisis Averaging
    print("LAPORAN KLASIFIKASI MULTIKELAS LENGKAP:")
    print(classification_report(y_test, y_pred_multi, target_names=class_names, digits=4))
    
    # 4. Contoh Probabilitas Softmax untuk Satu Sampel Uji
    sample_idx = 0
    sample_probs = y_prob_multi[sample_idx]
    true_lbl = class_names[y_test.iloc[sample_idx]]
    pred_lbl = class_names[y_pred_multi[sample_idx]]
    
    print(f"CONTOH PROBABILITAS SOFTMAX SAMPEL KE-1 (Kelas Asli: {true_lbl}):")
    for name, p in zip(class_names, sample_probs):
        bar = "#" * int(p * 30)
        print(f"  * {name:<18}: {p*100:>6.2f}% | {bar}")
    print(f"  -> Keputusan Akhir: {pred_lbl} (argmax probabilitas)\n")
    
    print("Pelajaran Metodologis Multi-Class Averaging:")
    print("1. Macro Average menghitung metrik setiap kelas secara terpisah lalu meratakannya;")
    print("   memberikan bobot yang SAMA pada kelas langka (Electrical Surge) maupun kelas dominan (Normal).")
    print("2. Weighted Average membobot metrik berdasarkan proporsi jumlah sampel masing-masing kelas.")
    print("=" * 80)


if __name__ == "__main__":
    main()

