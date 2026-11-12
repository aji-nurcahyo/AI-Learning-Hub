"""
Script 04: Multinomial Naive Bayes (Data Diskrit & Laplace Smoothing)
Chapter 14: Supervised Learning — Classification Algorithms

Mendemonstrasikan Multinomial Naive Bayes pada fitur diskrit/frekuensi alarm:
1. Diskritisasi data sensor menjadi frekuensi kejadian alarm (discrete counts)
2. Bahaya Zero-Frequency Problem: Jika satu kata/fitur tidak pernah muncul pada data latih,
   seluruh probabilitas posterior menjadi 0 (P(y|x) = 0).
3. Solusi Matematis: Laplace Smoothing (alpha = 1.0)
4. Evaluasi performa klasifikasi pada representasi diskrit
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report, accuracy_score


def load_fault_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI MULTINOMIAL NAIVE BAYES & LAPLACE SMOOTHING")
    print("=" * 75)
    
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
    
    # 1. Diskritisasi Fitur Sensor Menjadi 5 Tingkat Keparahan Diskrit (0 s/d 4)
    # Merepresentasikan level status alarm (Low, Moderate, Elevated, High, Critical)
    discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
    discretizer.fit(X_train)
    
    X_train_discrete = discretizer.transform(X_train)
    X_test_discrete = discretizer.transform(X_test)
    
    print("1. DISKRITISASI FITUR SENSOR MENJADI 5 LEVEL ALARM DISKRIT [0, 1, 2, 3, 4]:")
    print(f"   Contoh 3 baris data diskrit:\n{X_train_discrete[:3]}\n")
    
    # 2. Perbandingan Model Tanpa Smoothing (alpha=1e-9) vs Dengan Laplace Smoothing (alpha=1.0)
    alphas = [1e-9, 0.1, 1.0, 5.0]
    print(f"{'Nilai Alpha':<14} | {'Laplace Status':<22} | {'Test Accuracy':<16} | {'Test F1-Score':<14}")
    print("-" * 72)
    
    for a in alphas:
        mnb = MultinomialNB(alpha=a)
        mnb.fit(X_train_discrete, y_train)
        y_pred = mnb.predict(X_test_discrete)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        f1 = report["1"]["f1-score"]
        
        status = "Tanpa Smoothing" if a < 0.01 else ("Laplace Standar" if a == 1.0 else "Smoothing Terbobot")
        print(f"{a:<14.4f} | {status:<22} | {acc*100:>13.2f}% | {f1:>12.4f}")
        
    print("-" * 72)
    print("\nAnalisis Laplace Smoothing:")
    print("Formula Smoothing: P(x_i | y) = (N_yi + alpha) / (N_y + alpha * D)")
    print("Laplace smoothing (alpha=1.0) menambahkan 1 ke setiap frekuensi observasi,")
    print("menghilangkan risiko nilai nol yang dapat membatalkan seluruh perhitungan posterior.")
    print("=" * 75)


if __name__ == "__main__":
    main()

