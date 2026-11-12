"""
Script 03: Gaussian Naive Bayes (Probabilitas Bersyarat & Fitur Kontinu)
Chapter 14: Supervised Learning — Classification Algorithms

Mendemonstrasikan klasifikasi probabilistik Gaussian Naive Bayes:
1. Teorema Bayes: P(y|X) = P(y) * Prod_i P(x_i|y) / P(X)
2. Asumsi Independensi Bersyarat (Conditional Independence)
3. Likelihood Gaussian untuk fitur numerik kontinu: P(x_i|y) ~ N(mu_y, sigma_y^2)
4. Ekstraksi prior probability P(y), mean mu, dan varians sigma^2 tiap kelas
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def load_fault_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI GAUSSIAN NAIVE BAYES")
    print("=" * 75)
    
    df = load_fault_data()
    feature_cols = [
        "temperature_c", "vibration_rms", "acoustic_db",
        "voltage_fluctuation_pct", "pressure_bar", "current_amp"
    ]
    target_col = "fault_detected"
    class_names = ["NORMAL", "FAULT"]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # 1. Parameter yang Dipelajari Model
    print("1. PARAMETER PROBABILISTIK YANG DIPELAJARI:")
    print(f"  * Prior Probability P(Normal) : {gnb.class_prior_[0]:.4f} ({gnb.class_prior_[0]*100:.1f}%)")
    print(f"  * Prior Probability P(Fault)  : {gnb.class_prior_[1]:.4f} ({gnb.class_prior_[1]*100:.1f}%)\n")
    
    print("RATA-RATA GAUSSIAN (mu) DAN VARIANS (sigma^2) TIAP KELAS:")
    means_df = pd.DataFrame(gnb.theta_, index=class_names, columns=feature_cols)
    print("Nilai Rata-rata (mu):")
    print(means_df.round(2).to_string())
    
    print("\nNilai Varians (sigma^2):")
    vars_df = pd.DataFrame(gnb.var_, index=class_names, columns=feature_cols)
    print(vars_df.round(2).to_string())
    
    # 2. Evaluasi pada Test Set
    y_pred = gnb.predict(X_test)
    y_prob = gnb.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n2. EVALUASI KINERJA TEST SET:")
    print(f"  * Akurasi Klasifikasi : {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # 3. Analisis Asumsi Naive Bayes
    print("Analisis Karakteristik Gaussian Naive Bayes:")
    print("1. Asumsi 'Naive': Mengasumsikan setiap sensor saling bebas bersyarat terhadap status mesin.")
    print("2. Meskipun asumsi independensi jarang terpenuhi sempurna di dunia nyata (getaran dan suhu saling terkait),")
    print("   Naive Bayes terbukti sangat tangguh dan bekerja sangat cepat dengan komputasi minimal.")
    print("=" * 75)


if __name__ == "__main__":
    main()

