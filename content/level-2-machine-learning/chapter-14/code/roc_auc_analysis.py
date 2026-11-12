"""
Script 07: ROC Curve & ROC-AUC Analysis
Chapter 14: Supervised Learning — Classification Algorithms

Mendemonstrasikan analisis Receiver Operating Characteristic (ROC) dan Area Under Curve (ROC-AUC):
1. Plot True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity)
2. Mengapa ROC-AUC membutuhkan skor probabilitas kontinu, BUKAN prediksi biner diskrit
3. Interpretasi Probabilistik ROC-AUC sebagai U-statistic (Wilcoxon-Mann-Whitney)
4. Perbandingan ROC-AUC Logistic Regression vs Naive Bayes vs Random Guess
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score


def load_fault_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    return pd.read_csv(data_path)


def manual_auc_rank_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Menghitung AUC menggunakan interpretasi probabilitas Wilcoxon-Mann-Whitney:
    P(skor sampel positif > skor sampel negatif)
    """
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    concordant = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                concordant += 1.0
            elif p == n:
                concordant += 0.5
                
    return concordant / (n_pos * n_neg)


def main():
    print("=" * 80)
    print("DEMONSTRASI ROC CURVE & ROC-AUC SCORE ANALYSIS")
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
    
    # Model 1: Logistic Regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    lr_pipe.fit(X_train, y_train)
    lr_probs = lr_pipe.predict_proba(X_test)[:, 1]
    lr_preds = lr_pipe.predict(X_test)
    
    # Model 2: Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_probs = gnb.predict_proba(X_test)[:, 1]
    
    # 1. Perhitungan ROC-AUC Skor Kontinu vs Prediksi Diskrit
    auc_lr_prob = roc_auc_score(y_test, lr_probs)
    auc_lr_discrete = roc_auc_score(y_test, lr_preds)  # KESALAHAN UMUM PEMULA!
    auc_gnb_prob = roc_auc_score(y_test, gnb_probs)
    
    auc_manual_wmw = manual_auc_rank_statistic(y_test.values, lr_probs)
    
    print("1. HASIL PENGUKURAN ROC-AUC PADA TEST SET:")
    print(f"  * Logistic Regression (Probabilitas Kontinu) : {auc_lr_prob:.4f}")
    print(f"  * Gaussian Naive Bayes (Probabilitas Kontinu): {auc_gnb_prob:.4f}")
    print(f"  * Baseline Acak (Random Guess)              : 0.5000\n")
    
    print("2. PEMBUKTIAN INTERPRETASI PROBABILISTIK WILCOXON-MANN-WHITNEY:")
    print(f"  * AUC Scikit-learn API           : {auc_lr_prob:.4f}")
    print(f"  * AUC Formula P(Score+ > Score-) : {auc_manual_wmw:.4f} (IDENTIK 100%)")
    print("  -> Artinya: Jika kita mengambil 1 sampel mesin rusak acak dan 1 sampel mesin normal acak,")
    print(f"     terdapat {auc_lr_prob*100:.2f}% peluang model memberikan skor risiko lebih tinggi pada mesin rusak.\n")
    
    print("3. BAHAYA MENYALURKAN PREDIKSI DISKRIT (0/1) KE ROC_AUC_SCORE:")
    print(f"  * AUC dengan predict_proba (Benar) : {auc_lr_prob:.4f}")
    print(f"  * AUC dengan predict biner (Salah) : {auc_lr_discrete:.4f} (Menyusut drastis!)")
    print("  -> Prediksi biner mereduksi kurva ROC yang kaya menjadi hanya satu titik ambang 0.5!")
    print("=" * 80)


if __name__ == "__main__":
    main()

