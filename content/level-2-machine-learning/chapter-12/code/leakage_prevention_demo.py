"""
Script 12: Data Leakage Prevention Demonstration
Chapter 12: ML Workflow, Dataset & Feature Engineering

Eksperimen komparatif pembuktian Data Leakage:
1. Metode Bocor (Leaky Approach):
   Pra-pemrosesan (Imputasi & Scaling) dilakukan pada SELURUH dataset SEBELUM split.
   Informasi distribusi test set merembes ke pelatihan model.
2. Metode Bersih (Leak-Free Pipeline Approach):
   Pra-pemrosesan diisolasi strictly hanya pada Training Set menggunakan Pipeline.
3. Analisis dampak: Mengapa data leakage menghasilkan optimisme palsu dan kegagalan di produksi.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


def generate_synthetic_leakage_experiment(n_samples: int = 200, n_features: int = 500, seed: int = 42):
    """
    Eksperimen klasik Guyon/Hastie tentang feature selection / preprocessing leakage:
    Dataset murni noise acak (tidak ada hubungan nyata antara X dan y).
    Jika fitur diseleksi / diskalakan sebelum split, akurasi tampak tinggi palsu.
    """
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    y = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    return X, y


def test_leakage_on_telemetry():
    print("--- 1. EKSPERIMEN PADA DATASET TELEMETRI INDUSTRI ---")
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    df = pd.read_csv(data_path)
    
    features = ["operating_hours", "vibration_rms", "temperature_c", "pressure_kpa"]
    X = df[features].copy()
    y = df["failure"].values
    
    # K-Fold CV untuk evaluasi perbandingan
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # A. Skenario LEAKY: Imputasi dan scaling dilakukan secara global sebelum CV
    imputer_global = SimpleImputer(strategy="median").fit(X)
    X_imputed = imputer_global.transform(X)
    scaler_global = StandardScaler().fit(X_imputed)
    X_leaky = scaler_global.transform(X_imputed)
    
    leaky_scores = []
    for train_idx, test_idx in kf.split(X_leaky):
        clf = LogisticRegression(class_weight="balanced", random_state=42)
        clf.fit(X_leaky[train_idx], y[train_idx])
        pred = clf.predict(X_leaky[test_idx])
        leaky_scores.append(f1_score(y[test_idx], pred, zero_division=0))
        
    # B. Skenario CLEAN: Imputasi dan scaling dibungkus dalam Pipeline (fit hanya pada fold train)
    clean_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42))
    ])
    
    clean_scores = []
    for train_idx, test_idx in kf.split(X):
        clean_pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = clean_pipe.predict(X.iloc[test_idx])
        clean_scores.append(f1_score(y[test_idx], pred, zero_division=0))
        
    print(f"Skor F1 CV - Leaky Global Preprocessing : Mean = {np.mean(leaky_scores):.4f} (+/- {np.std(leaky_scores):.4f})")
    print(f"Skor F1 CV - Clean In-Pipeline Preprocessing: Mean = {np.mean(clean_scores):.4f} (+/- {np.std(clean_scores):.4f})\n")


def test_extreme_leakage_demonstration():
    print("--- 2. DEMONSTRASI EKSTREM: SELEKSI FITUR DENGAN NOISE MURNI ---")
    print("Skenario: 200 sampel acak dengan 500 fitur noise murni (Ground Truth = 50% Random Guess).")
    X, y = generate_synthetic_leakage_experiment(200, 500, seed=42)
    
    # Pendekatan LEAKY: Pilih 10 fitur berkorelasi tertinggi dari SELURUH dataset sebelum split
    corrs = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
    top_leaky_features = np.argsort(corrs)[-10:]
    
    X_leaky_subset = X[:, top_leaky_features]
    X_tr_leak, X_ts_leak, y_tr_leak, y_ts_leak = train_test_split(X_leaky_subset, y, test_size=0.30, random_state=42)
    
    clf_leaky = LogisticRegression()
    clf_leaky.fit(X_tr_leak, y_tr_leak)
    acc_leaky_test = clf_leaky.score(X_ts_leak, y_ts_leak)
    
    # Pendekatan CLEAN: Pilih 10 fitur HANYA dari data latih setelah split
    X_tr_raw, X_ts_raw, y_tr_raw, y_ts_raw = train_test_split(X, y, test_size=0.30, random_state=42)
    corrs_train = np.array([np.abs(np.corrcoef(X_tr_raw[:, i], y_tr_raw)[0, 1]) for i in range(X_tr_raw.shape[1])])
    top_clean_features = np.argsort(corrs_train)[-10:]
    
    clf_clean = LogisticRegression()
    clf_clean.fit(X_tr_raw[:, top_clean_features], y_tr_raw)
    acc_clean_test = clf_clean.score(X_ts_raw[:, top_clean_features], y_ts_raw)
    
    print(f"Hasil Akurasi Uji pada Noise Murni:")
    print(f"  * Metode LEAKY (Bocor) : {acc_leaky_test*100:.2f}% (Tampak 'Hebat' padahal sepenuhnya ilusi/palsu!)")
    print(f"  * Metode CLEAN (Jujur) : {acc_clean_test*100:.2f}% (Mencerminkan realitas data murni random ~50%)")
    print("\nKesimpulan Mutlak:")
    print("Data leakage dapat memanipulasi metrik dari tebakan acak murni (50%) menjadi ilusi performa tinggi.")
    print("Selalu bungkus transformasi dalam Scikit-learn Pipeline!")


def main():
    print("=" * 75)
    print("DEMONSTRASI BAHAYA DATA LEAKAGE & PENCEGAHAN METODOLOGIS")
    print("=" * 75)
    test_leakage_on_telemetry()
    test_extreme_leakage_demonstration()
    print("=" * 75)


if __name__ == "__main__":
    main()

