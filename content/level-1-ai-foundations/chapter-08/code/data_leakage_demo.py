"""Data Leakage in Preprocessing: Pitfalls & Prevention.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan fenomena Kebocoran Data (Data Leakage):
1. Mengapa Urutan Tahapan Prapemrosesan Sangat Menentukan Integritas Model
2. Skenario Salah: Memanggil fit() Sebelum train_test_split (Leaked Pipeline)
3. Skenario Benar: Pemisahan Data Terlebih Dahulu, fit() HANYA pada Data Latih (Isolated Pipeline)
4. Perbandingan Kuantitatif Parameter Statistik (Mean & Std Dev) Antar Skenario
5. Daftar Periksa (Checklist) Pencegahan Kebocoran Data dalam Rekayasa AI
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.stdout.reconfigure(encoding="utf-8")


def demonstrate_data_leakage():
    print("=" * 75)
    print("PRAKTIK 9: DEMONSTRASI & PENCEGAHAN KEBOCORAN DATA (DATA LEAKAGE)")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"
    df = pd.read_csv(raw_csv).drop_duplicates().reset_index(drop=True)

    # Fitur numerik yang akan diproses
    features = ["temperature_c", "vibration_mms"]
    X = df[features].values
    # Target sintetis: 1 jika getaran tinggi (> 0.5), 0 jika getaran rendah
    y = (df["vibration_mms"] > 0.5).astype(int).values

    print(f"Total Dataset: {len(X)} sampel.")
    print("Membagi data: 70% Train (16 sampel), 30% Test (7 sampel), random_state=42\n")

    # =========================================================================
    # SKENARIO 1: POLA SALAH — FIT PADA SELURUH DATASET (DATA LEAKAGE)
    # =========================================================================
    print("--- 1. SKENARIO 1: POLA SALAH (DATA LEAKAGE TERJADI) ---")
    print("Urutan Salah: Imputer & Scaler di-fit() pada SEMUA data SEBELUM split:")

    # Kesalahan fatal: fit pada seluruh X
    leaked_imputer = SimpleImputer(strategy="median")
    leaked_scaler = StandardScaler()

    X_leaked_imputed = leaked_imputer.fit_transform(X)
    X_leaked_scaled = leaked_scaler.fit_transform(X_leaked_imputed)

    # Split baru dilakukan setelah fit
    X_train_leaked, X_test_leaked, y_train_1, y_test_1 = train_test_split(
        X_leaked_scaled, y, test_size=0.30, random_state=42
    )

    leaked_mean_temp = leaked_scaler.mean_[0]
    leaked_var_temp = leaked_scaler.var_[0]
    print(f"  -> Mean Suhu yang Dipelajari Scaler   : {leaked_mean_temp:.4f} °C (Terkontaminasi data uji!)")
    print(f"  -> Varians Suhu yang Dipelajari Scaler: {leaked_var_temp:.4f}\n")

    # =========================================================================
    # SKENARIO 2: POLA BENAR — SPLIT DAHULU, FIT HANYA PADA DATA LATIH
    # =========================================================================
    print("--- 2. SKENARIO 2: POLA BENAR (TERISOLASI SEMPURNA) ---")
    print("Urutan Benar: train_test_split DILAKUKAN TERLEBIH DAHULU:")

    # 1. Split data mentah pertama kali
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # 2. Fit imputer dan scaler HANYA pada data latih
    correct_imputer = SimpleImputer(strategy="median")
    correct_scaler = StandardScaler()

    X_train_imputed = correct_imputer.fit_transform(X_train_raw)
    X_train_scaled = correct_scaler.fit_transform(X_train_imputed)

    # 3. Transform data uji menggunakan parameter yang dipelajari DARI DATA LATIH
    X_test_imputed = correct_imputer.transform(X_test_raw)
    X_test_scaled = correct_scaler.transform(X_test_imputed)

    correct_mean_temp = correct_scaler.mean_[0]
    correct_var_temp = correct_scaler.var_[0]
    print(f"  -> Mean Suhu yang Dipelajari Scaler   : {correct_mean_temp:.4f} °C (Murni dari data latih)")
    print(f"  -> Varians Suhu yang Dipelajari Scaler: {correct_var_temp:.4f}\n")

    # =========================================================================
    # ANALISIS PERBEDAAN
    # =========================================================================
    print("--- 3. Analisis Dampak Kebocoran Data ---")
    diff_mean = abs(leaked_mean_temp - correct_mean_temp)
    print(f"Selisih Parameter Mean yang Dipelajari : {diff_mean:.4f} °C")
    print("Konsekuensi:")
    print("Pada Skenario 1, model menerima 'bocoran informasi statistik' mengenai")
    print("distribusi data uji sebelum proses evaluasi. Hal ini menyebabkan estimasi performa")
    print("model tampak terlalu percaya diri (over-optimistic) pada lingkungan lab,")
    print("namun mengalami penurunan performa drastis ketika di-deploy ke sistem produksi nyata!\n")

    print("[CHECKLIST PENCEGAHAN DATA LEAKAGE]:")
    print("1. [x] Lakukan train/test split SEBELUM langkah prapemrosesan apapun.")
    print("2. [x] Selalu panggil .fit() atau .fit_transform() HANYA pada data latih (X_train).")
    print("3. [x] Panggil .transform() (tanpa fit) pada data validasi dan data uji (X_test).")
    print("4. [x] Jangan pernah menghitung statistik global (mean/max) sebelum partisi data.")
    print("=" * 75)


if __name__ == "__main__":
    demonstrate_data_leakage()

