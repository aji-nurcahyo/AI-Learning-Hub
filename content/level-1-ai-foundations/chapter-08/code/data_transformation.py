"""Data Transformation: Min-Max Scaling, Standardization & Log Transformation.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan transformasi data:
1. Min-Max Scaling (Penskalaan Rentang [0, 1])
2. Standarisasi Z-Score (Mean = 0, Std = 1)
3. Mengapa Min-Max Sensitif terhadap Outlier vs Keunggulan Standarisasi
4. Transformasi Logaritmik (Log Transformation) untuk Menangani Sebaran Miring (Skewed Distribution)
5. Perhitungan Metrik Kemiringan (Skewness Coefficient) Sebelum dan Sesudah Transformasi
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def run_data_transformations():
    print("=" * 75)
    print("PRAKTIK 5: TRANSFORMASI DATA — SCALING, STANDARISASI & LOG TRANSFORMATION")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"
    df = pd.read_csv(raw_csv).drop_duplicates().reset_index(drop=True)

    # Menangani missing values sementara
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].median())
    df["humidity_pct"] = df["humidity_pct"].fillna(df["humidity_pct"].median())

    # 1. Min-Max Scaling: x' = (x - min) / (max - min)
    temp = df["temperature_c"]
    temp_min_max = (temp - temp.min()) / (temp.max() - temp.min())

    # 2. Standarisasi Z-Score: z = (x - mean) / std
    temp_z_score = (temp - temp.mean()) / temp.std()

    print("--- 1. Penskalaan Fitur: Min-Max vs Standarisasi Z-Score ---")
    scaling_comp = pd.DataFrame({
        "Suhu Asli (°C)": temp.head(6),
        "Min-Max [0, 1]": temp_min_max.head(6).round(4),
        "Z-Score (μ=0, σ=1)": temp_z_score.head(6).round(4)
    })
    print(scaling_comp)
    print()

    print("[ANALISIS SENSITIVITAS OUTLIER]:")
    print("Karena ada outlier 88.5 °C, nilai suhu normal (22-23 °C) tertekan menjadi ~0.00-0.01 pada Min-Max.")
    print("Pada Z-score, titik normal tetap memiliki variasi wajar sekitar -0.5 hingga -0.4!\n")

    # 3. Transformasi Logaritmik pada Fitur 'power_kw' yang Miring (Skewed)
    power = df["power_kw"]
    power_skew_before = power.skew()
    log_power = np.log(power)  # Syarat log: data harus bernilai positif strictly x > 0
    power_skew_after = log_power.skew()

    print("--- 2. Transformasi Logaritmik pada Fitur Daya Listrik (power_kw) ---")
    print(f"Kemiringan (Skewness) Sebelum Transformasi : {power_skew_before:.4f} (Miring Kanan Positif)")
    print(f"Kemiringan (Skewness) Setelah Log(power)   : {power_skew_after:.4f} (Mendekati Simetris Normal!)")
    print()

    log_comp = pd.DataFrame({
        "Power Asli (kW)": power.head(6),
        "Log(Power)": log_power.head(6).round(4)
    })
    print(log_comp)
    print("\nCatatan: Transformasi logaritmik merapatkan ekor kanan data sehingga")
    print("model AI yang mengasumsikan distribusi normal tidak terdistorsi oleh nilai besar.")
    print("=" * 75)


if __name__ == "__main__":
    run_data_transformations()

