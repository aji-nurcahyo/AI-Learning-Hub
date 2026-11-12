"""Outlier Detection: IQR Method, Z-Score & Contextual Evaluation.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan:
1. Deteksi Pencilan Univariat menggunakan Aturan Interquartile Range (IQR Tukey)
2. Deteksi Pencilan menggunakan Standarisasi Z-Score (|Z| > threshold)
3. Evaluasi Kontekstual: Mengapa Penghapusan Otomatis (Blind Removal) Berbahaya
4. Strategi Penanganan: Capping (Winsorization) vs Penandaan Anomali (Flagging)
"""

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def detect_and_analyze_outliers():
    print("=" * 75)
    print("PRAKTIK 4: DETEKSI PENCILAN (OUTLIERS) & EVALUASI KONTEKSTUAL")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"
    df = pd.read_csv(raw_csv).drop_duplicates().reset_index(drop=True)

    # Menangani missing values sementara dengan median
    temp_series = df["temperature_c"].fillna(df["temperature_c"].median())

    # 1. Metode Interquartile Range (IQR)
    q1 = temp_series.quantile(0.25)
    q3 = temp_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    iqr_outliers = df[temp_series > upper_bound]

    print("--- 1. Deteksi Berbasis IQR (Metode Tukey) ---")
    print(f"Kuartil 1 (Q1) : {q1:.2f} °C")
    print(f"Kuartil 3 (Q3) : {q3:.2f} °C")
    print(f"IQR (Q3 - Q1)  : {iqr:.2f} °C")
    print(f"Pagar Bawah    : {lower_bound:.2f} °C")
    print(f"Pagar Atas     : {upper_bound:.2f} °C")
    print(f"Pencilan Terdeteksi IQR: {len(iqr_outliers)} rekaman")
    for _, row in iqr_outliers.iterrows():
        print(f"  -> Device: {row['device_id']}, Waktu: {row['timestamp']}, Suhu: {row['temperature_c']} °C, Getaran: {row['vibration_mms']} mm/s, Daya: {row['power_kw']} kW")
    print()

    # 2. Metode Z-Score
    mean_val = temp_series.mean()
    std_val = temp_series.std()
    z_scores = (temp_series - mean_val) / std_val
    z_threshold = 2.5
    z_outliers = df[z_scores.abs() > z_threshold]

    print(f"--- 2. Deteksi Berbasis Z-Score (|Z| > {z_threshold}) ---")
    print(f"Rata-rata (Mean) : {mean_val:.2f} °C, Standar Deviasi : {std_val:.2f} °C")
    print(f"Pencilan Terdeteksi Z-Score: {len(z_outliers)} rekaman")
    for idx, row in z_outliers.iterrows():
        print(f"  -> Device: {row['device_id']}, Suhu: {row['temperature_c']} °C, Nilai Z-Score: {z_scores.loc[idx]:.2f}")
    print()

    # 3. Analisis Kontekstual & Strategi Penanganan
    print("--- 3. Analisis Kontekstual Domain Industri ---")
    print("Pertanyaan Kritis:")
    print("Apakah suhu 88.5 °C pada DEV-104 (Turbine Bay) adalah kerusakan sensor (glitch)")
    print("ataukah indikasi lonjakan bahaya kegagalan turbin mekanis yang sesungguhnya?")
    print("Pemeriksaan Fitur Pendukung:")
    outlier_row = iqr_outliers.iloc[0]
    print(f"  - Tingkat Getaran Mekanis: {outlier_row['vibration_mms']} mm/s (Sangat Tinggi, normal < 1.0)")
    print(f"  - Konsumsi Daya Turbin   : {outlier_row['power_kw']} kW (Puncak beban tertinggi, normal ~80 kW)")
    print("Kesimpulan: Suhu 88.5 °C BUKAN galat sensor acak, melainkan kejadian anomali mesin nyata!")
    print("Menghapus baris ini secara membabi buta (*blind removal*) akan menghilangkan data kegagalan kritis yang vital untuk model deteksi dini kegagalan!\n")

    # 4. Strategi Capping / Winsorization (Alternatif Aman jika Wajib Ditransformasi)
    df_capped = df.copy()
    df_capped["temp_capped"] = temp_series.clip(lower=lower_bound, upper=upper_bound)
    print("--- 4. Demonstrasi Strategi Capping (Winsorization) ---")
    print(f"Nilai asli 88.5 °C dipotong (capped) menjadi batas atas wajar: {upper_bound:.2f} °C")
    print(f"Mean sebelum capping: {temp_series.mean():.2f} °C -> Mean setelah capping: {df_capped['temp_capped'].mean():.2f} °C")
    print("=" * 75)


if __name__ == "__main__":
    detect_and_analyze_outliers()

