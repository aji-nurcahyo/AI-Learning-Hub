"""Pandas CSV Processing: Handling Missing Values, GroupBy & Feature Creation.

Chapter 07: Python Ecosystem for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan:
1. Membaca Data CSV Menggunakan Path Relatif Aman (pathlib)
2. Deteksi dan Imputasi Nilai Hilang (Missing Values Handling)
3. Operasi Agregasi Terkelompok (GroupBy Operations)
4. Rekayasa Fitur Numerik Sederhana (Feature Engineering)
5. Ekspor Data Bersih ke Berkas CSV Baru
"""

import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def run_pandas_csv_demo():
    print("=" * 70)
    print("DEMO 4: PANDAS CSV HANDLING — MISSING VALUES, GROUPBY & EXPORT")
    print("=" * 70)

    # 1. Menentukan Lokasi File Menggunakan Pathlib Relatif
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data" / "sensor_readings.csv"
    output_path = base_dir / "data" / "sensor_readings_clean.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Berkas dataset tidak ditemukan: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    print("--- 1. Membaca Berkas CSV Mentah ---")
    print(f"Jalur Berkas      : {csv_path.name}")
    print(f"Jumlah Baris/Kolom: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
    print("5 Baris Pertama:\n", df_raw.head(), "\n")

    # 2. Deteksi Nilai Hilang (Missing Values)
    missing_counts = df_raw.isna().sum()
    print("--- 2. Deteksi Nilai Hilang (Missing Values) ---")
    print(missing_counts[missing_counts > 0])
    print(f"Total Nilai Hilang di Dataset: {df_raw.isna().sum().sum()} elemen\n")

    # Imputasi Nilai Hilang dengan Nilai Rata-rata Kolom Masing-Masing
    df_clean = df_raw.copy()
    mean_temp = df_clean["temperature_c"].mean()
    mean_humid = df_clean["humidity_pct"].mean()

    df_clean["temperature_c"] = df_clean["temperature_c"].fillna(round(mean_temp, 1))
    df_clean["humidity_pct"] = df_clean["humidity_pct"].fillna(round(mean_humid, 1))

    print("Hasil Setelah Imputasi (Missing Values = 0):")
    print(f"Sisa Missing Values: {df_clean.isna().sum().sum()} elemen\n")

    # 3. Operasi Agregasi GroupBy
    print("--- 3. Analisis Terkelompok Berdasarkan Lokasi (GroupBy) ---")
    grouped = df_clean.groupby("location").agg({
        "temperature_c": ["mean", "max"],
        "vibration_mms": ["mean", "max"],
        "sensor_id": "count"
    }).rename(columns={"count": "total_readings"})

    print(grouped.round(2), "\n")

    # 4. Rekayasa Fitur (Feature Engineering Sederhana)
    # Menghitung deviasi suhu dari rata-rata global
    global_mean_temp = df_clean["temperature_c"].mean()
    df_clean["temp_deviation"] = (df_clean["temperature_c"] - global_mean_temp).round(2)

    # Indikator biner anomali: 1 jika bukan NORMAL, 0 jika NORMAL
    df_clean["is_anomaly"] = (df_clean["status"] != "NORMAL").astype(int)

    print("--- 4. Kolom Fitur Baru yang Ditambahkan ---")
    print(df_clean[["sensor_id", "location", "temperature_c", "temp_deviation", "is_anomaly"]].head(6), "\n")

    # 5. Ekspor Data Bersih
    df_clean.to_csv(output_path, index=False)
    print("--- 5. Ekspor Data Bersih ---")
    print(f"Data bersih berhasil disimpan ke: {output_path.name}")
    print(f"Ukuran Berkas Tersimpan        : {output_path.stat().st_size} bytes")
    print("=" * 70)


if __name__ == "__main__":
    run_pandas_csv_demo()

