"""Data Inspection & Quality Profiling.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan tahapan inspeksi awal (profiling) dataset mentah:
1. Pemuatan berkas CSV menggunakan jalur relatif aman (pathlib)
2. Pemeriksaan dimensi, tipe data (dtypes), dan penggunaan memori
3. Deteksi keberadaan nilai hilang (missing values) dan persentasenya
4. Identifikasi baris duplikat (exact duplicates)
5. Ringkasan statistik lima angka (five-number summary)
6. Deteksi inkonsistensi label kategorikal (capitalization mismatch)
"""

import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def inspect_dataset():
    print("=" * 75)
    print("PRAKTIK 1: INSPEKSI & PROFILING KUALITAS DATA AWAL (DATA PROFILING)")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"

    if not raw_csv.exists():
        raise FileNotFoundError(f"Berkas tidak ditemukan: {raw_csv}")

    df = pd.read_csv(raw_csv)

    print("--- 1. Dimensi & Metadata Tabel ---")
    print(f"Jalur Berkas      : {raw_csv.name}")
    print(f"Bentuk (Shape)    : {df.shape[0]} baris, {df.shape[1]} kolom")
    mem_usage = df.memory_usage(deep=True).sum() / 1024
    print(f"Estimasi Memori   : {mem_usage:.2f} KB\n")

    print("Tipe Data Kolom (Dtypes):")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:<18} : {dtype}")
    print()

    print("--- 2. Deteksi Nilai Hilang (Missing Values) ---")
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_pct.round(2)})
    print(missing_df[missing_df["Missing Count"] > 0])
    total_missing = missing_counts.sum()
    print(f"Total Nilai Hilang: {total_missing} sel dari {df.size} total sel ({total_missing / df.size * 100:.2f}%)\n")

    print("--- 3. Deteksi Rekaman Duplikat ---")
    exact_duplicates = df.duplicated().sum()
    print(f"Jumlah Baris Duplikat Tepat (Exact Duplicates): {exact_duplicates} baris")
    if exact_duplicates > 0:
        print("Baris yang teridentifikasi duplikat:")
        print(df[df.duplicated(keep=False)])
    print()

    print("--- 4. Deteksi Inkonsistensi Kategori Teks ---")
    print("Nilai unik pada kolom 'zone' (perhatikan inkonsistensi huruf kapital):")
    zone_counts = df["zone"].value_counts()
    for zone_name, count in zone_counts.items():
        print(f"  - '{zone_name}' : {count} rekaman")
    print("Catatan: 'Server Room', 'server room', dan 'SERVER ROOM' merujuk ke entitas yang sama!\n")

    print("--- 5. Ringkasan Statistik Numerik Awal (.describe()) ---")
    numeric_summary = df.describe().round(2)
    print(numeric_summary)
    print()

    print("[INSIGHT INSPEKSI]:")
    print("1. Nilai maksimum 'temperature_c' adalah 88.5 °C (potensi outlier ekstrem dibanding Q3=31.4 °C).")
    print("2. Kolom 'power_kw' memiliki rentang lebar (14.1 s/d 210.5 kW) yang mengindikasikan sebaran skewed.")
    print("3. Terdapat 1 baris duplikat identik pada DEV-105 pukul 08:15:00.")
    print("=" * 75)


if __name__ == "__main__":
    inspect_dataset()

