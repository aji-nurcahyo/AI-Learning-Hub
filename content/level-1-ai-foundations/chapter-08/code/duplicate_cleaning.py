"""Duplicate Removal, Text Standardization & Validation Checks.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan pembersihan duplikasi dan inkonsistensi data:
1. Deteksi dan Penghapusan Duplikat Tepat (Exact Duplicates)
2. Deteksi Duplikat Parsial pada Kunci Primer Pengamatan (device_id + timestamp)
3. Standardisasi String Kategori Teks (Trimming & Title Casing)
4. Pemeriksaan Validitas Batas Rentang Fisik (Physical Range Validation)
5. Pencatatan Jejak Audit Pembersihan (Data Cleaning Audit Trail)
"""

import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def clean_duplicates_and_inconsistencies():
    print("=" * 75)
    print("PRAKTIK 3: PEMBERSIHAN DUPLIKASI, STANDARISASI TEKS & VALIDASI RENTANG")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"
    df = pd.read_csv(raw_csv)
    audit_trail = []

    print(f"Dataset Awal: {len(df)} baris.")

    # 1. Deteksi & Hapus Exact Duplicates
    exact_dups = df.duplicated().sum()
    audit_trail.append(f"1. Deteksi Duplikat Tepat: Ditemukan {exact_dups} baris identik.")
    if exact_dups > 0:
        df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
        audit_trail.append(f"   Tindakan: Menghapus {exact_dups} baris duplikat (Baris tersisa: {len(df_clean)}).")
    else:
        df_clean = df.copy()

    # 2. Deteksi Duplikat Parsial pada (device_id, timestamp)
    partial_dups = df_clean.duplicated(subset=["device_id", "timestamp"]).sum()
    audit_trail.append(f"2. Pemeriksaan Kunci Primer (device_id + timestamp): Ditemukan {partial_dups} konflik rekaman.")

    # 3. Standardisasi Kategori Teks Kolom 'zone'
    zones_before = df_clean["zone"].unique().tolist()
    audit_trail.append(f"3. Nilai Unik 'zone' Sebelum Harmonisasi: {zones_before}")
    df_clean["zone"] = df_clean["zone"].astype(str).str.strip().str.title()
    zones_after = df_clean["zone"].unique().tolist()
    audit_trail.append(f"   Nilai Unik 'zone' Setelah Harmonisasi: {zones_after}")

    # 4. Validasi Batas Rentang Fisik (Domain Range Checking)
    # Suhu wajar lingkungan industri: -10 °C s/d 100 °C
    # Kelembaban: 0% s/d 100%
    # Getaran: >= 0 mm/s
    invalid_temp = ((df_clean["temperature_c"] < -10.0) | (df_clean["temperature_c"] > 100.0)).sum()
    invalid_humid = ((df_clean["humidity_pct"] < 0.0) | (df_clean["humidity_pct"] > 100.0)).sum()
    invalid_vibe = (df_clean["vibration_mms"] < 0.0).sum()

    audit_trail.append(f"4. Validasi Rentang Fisik: Suhu Luar Rentang = {invalid_temp}, Kelembaban Ilegal = {invalid_humid}, Getaran Negatif = {invalid_vibe}.")

    print("--- Ringkasan Jejak Audit Pembersihan (Audit Trail) ---")
    for entry in audit_trail:
        print(entry)
    print()

    print("Distribusi Frekuensi Zona Bersih:")
    print(df_clean["zone"].value_counts())
    print(f"\nTotal Baris Bersih Siap Pakai: {len(df_clean)} baris (Berkurang 1 baris duplikat).")
    print("=" * 75)


if __name__ == "__main__":
    clean_duplicates_and_inconsistencies()

