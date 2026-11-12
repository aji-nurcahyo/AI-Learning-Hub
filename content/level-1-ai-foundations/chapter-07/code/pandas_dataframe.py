"""Pandas Fundamentals: Series, DataFrames & Data Inspection.

Chapter 07: Python Ecosystem for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan:
1. Pembuatan Pandas Series dan DataFrame
2. Anatomi DataFrame (Index, Columns, Values ndarray)
3. Inspeksi Struktur Data (head, info, describe, dtypes)
4. Pengaksesan Data: Kolom Tunggal, Multi-Kolom, .loc vs .iloc
5. Penyaringan Bersyarat (Conditional Filtering)
6. Ringkasan Frekuensi Kategori (value_counts)
"""

import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def run_pandas_dataframe_demo():
    print("=" * 70)
    print("DEMO 3: PANDAS FUNDAMENTALS — DATAFRAME ANATOMY & INSPECTION")
    print("=" * 70)

    # 1. Pembuatan DataFrame dari Dictionary
    raw_dict = {
        "sensor_id": ["S01", "S02", "S03", "S04", "S05", "S06"],
        "location": ["Room-A", "Room-B", "Bay-1", "Bay-2", "Room-A", "Bay-2"],
        "temperature_c": [22.4, 23.1, 28.5, 31.2, 22.6, 36.5],
        "vibration_mms": [0.12, 0.15, 0.85, 1.42, 0.11, 2.75],
        "status": ["NORMAL", "NORMAL", "NORMAL", "WARNING", "NORMAL", "CRITICAL"]
    }
    df = pd.DataFrame(raw_dict)

    print("--- 1. Tampilan Awal DataFrame ---")
    print(df.to_string(index=True), "\n")

    # 2. Anatomi DataFrame
    print("--- 2. Anatomi DataFrame ---")
    print(f"Index (Label Baris)   : {df.index.tolist()}")
    print(f"Columns (Nama Kolom)  : {df.columns.tolist()}")
    print(f"Shape (Dimensi)       : {df.shape} (6 baris, 5 kolom)")
    print(f"Tipe Data Underlying  : {type(df['temperature_c'].values)} (NumPy Array!)\n")

    # 3. Inspeksi Statistik Cepat
    print("--- 3. Ringkasan Statistik (.describe()) ---")
    print(df.describe().round(2), "\n")

    # 4. Pengaksesan Data: .loc vs .iloc
    print("--- 4. Pengaksesan Data (.loc vs .iloc) ---")
    # .loc berbasis label nama
    val_loc = df.loc[3, "temperature_c"]
    # .iloc berbasis indeks integer posisi
    val_iloc = df.iloc[3, 2]
    print(f"df.loc[3, 'temperature_c'] : {val_loc} °C")
    print(f"df.iloc[3, 2]              : {val_iloc} °C (Hasil identik!)\n")

    subset_cols = df[["sensor_id", "temperature_c", "status"]]
    print("Subset Kolom Tertentu:\n", subset_cols.head(3), "\n")

    # 5. Penyaringan Bersyarat (Filtering)
    # Kondisi: Suhu > 25.0 °C ATAU Status != 'NORMAL'
    anomalous = df[(df["temperature_c"] > 25.0) | (df["status"] != "NORMAL")]
    print("--- 5. Penyaringan Bersyarat (Deteksi Anomali) ---")
    print("Data dengan Suhu > 25°C ATAU Status Bukan NORMAL:")
    print(anomalous[["sensor_id", "location", "temperature_c", "vibration_mms", "status"]].to_string(index=False), "\n")

    # 6. Frekuensi Kategori
    print("--- 6. Distribusi Frekuensi Kategori Status ---")
    print(df["status"].value_counts())
    print("=" * 70)


if __name__ == "__main__":
    run_pandas_dataframe_demo()

