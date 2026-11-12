"""Missing Data Analysis & Imputation Strategies.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan evaluasi dan penanganan nilai hilang:
1. Pemahaman Konseptual Mekanisme Pelesapan (MCAR, MAR, MNAR)
2. Komparasi Strategi: Listwise Deletion (dropna) vs Imputasi Statistik
3. Mengapa Median Lebih Unggul daripada Mean saat Terdapat Outlier Ekstrem
4. Hubungan Harmonis: Standardisasi Teks Kategori Sebelum Imputasi Kelompok
5. Penambahan Kolom Indikator Pelesapan (Missing Indicator Feature)
"""

import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def handle_missing_data():
    print("=" * 75)
    print("PRAKTIK 2: ANALISIS & STRATEGI PENANGANAN NILAI HILANG (MISSING DATA)")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"
    df = pd.read_csv(raw_csv)

    print("--- 1. Mekanisme Pelesapan Data (Missingness Mechanisms) ---")
    print("a. MCAR (Missing Completely at Random): Kehilangan murni acak tanpa pola.")
    print("b. MAR  (Missing at Random)          : Kehilangan berkorelasi dengan variabel lain.")
    print("c. MNAR (Missing Not at Random)      : Kehilangan berkorelasi dengan nilai itu sendiri.")
    print("   (Contoh MNAR: Sensor crash saat suhu melonjak ekstrem melebihi daya ukur)\n")

    print(f"Dataset Awal: {len(df)} baris. Baris dengan nilai hilang:")
    missing_rows = df[df.isna().any(axis=1)]
    print(missing_rows[["device_id", "timestamp", "zone", "temperature_c", "humidity_pct"]])
    print()

    # Strategi A: Listwise Deletion
    df_dropped = df.dropna()
    print("--- 2. Strategi A: Listwise Deletion (df.dropna()) ---")
    print(f"Jumlah baris sebelum: {len(df)}, setelah penghapusan: {len(df_dropped)}")
    print(f"Data hilang terbuang: {len(df) - len(df_dropped)} baris ({((len(df)-len(df_dropped))/len(df))*100:.1f}%)")
    print("Peringatan: Listwise deletion dapat memangkas ukuran sampel berharga!\n")

    # Strategi B: Mean vs Median Imputation
    temp_mean = df["temperature_c"].mean()
    temp_median = df["temperature_c"].median()
    print("--- 3. Strategi B: Evaluasi Mean vs Median Imputation ---")
    print(f"Mean 'temperature_c'   : {temp_mean:.2f} °C (Terangkat oleh outlier 88.5 °C!)")
    print(f"Median 'temperature_c' : {temp_median:.2f} °C (Resisten terhadap outlier!)")
    print("Keputusan Metodologis : Memilih MEDIAN agar tidak mendistorsi distribusi data suhu normal.\n")

    # Strategi C: Standardisasi Kategori + Imputasi Terkelompok per Zona
    df_imputed = df.copy()
    # Harmonisasi teks kategori zona ke Title Case terlebih dahulu
    df_imputed["zone_clean"] = df_imputed["zone"].astype(str).str.strip().str.title()

    # Mengisi kelembaban dengan median per zona yang sudah bersih
    df_imputed["humidity_pct"] = df_imputed.groupby("zone_clean")["humidity_pct"].transform(
        lambda s: s.fillna(s.median())
    )
    # Mengisi suhu dengan median per zona yang sudah bersih
    df_imputed["temperature_c"] = df_imputed.groupby("zone_clean")["temperature_c"].transform(
        lambda s: s.fillna(s.median())
    )

    # Strategi D: Missing Indicator Flag
    df_imputed["temp_was_missing"] = df["temperature_c"].isna().astype(int)

    print("--- 4. Hasil Imputasi Cerdas & Indikator Pelesapan ---")
    imputed_rows = df_imputed.loc[missing_rows.index]
    print(imputed_rows[["device_id", "timestamp", "zone_clean", "temperature_c", "humidity_pct", "temp_was_missing"]])
    print()

    total_remaining_missing = df_imputed[["temperature_c", "humidity_pct"]].isna().sum().sum()
    print(f"Sisa missing values di dataset: {total_remaining_missing} sel (Tuntas 100%!)")
    print("=" * 75)


if __name__ == "__main__":
    handle_missing_data()
