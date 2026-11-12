"""Categorical Data Encoding: Nominal vs Ordinal & One-Hot Encoding.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan teknik encoding fitur kategorikal:
1. Pembedaan Variabel Nominal vs Ordinal
2. Ordinal Encoding: Pemetaan Berperingkat untuk Kategori Berurutan
3. Jebakan Kesalahan Ordinal Encoding pada Variabel Nominal
4. One-Hot Encoding: Representasi Biner untuk Variabel Nominal
5. Penanganan Kategori Baru Tak Terlihat (Unseen Categories) & Dummy Variable Trap
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.stdout.reconfigure(encoding="utf-8")


def run_categorical_encoding_demo():
    print("=" * 75)
    print("PRAKTIK 6: ENCODING KATEGORIKAL — ORDINAL VS ONE-HOT ENCODING")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"
    df = pd.read_csv(raw_csv).drop_duplicates().reset_index(drop=True)
    # Harmonisasi teks zona
    df["zone"] = df["zone"].astype(str).str.strip().str.title()

    print("--- 1. Identifikasi Sifat Variabel Kategorikal ---")
    print("a. 'cooling_status' : Bersifat ORDINAL (Memiliki peringkat inheren: LOW < MEDIUM < HIGH)")
    print("b. 'zone'           : Bersifat NOMINAL (Tidak memiliki urutan matematis lebih besar/kecil)\n")

    # 1. Ordinal Encoding yang Tepat pada 'cooling_status'
    ordinal_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    df["cooling_ordinal"] = df["cooling_status"].map(ordinal_map)

    print("--- 2. Penerapan Ordinal Encoding pada 'cooling_status' ---")
    sample_ord = df[["device_id", "cooling_status", "cooling_ordinal"]].drop_duplicates()
    print(sample_ord.to_string(index=False))
    print("Hasil: Hubungan urutan fisik (LOW=1 < MEDIUM=2 < HIGH=3) terjaga dengan tepat!\n")

    # 2. Bahaya Menerapkan Ordinal Encoding pada Variabel Nominal 'zone'
    false_map = {"Server Room": 1, "Generator Bay": 2, "Turbine Bay": 3}
    print("--- 3. JEBAKAN: Ordinal Encoding pada Variabel Nominal ('zone') ---")
    print("Jika kita memetakan: Server Room -> 1, Generator Bay -> 2, Turbine Bay -> 3")
    print("Model AI secara keliru akan mengasumsikan bahwa:")
    print("  'Turbine Bay' (3) bernilai 3 KALI LIPAT lebih besar dari 'Server Room' (1)!")
    print("  'Generator Bay' (2) berada tepat di tengah rata-rata kedua lokasi tersebut.")
    print("Ini adalah asumsi matematis palsu yang merusak representasi model!\n")

    # 3. Solusi yang Benar: One-Hot Encoding pada Variabel Nominal
    print("--- 4. Solusi Benar: One-Hot Encoding pada Kolom 'zone' ---")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    zone_encoded = ohe.fit_transform(df[["zone"]])
    zone_cols = [f"zone_{cat.replace(' ', '_')}" for cat in ohe.categories_[0]]

    df_ohe = pd.DataFrame(zone_encoded, columns=zone_cols, index=df.index)
    result_combined = pd.concat([df[["device_id", "zone", "cooling_ordinal"]], df_ohe], axis=1)

    print("5 Sampel Baris Hasil Encoding Terpadu:")
    print(result_combined.head(6).to_string(index=False))
    print()

    print("[REKOMENDASI ENCODING]:")
    print("- Gunakan Ordinal Encoding HANYA jika kategori memiliki urutan hierarkis nyata.")
    print("- Gunakan One-Hot Encoding untuk kategori nominal agar tidak ada bias urutan semu.")
    print("=" * 75)


if __name__ == "__main__":
    run_categorical_encoding_demo()

