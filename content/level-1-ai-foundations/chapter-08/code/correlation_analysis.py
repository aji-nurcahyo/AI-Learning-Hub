"""Correlation Analysis & Heatmap Generation.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan:
1. Perhitungan Koefisien Korelasi Pearson r Antar Fitur Numerik
2. Pembentukan Matriks Korelasi (Correlation Matrix)
3. Pembuatan dan Ekspor Visualisasi Heatmap Korelasi Menggunakan Matplotlib Murni
4. Prinsip Kritis: Korelasi Bukan Berarti Kausalitas (Correlation != Causation)
5. Dampak Outlier Ekstrem terhadap Nilai Koefisien Korelasi Linier
"""

import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def run_correlation_analysis():
    print("=" * 75)
    print("PRAKTIK 8: ANALISIS KORELASI & PEMBUATAN HEATMAP KORELASI")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"
    output_png = base_dir / "data" / "telemetry_correlation_heatmap.png"

    df = pd.read_csv(raw_csv).drop_duplicates().reset_index(drop=True)
    df["zone"] = df["zone"].astype(str).str.strip().str.title()
    df["temperature_c"] = df["temperature_c"].fillna(df.groupby("zone")["temperature_c"].transform("median"))
    df["humidity_pct"] = df["humidity_pct"].fillna(df.groupby("zone")["humidity_pct"].transform("median"))

    numeric_cols = ["temperature_c", "humidity_pct", "vibration_mms", "power_kw"]
    corr_matrix = df[numeric_cols].corr(method="pearson").round(3)

    print("--- 1. Matriks Korelasi Pearson (r) ---")
    print(corr_matrix.to_string())
    print()

    print("--- 2. Interpretasi Pasangan Fitur Kunci ---")
    r_temp_vibe = corr_matrix.loc["temperature_c", "vibration_mms"]
    r_temp_power = corr_matrix.loc["temperature_c", "power_kw"]
    r_vibe_power = corr_matrix.loc["vibration_mms", "power_kw"]
    r_temp_humid = corr_matrix.loc["temperature_c", "humidity_pct"]

    print(f"a. Suhu vs Getaran (r = {r_temp_vibe:+.3f}) : Korelasi positif kuat (gesekan mekanis memicu panas).")
    print(f"b. Getaran vs Daya  (r = {r_vibe_power:+.3f}) : Korelasi positif sangat kuat (turbin berdaya tinggi bergetar lebih besar).")
    print(f"c. Suhu vs Kelembaban (r = {r_temp_humid:+.3f}) : Korelasi positif moderat di ruang mesin.\n")

    # Membuat Heatmap Visual Menggunakan Matplotlib Murni
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cax = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Koefisien Korelasi Pearson (r)")

    # Label sumbu
    col_labels = ["Suhu (°C)", "Kelembaban (%)", "Getaran (mm/s)", "Daya (kW)"]
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(col_labels, fontsize=10, fontweight="bold")

    # Menambahkan teks angka di dalam tiap sel heatmap
    for i in range(len(col_labels)):
        for j in range(len(col_labels)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")

    ax.set_title("Heatmap Matriks Korelasi Fitur Telemetri Sensor", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap korelasi tersimpan ke: {output_png.name} ({output_png.stat().st_size} bytes)")
    print("\n[PERINGATAN METODOLOGIS]:")
    print("1. KORELASI BUKAN KAUSALITAS: Angka r tinggi tidak membuktikan bahwa variabel X secara langsung")
    print("   menyebabkan perubahan Y (keduanya mungkin dipengaruhi variabel tersembunyi/confounder).")
    print("2. Pearson hanya mengukur hubungan linear. Pola non-linear (misal kurva parabola U-shape)")
    print("   dapat memiliki nilai r mendekati 0 meskipun kedua variabel sangat bergantung erat!")
    print("=" * 75)


if __name__ == "__main__":
    run_correlation_analysis()

