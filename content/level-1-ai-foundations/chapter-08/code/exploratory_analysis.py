"""Exploratory Data Analysis: Univariate, Bivariate & Multi-Plot Dashboard.

Chapter 08: Data Preparation & Exploratory Analysis
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan tahapan Exploratory Data Analysis (EDA):
1. Analisis Univariat: Histogram Suhu & Boxplot Getaran
2. Analisis Bivariat: Diagram Tebaran (Scatter Plot) Suhu vs Getaran
3. Analisis Multivariat Terkelompok: Distribusi Konsumsi Daya per Zona
4. Pembuatan Dashboard Visual Multi-Panel dan Ekspor ke Gambar PNG (Headless Mode)
"""

import sys
from pathlib import Path
import matplotlib

# Gunakan backend non-GUI 'Agg' untuk eksekusi headless stabil di terminal
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def run_exploratory_data_analysis():
    print("=" * 75)
    print("PRAKTIK 7: ANALISIS DATA EKSPLORATIF (EXPLORATORY DATA ANALYSIS - EDA)")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    raw_csv = base_dir / "data" / "telemetry_raw.csv"
    output_png = base_dir / "data" / "telemetry_eda_dashboard.png"

    df = pd.read_csv(raw_csv).drop_duplicates().reset_index(drop=True)
    df["zone"] = df["zone"].astype(str).str.strip().str.title()
    df["temperature_c"] = df["temperature_c"].fillna(df.groupby("zone")["temperature_c"].transform("median"))
    df["humidity_pct"] = df["humidity_pct"].fillna(df.groupby("zone")["humidity_pct"].transform("median"))

    print(f"Data Siap Dianalisis: {len(df)} rekaman valid.")

    # Membuat kanvas dashboard 2x2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))
    fig.suptitle("Dashboard Analisis Data Eksploratif (EDA) Sensor Telemetri", fontsize=15, fontweight="bold")

    # 1. Panel 1: Histogram Distribusi Suhu (Univariat)
    ax1 = axes[0, 0]
    ax1.hist(df["temperature_c"], bins=8, color="#0288D1", edgecolor="black", alpha=0.75)
    median_t = df["temperature_c"].median()
    ax1.axvline(median_t, color="red", linestyle="--", linewidth=2, label=f"Median: {median_t:.1f} °C")
    ax1.set_title("1. Distribusi Suhu (Histogram Univariat)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Suhu (°C)")
    ax1.set_ylabel("Frekuensi Pengamatan")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Panel 2: Boxplot Getaran per Zona (Bivariat)
    ax2 = axes[0, 1]
    zone_names = df["zone"].unique()
    vibe_data = [df[df["zone"] == z]["vibration_mms"].values for z in zone_names]
    ax2.boxplot(vibe_data, tick_labels=zone_names, patch_artist=True,
                boxprops=dict(facecolor="#81C784", color="black"),
                medianprops=dict(color="red", linewidth=2))
    ax2.set_title("2. Sebaran Getaran per Zona (Boxplot Bivariat)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Getaran Mekanis (mm/s)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Panel 3: Scatter Plot Korelasi Suhu vs Getaran
    ax3 = axes[1, 0]
    color_map = {"Server Room": "#1976D2", "Turbine Bay": "#D32F2F", "Generator Bay": "#F57C00"}
    for z in zone_names:
        sub = df[df["zone"] == z]
        ax3.scatter(sub["temperature_c"], sub["vibration_mms"], label=z, color=color_map.get(z, "gray"), s=70, edgecolors="black", alpha=0.8)
    ax3.set_title("3. Korelasi Suhu vs Getaran Mekanis", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Suhu (°C)")
    ax3.set_ylabel("Getaran (mm/s)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    # 4. Panel 4: Bar Chart Rata-rata Konsumsi Daya per Zona
    ax4 = axes[1, 1]
    mean_power = df.groupby("zone")["power_kw"].mean()
    bars = ax4.bar(mean_power.index, mean_power.values, color=["#FFB74D", "#4DB6AC", "#E57373"], edgecolor="black")
    ax4.set_title("4. Rata-rata Konsumsi Daya Listrik per Zona", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Daya Rata-rata (kW)")
    ax4.grid(True, axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width()/2, h + 2, f"{h:.1f} kW", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualisasi EDA berhasil disimpan ke: {output_png.name} ({output_png.stat().st_size} bytes)")
    print("\n[TEMUAN UTAMA EDA]:")
    print("1. Zona 'Turbine Bay' memiliki suhu, getaran mekanis, dan konsumsi daya tertinggi.")
    print("2. Terlihat korelasi positif kuat antara kenaikan suhu dan peningkatan getaran mekanis.")
    print("3. 'Server Room' memiliki karakteristik stabil dingin (< 24 °C) dengan getaran mendekati nol.")
    print("=" * 75)


if __name__ == "__main__":
    run_exploratory_data_analysis()

