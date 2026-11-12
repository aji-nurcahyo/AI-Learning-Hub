"""Matplotlib Visualizations: Figure, Axes & Multi-Plot Dashboards.

Chapter 07: Python Ecosystem for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan:
1. Paradigma Object-Oriented Matplotlib (Figure & Axes)
2. Backend Non-Interaktif ('Agg') untuk Eksekusi Headless Terminal
3. Pembuatan 4 Jenis Plot Fundamental:
   - Line Plot: Tren runtut waktu (Time-Series Trend)
   - Scatter Plot: Korelasi antar fitur numerik (Suhu vs Getaran)
   - Histogram: Distribusi frekuensi variabel kontinu
   - Bar Chart: Perbandingan metrik rata-rata antar kategori
4. Penataan Layout, Label, Legend, dan Penyimpanan Berkas Gambar (PNG)
"""

import sys
from pathlib import Path
import matplotlib

# Gunakan backend non-GUI 'Agg' agar aman dieksekusi di server/terminal tanpa display server
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def run_matplotlib_demo():
    print("=" * 70)
    print("DEMO 5: MATPLOTLIB FUNDAMENTALS — 4 CHART TYPES & EXPORT")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    clean_csv = base_dir / "data" / "sensor_readings_clean.csv"
    if not clean_csv.exists():
        # Fallback baca raw jika clean belum ada
        clean_csv = base_dir / "data" / "sensor_readings.csv"

    df = pd.read_csv(clean_csv)
    # Isi missing values jika masih ada
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].mean())
    df["vibration_mms"] = df["vibration_mms"].fillna(df["vibration_mms"].mean())

    output_fig = base_dir / "data" / "sensor_analysis_plot.png"

    # Membuat Kanvas Dashboard 2x2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle("Analisis Visual Data Sensor Industri — Chapter 07 AI Hub", fontsize=15, fontweight="bold")

    # 1. Line Plot (Tren Pengamatan Berurutan)
    ax1 = axes[0, 0]
    ax1.plot(df.index, df["temperature_c"], color="#1976D2", marker="o", linewidth=2, label="Suhu (°C)")
    ax1.set_title("1. Tren Suhu Pengamatan", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Indeks Waktu Sampel")
    ax1.set_ylabel("Suhu (°C)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Scatter Plot (Korelasi Suhu vs Getaran dengan Pembeda Status)
    ax2 = axes[0, 1]
    colors = df["status"].map({"NORMAL": "#388E3C", "WARNING": "#F57C00", "CRITICAL": "#D32F2F"}).fillna("#757575")
    scatter = ax2.scatter(df["temperature_c"], df["vibration_mms"], c=colors, s=80, edgecolors="black", alpha=0.8)
    ax2.set_title("2. Korelasi Suhu vs Getaran", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Suhu (°C)")
    ax2.set_ylabel("Getaran (mm/s)")
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Histogram (Distribusi Frekuensi Suhu)
    ax3 = axes[1, 0]
    ax3.hist(df["temperature_c"], bins=6, color="#0288D1", edgecolor="black", alpha=0.7)
    mean_val = df["temperature_c"].mean()
    ax3.axvline(mean_val, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_val:.1f}°C")
    ax3.set_title("3. Distribusi Frekuensi Suhu", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Rentang Suhu (°C)")
    ax3.set_ylabel("Frekuensi Pengamatan")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # 4. Bar Chart (Rata-rata Getaran per Lokasi)
    ax4 = axes[1, 1]
    loc_vibe = df.groupby("location")["vibration_mms"].mean().sort_values()
    bar_colors = ["#81C784", "#AED581", "#FFB74D", "#E57373"]
    bars = ax4.barh(loc_vibe.index, loc_vibe.values, color=bar_colors[:len(loc_vibe)], edgecolor="black")
    ax4.set_title("4. Rata-rata Getaran per Lokasi", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Rata-rata Getaran (mm/s)")
    ax4.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Menambahkan label nilai pada tiap bar
    for bar in bars:
        width = bar.get_width()
        ax4.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va="center", fontsize=10)

    # Menyimpan Dashboard ke Disk
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualisasi 4 plot berhasil digenerasi dan disimpan ke:")
    print(f"  {output_fig.name} ({output_fig.stat().st_size} bytes)")
    print("Dashboard visual berhasil dirender secara headless tanpa kendala display GUI!")
    print("=" * 70)


if __name__ == "__main__":
    run_matplotlib_demo()

