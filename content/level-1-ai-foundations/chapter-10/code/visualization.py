"""
Smart Sensor Data Quality Monitor - Visualization Engine
Chapter 10: AI Foundations Capstone Reference Implementation

Merender dashboard visualisasi kualitas data sensor secara headless (tanpa GUI) menggunakan Matplotlib.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def render_quality_dashboard(
    df_validated: pd.DataFrame,
    stats_summary: Dict[str, Any],
    dqi_scores: Dict[str, float],
    output_image_path: Path
) -> None:
    """Menghasilkan dashboard 3-panel kualitas data sensor."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # -------------------------------------------------------------
    # Panel 1: Runtutan Waktu Sensor Suhu (TEMP-01) & Status Validasi
    # -------------------------------------------------------------
    ax1 = axes[0]
    temp_data = df_validated[df_validated["sensor_id"] == "TEMP-01"].copy()
    temp_data["time_idx"] = range(len(temp_data))
    
    valid_pts = temp_data[temp_data["validation_status"] == "VALID"]
    invalid_pts = temp_data[temp_data["validation_status"] == "INVALID_BOUNDS"]
    missing_pts = temp_data[temp_data["validation_status"] == "MISSING"]
    
    ax1.plot(temp_data["time_idx"], temp_data["measurement_value"], color="#1976D2", alpha=0.6, label="Tren Suhu (degC)")
    ax1.scatter(valid_pts["time_idx"], valid_pts["measurement_value"], color="#388E3C", s=25, label="Data Valid")
    if not invalid_pts.empty:
        ax1.scatter(invalid_pts["time_idx"], invalid_pts["measurement_value"], color="#D32F2F", s=60, marker="x", label="Batas Fisik Tidak Valid")
    if not missing_pts.empty:
        ax1.scatter(missing_pts["time_idx"], [28.0]*len(missing_pts), color="#FFA000", s=40, marker="o", label="Data Hilang (Imputasi Indikator)")
        
    ax1.set_title("1. Inspeksi Telemetri Suhu (TEMP-01)")
    ax1.set_xlabel("Indeks Waktu Pengamatan")
    ax1.set_ylabel("Nilai Pengukuran (degC)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)
    
    # -------------------------------------------------------------
    # Panel 2: Boxplot Distribusi Normal Valid per Sensor
    # -------------------------------------------------------------
    ax2 = axes[1]
    box_data = []
    labels = []
    for sid in sorted(stats_summary.keys()):
        valid_vals = df_validated[(df_validated["sensor_id"] == sid) & (df_validated["validation_status"] == "VALID")]["measurement_value"].dropna()
        if not valid_vals.empty:
            box_data.append(valid_vals)
            labels.append(f"{sid}\n({stats_summary[sid]['unit']})")
            
    if box_data:
        ax2.boxplot(box_data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#E3F2FD", color="#1976D2"))
    ax2.set_title("2. Sebaran Nilai Valid (Boxplot Tukey)")
    ax2.set_ylabel("Nilai Numerik")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # -------------------------------------------------------------
    # Panel 3: Data Quality Index (DQI) per Sensor
    # -------------------------------------------------------------
    ax3 = axes[2]
    sids = list(dqi_scores.keys())
    scores = [dqi_scores[s] for s in sids]
    colors = ["#4CAF50" if sc >= 85 else "#FF9800" if sc >= 70 else "#F44336" for sc in scores]
    
    bars = ax3.bar(sids, scores, color=colors, width=0.5)
    ax3.set_ylim(0, 110)
    ax3.axhline(85, color="#388E3C", linestyle="--", label="Target Kualitas Tinggi (>= 85)")
    ax3.set_title("3. Data Quality Index (DQI)")
    ax3.set_ylabel("Skor Kualitas (0 - 100)")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.legend(loc="lower right", fontsize=8)
    
    plt.tight_layout()
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=150)
    plt.close(fig)
