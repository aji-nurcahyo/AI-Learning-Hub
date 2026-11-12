"""
Smart Sensor Data Quality Monitor - Statistical Analyzer
Chapter 10: AI Foundations Capstone Reference Implementation

Menghitung statistik deskriptif dan pagar Tukey (Tukey's Fences) untuk mendeteksi outlier
pada data yang telah lolos validasi batas fisik.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_sensor_statistics(df_valid: pd.DataFrame) -> Dict[str, Any]:
    """Menghitung ringkasan statistik deskriptif dan ambang batas Tukey IQR per sensor."""
    stats_summary = {}
    
    # Filter hanya data berstatus VALID untuk analisis statistik distribusi normal
    clean_data = df_valid[df_valid["validation_status"] == "VALID"].copy()
    
    for sid, group in clean_data.groupby("sensor_id"):
        vals = group["measurement_value"].dropna().to_numpy()
        if len(vals) == 0:
            continue
            
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        iqr = float(q3 - q1)
        lower_fence = float(q1 - 1.5 * iqr)
        upper_fence = float(q3 + 1.5 * iqr)
        
        # Deteksi data di luar pagar Tukey
        outlier_indices = group[
            (group["measurement_value"] < lower_fence) | (group["measurement_value"] > upper_fence)
        ].index.tolist()
        
        stats_summary[sid] = {
            "sensor_id": sid,
            "unit": group["unit"].iloc[0],
            "count": int(len(vals)),
            "mean": round(float(np.mean(vals)), 2),
            "median": round(float(np.median(vals)), 2),
            "std": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0,
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_fence": round(lower_fence, 2),
            "upper_fence": round(upper_fence, 2),
            "outlier_count": len(outlier_indices),
            "outlier_indices": outlier_indices
        }
        
    return stats_summary

