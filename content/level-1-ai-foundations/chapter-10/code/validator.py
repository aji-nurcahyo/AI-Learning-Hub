"""
Smart Sensor Data Quality Monitor - Schema & Data Validator
Chapter 10: AI Foundations Capstone Reference Implementation

Memvalidasi skema data, mendeteksi missing values, dan memeriksa batas fisika pengukuran
tanpa langsung membuang data (mempertahankan jejak audit).
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


PHYSICAL_BOUNDS = {
    "TEMP-01": {"min": -10.0, "max": 100.0, "expected_unit": "degC"},
    "HUMID-01": {"min": 0.0, "max": 100.0, "expected_unit": "%RH"},
    "VIB-01": {"min": 0.0, "max": 25.0, "expected_unit": "mm/s"}
}


def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """Memverifikasi bahwa DataFrame memuat seluruh kolom wajib."""
    required_cols = {"timestamp", "sensor_id", "measurement_value", "unit"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Kolom wajib tidak ditemukan: {missing}"
    return True, "Skema valid"


def validate_sensor_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Memeriksa setiap rekaman: integritas tipe data, kelengkapan, dan batas fisika."""
    report = {
        "total_records": len(df),
        "missing_count": int(df["measurement_value"].isna().sum()),
        "invalid_bounds_count": 0,
        "valid_records_count": 0,
        "invalid_details": []
    }
    
    validated_df = df.copy()
    validated_df["validation_status"] = "VALID"
    validated_df["status_reason"] = "OK"
    
    for idx, row in validated_df.iterrows():
        sid = row["sensor_id"]
        val = row["measurement_value"]
        
        # 1. Missing Value Check
        if pd.isna(val):
            validated_df.at[idx, "validation_status"] = "MISSING"
            validated_df.at[idx, "status_reason"] = "Nilai pengukuran kosong / hilang (NaN)"
            continue
            
        # 2. Unknown Sensor Check
        if sid not in PHYSICAL_BOUNDS:
            validated_df.at[idx, "validation_status"] = "UNKNOWN_SENSOR"
            validated_df.at[idx, "status_reason"] = f"Sensor ID {sid} tidak terdaftar pada konfigurasi sistem"
            continue
            
        bounds = PHYSICAL_BOUNDS[sid]
        # 3. Physical Boundary Check
        if val < bounds["min"] or val > bounds["max"]:
            validated_df.at[idx, "validation_status"] = "INVALID_BOUNDS"
            validated_df.at[idx, "status_reason"] = (
                f"Nilai {val} di luar batas fisika yang dimungkinkan [{bounds['min']}, {bounds['max']}]"
            )
            report["invalid_bounds_count"] += 1
            report["invalid_details"].append({
                "index": int(idx),
                "timestamp": str(row["timestamp"]),
                "sensor_id": sid,
                "value": float(val),
                "reason": validated_df.at[idx, "status_reason"]
            })
            
    report["valid_records_count"] = int((validated_df["validation_status"] == "VALID").sum())
    return validated_df, report

