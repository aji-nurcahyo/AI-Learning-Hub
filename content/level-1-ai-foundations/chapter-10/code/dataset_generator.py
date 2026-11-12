"""
Smart Sensor Data Quality Monitor - Dataset Generator
Chapter 10: AI Foundations Capstone Reference Implementation

Menghasilkan dataset pengukuran sensor sintetis yang deterministik (menggunakan random seed tetap)
untuk keperluan pengujian Data Quality Monitor.
Fitur data:
- timestamp: runtutan waktu pengamatan
- sensor_id: identitas unik sensor (TEMP-01, HUMID-01, VIB-01)
- measurement_value: nilai pengukuran fisik
- unit: satuan standar (degC, %RH, mm/s)
- Anomali terkontrol: missing values (NaN), invalid values (misal < -273 C atau string anomali), dan statistical outliers.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_sensor_data(output_path: Path, n_records: int = 120, seed: int = 42) -> pd.DataFrame:
    """Menghasilkan dataset sensor sintetis dengan anomali terkontrol."""
    np.random.seed(seed)
    base_time = pd.Timestamp("2026-11-05 08:00:00")
    
    sensor_types = [
        {"id": "TEMP-01", "unit": "degC", "mean": 28.0, "std": 1.5, "min_valid": -10.0, "max_valid": 100.0},
        {"id": "HUMID-01", "unit": "%RH", "mean": 65.0, "std": 4.0, "min_valid": 0.0, "max_valid": 100.0},
        {"id": "VIB-01", "unit": "mm/s", "mean": 2.2, "std": 0.4, "min_valid": 0.0, "max_valid": 25.0}
    ]
    
    records = []
    
    for i in range(n_records):
        st = sensor_types[i % len(sensor_types)]
        timestamp = base_time + pd.Timedelta(minutes=5 * (i // len(sensor_types)))
        
        # Nilai dasar Gaussian
        val = float(np.random.normal(loc=st["mean"], scale=st["std"]))
        
        # Sematan anomali terkontrol pada indeks tertentu:
        # 1. Missing values (~5% kasus)
        if i in [12, 45, 88]:
            val = np.nan
        # 2. Nilai fisik tidak valid / impossible measurement
        elif i == 24: # Suhu di bawah nol mutlak
            val = -300.0
        elif i == 66: # Kelembapan di atas 100%
            val = 145.0
        # 3. Statistical outliers (lonjakan sesaat / spike)
        elif i == 51: # Spike vibrasi mesin turbin
            val = 18.5
        elif i == 93: # Spike panas anomali
            val = 62.0
            
        records.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "sensor_id": st["id"],
            "measurement_value": round(val, 2) if not np.isnan(val) else np.nan,
            "unit": st["unit"]
        })
        
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data" / "sensor_telemetry_raw.csv"
    df = generate_synthetic_sensor_data(out)
    print(f"[OK] Berhasil menghasilkan {len(df)} rekaman sensor sintetis di: {out}")
    print(f"     Missing values: {df['measurement_value'].isna().sum()} sel")

