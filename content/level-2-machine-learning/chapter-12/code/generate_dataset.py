"""
Script untuk membangkitkan dataset telemetri peralatan industri sintetis (equipment_telemetry.csv)
untuk keperluan pengujian dan demonstrasi Chapter 12: ML Workflow & Feature Engineering.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_equipment_dataset(csv_path: Path, n_samples: int = 250, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    base_time = pd.Timestamp("2026-11-01 00:00:00")
    
    machines = ["MCH-Alpha", "MCH-Beta", "MCH-Gamma", "MCH-Delta", "MCH-Epsilon"]
    cooling_modes = ["ACTIVE", "STANDBY", "OFF"]
    warning_levels = ["LOW", "MEDIUM", "HIGH"]
    
    records = []
    for i in range(n_samples):
        # Time-series timestamps: interval 1 jam
        ts = base_time + pd.Timedelta(hours=i)
        m_id = machines[i % len(machines)]
        
        op_hours = float(np.random.uniform(100.0, 5000.0))
        vib_rms = float(np.random.gamma(shape=2.0, scale=1.2))
        temp_c = float(np.random.normal(loc=65.0, scale=12.0))
        pressure_kpa = float(np.random.normal(loc=300.0, scale=35.0))
        
        cooling = cooling_modes[int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))]
        warning = warning_levels[int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))]
        
        # Logika pembentukan label kegagalan (Failure: imbalanced ~12%)
        # Kegagalan terjadi jika kombinasi suhu ekstrem + vibrasi tinggi + pendingin OFF
        risk_score = (
            (vib_rms > 3.0) * 1.5 +
            (temp_c > 75.0) * 1.8 +
            (pressure_kpa > 330.0) * 1.2 +
            (cooling == "OFF") * 1.5 +
            (warning == "HIGH") * 1.5 +
            np.random.normal(0, 0.5)
        )
        failure = 1 if risk_score > 3.0 else 0
        
        # Missing values terkontrol pada beberapa baris (~3%)
        if i in [15, 78, 142]:
            temp_c = np.nan
        if i in [33, 110, 205]:
            vibration_rms = np.nan
        else:
            vibration_rms = vib_rms
            
        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "machine_id": m_id,
            "operating_hours": round(op_hours, 1),
            "vibration_rms": round(vibration_rms, 2) if not np.isnan(vibration_rms) else np.nan,
            "temperature_c": round(temp_c, 2) if not np.isnan(temp_c) else np.nan,
            "pressure_kpa": round(pressure_kpa, 1),
            "cooling_status": cooling,
            "warning_flag": warning,
            "failure": failure
        })
        
    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    out_csv = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    df = generate_equipment_dataset(out_csv)
    print(f"[OK] Berhasil menghasilkan {len(df)} sampel telemetri di: {out_csv}")
    print(f"     Distribusi Target Failure: 0={sum(df['failure']==0)}, 1={sum(df['failure']==1)} ({sum(df['failure']==1)/len(df)*100:.1f}%)")
    print(f"     Missing values: Temp={df['temperature_c'].isna().sum()}, Vib={df['vibration_rms'].isna().sum()}")
