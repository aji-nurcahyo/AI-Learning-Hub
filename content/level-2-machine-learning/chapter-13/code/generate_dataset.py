"""
Script pembangkit dataset sintetis konsumsi daya peralatan industri:
equipment_power_demand.csv
Chapter 13: Supervised Learning — Regression Algorithms
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_energy_dataset(csv_path: Path, n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. Fitur Operasional dan Lingkungan
    ambient_temp = np.random.normal(loc=28.0, scale=6.0, size=n_samples) # 15 - 45 °C
    ambient_temp = np.clip(ambient_temp, 16.0, 44.0)
    
    operating_load = np.random.uniform(20.0, 95.0, size=n_samples)       # 20 - 95 %
    machine_speed = 800.0 + (operating_load * 12.5) + np.random.normal(0, 25.0, size=n_samples) # RPM
    inlet_pressure = 4.0 + (operating_load * 0.05) + np.random.normal(0, 0.3, size=n_samples)   # 4 - 9 bar
    vibration = 1.0 + (operating_load * 0.03) + np.random.normal(0, 0.4, size=n_samples)       # 1 - 5 mm/s
    vibration = np.clip(vibration, 0.5, 7.0)
    equipment_age = np.random.randint(3, 72, size=n_samples)             # 3 - 72 bulan
    
    # 2. Formula Fisika Konsumsi Daya (kW):
    # Base power: 45 kW
    # Pengaruh linier dominan: beban kerja (load) dan kecepatan mesin (speed)
    # Pengaruh non-linear kuadratik: beban kerja kuadrat (kehilangan efisiensi termal di beban tinggi)
    # Pengaruh lingkungan: suhu ambien tinggi menambah beban pendinginan motor
    # Pengaruh keausan: usia mesin dan vibrasi
    power_kw = (
        45.0 +
        (operating_load * 1.85) +
        (machine_speed * 0.045) +
        (0.008 * (operating_load ** 2)) +
        (ambient_temp * 0.65) +
        (inlet_pressure * 3.2) +
        (vibration * 4.1) +
        (equipment_age * 0.15) +
        np.random.normal(0, 4.5, size=n_samples) # Derau instrumen pengukuran riil
    )
    
    df = pd.DataFrame({
        "ambient_temp_c": np.round(ambient_temp, 2),
        "operating_load_pct": np.round(operating_load, 2),
        "machine_speed_rpm": np.round(machine_speed, 1),
        "inlet_pressure_bar": np.round(inlet_pressure, 2),
        "vibration_severity_mms": np.round(vibration, 2),
        "equipment_age_months": equipment_age,
        "power_consumption_kw": np.round(power_kw, 2)
    })
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    out_csv = Path(__file__).resolve().parent / "data" / "equipment_power_demand.csv"
    df = generate_energy_dataset(out_csv)
    print(f"[OK] Berhasil menghasilkan {len(df)} sampel telemetri daya di: {out_csv}")
    print(df.describe().round(2).to_string())

