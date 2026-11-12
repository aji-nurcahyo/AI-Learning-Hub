"""Pembangkit Dataset Telemetri Sensor Kerusakan Mesin (Deterministik)
Untuk Chapter 15: Decision Trees & Ensemble Learning
"""

import pathlib
import numpy as np
import pandas as pd

def generate_sensor_data(output_path: pathlib.Path, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    n_samples = 360

    # Pembangkitan fitur telemetri peralatan
    # 1. Operating hours: 100 - 8000 jam
    operating_hours = np.random.uniform(100, 8000, n_samples)

    # 2. Load percentage: 20% - 100%
    load_pct = np.random.uniform(20, 100, n_samples)

    # 3. Base characteristics dependent on load
    temp_base = 45.0 + 0.35 * load_pct + 0.002 * operating_hours + np.random.normal(0, 3.0, n_samples)
    vibration_base = 1.0 + 0.025 * load_pct + 0.0003 * operating_hours + np.random.normal(0, 0.4, n_samples)
    voltage_base = np.random.normal(380.0, 5.0, n_samples)
    current_base = 15.0 + 0.30 * load_pct + np.random.normal(0, 2.0, n_samples)

    # Tentukan anomali / kerusakan berdasarkan ambang batas fisik sistem industri
    # Kelas: 0 = NORMAL, 1 = WARNING, 2 = FAULT
    targets = []
    temps = temp_base.copy()
    vibs = vibration_base.copy()
    volts = voltage_base.copy()
    currs = current_base.copy()

    for i in range(n_samples):
        # Induksi kondisi abnormal deterministik
        is_fault = (temps[i] > 84.0) or (vibs[i] > 5.2) or (volts[i] < 360.0) or (volts[i] > 405.0) or (currs[i] > 48.0)
        is_warning = (temps[i] > 74.0) or (vibs[i] > 3.6) or (volts[i] < 370.0) or (volts[i] > 395.0) or (currs[i] > 40.0)

        # Modifikasi beberapa sampel agar terjadi distribusi realistis
        if i % 7 == 0:  # Injeksi degradasi vibrasi (bearing fault)
            vibs[i] += np.random.uniform(2.5, 4.0)
            is_fault = True
        elif i % 11 == 0:  # Injeksi lonjakan termal (cooling failure)
            temps[i] += np.random.uniform(12.0, 22.0)
            is_fault = True
        elif i % 13 == 0:  # Injeksi fluktuasi voltase (power sag)
            volts[i] -= np.random.uniform(18.0, 30.0)
            is_fault = True

        if is_fault:
            targets.append("FAULT")
        elif is_warning:
            targets.append("WARNING")
        else:
            targets.append("NORMAL")

    df = pd.DataFrame({
        "temperature_c": np.round(temps, 2),
        "vibration_mm_s": np.round(np.clip(vibs, 0.5, 12.0), 3),
        "voltage_v": np.round(volts, 2),
        "current_a": np.round(currs, 2),
        "operating_hours": np.round(operating_hours, 1),
        "load_percentage": np.round(load_pct, 1),
        "fault_category": targets
    })

    # Encode label numerik
    mapping = {"NORMAL": 0, "WARNING": 1, "FAULT": 2}
    df["fault_code"] = df["fault_category"].map(mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "data" / "equipment_fault_sensor_data.csv"
    data = generate_sensor_data(out)
    print(f"Dataset generated at: {out}")
    print(f"Total samples: {len(data)}")
    print(f"Distribution:\n{data['fault_category'].value_counts()}")

