"""Pembangkit Dataset Proyek Capstone: Pemeliharaan Prediktif Mesin Industri
Menghasilkan 600 observasi telemetri mesin kompresor industri lintas 12 unit mesin.
"""

import pathlib
import numpy as np
import pandas as pd

def generate_capstone_dataset():
    np.random.seed(42)
    n_machines = 12
    samples_per_machine = 50
    records = []

    for m_idx in range(1, n_machines + 1):
        machine_id = f"PUMP_{m_idx:02d}"
        machine_health_bias = np.random.normal(0.0, 0.4)

        for s_idx in range(1, samples_per_machine + 1):
            # Progresi keausan kumulatif
            wear_factor = (s_idx / samples_per_machine) * 1.8 + machine_health_bias * 0.3
            noise = np.random.normal(0.0, 0.3)

            # Fitur sensor fisik
            vibration = 1.2 + 1.1 * wear_factor + np.random.normal(0.0, 0.2)
            temperature = 55.0 + 22.0 * wear_factor + np.random.normal(0.0, 3.0)
            rpm = 2950.0 - 45.0 * wear_factor + np.random.normal(0.0, 15.0)
            pressure = 6.5 - 1.2 * wear_factor + np.random.normal(0.0, 0.25)
            current = 18.5 + 4.5 * wear_factor + np.random.normal(0.0, 0.8)
            tan_acidity = 0.8 + 1.2 * wear_factor + np.random.normal(0.0, 0.15)

            # Target status kerusakan deterministik dengan komponen noise
            failure_score = (
                20.0 * (vibration - 1.2)
                + 0.7 * (temperature - 55.0)
                + 5.0 * (current - 18.5)
                + 8.0 * (tan_acidity - 0.8)
                - 10.0 * (pressure - 6.5)
                + noise * 6.0
            )

            prob_failure = 1.0 / (1.0 + np.exp(-(failure_score - 65.0) / 8.0))
            is_failure = int(np.random.binomial(1, prob_failure))

            records.append({
                "machine_id": machine_id,
                "sample_seq": s_idx,
                "vibration_rms_g": round(float(vibration), 4),
                "bearing_temperature_c": round(float(temperature), 2),
                "rotational_speed_rpm": round(float(rpm), 1),
                "hydraulic_pressure_bar": round(float(pressure), 3),
                "motor_current_amp": round(float(current), 3),
                "oil_acidity_tan": round(float(tan_acidity), 3),
                "machine_failure": is_failure
            })

    df = pd.DataFrame(records)
    out_dir = pathlib.Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictive_maintenance_telemetry.csv"
    df.to_csv(out_file, index=False)

    print("======================================================================")
    print("PEMBANGKITAN DATASET CAPSTONE CHAPTER 20 SELESAI")
    print("======================================================================")
    print(f"Berkas tersimpan di   : {out_file}")
    print(f"Dimensi data          : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Jumlah Mesin Unik     : {df['machine_id'].nunique()} ({df['machine_id'].unique().tolist()})")
    print(f"Distribusi Kegagalan  : Normal (0) = {(df['machine_failure'] == 0).sum()}, Failure (1) = {(df['machine_failure'] == 1).sum()} ({(df['machine_failure'] == 1).mean()*100:.2f}%)")
    print("======================================================================")

if __name__ == "__main__":
    generate_capstone_dataset()
