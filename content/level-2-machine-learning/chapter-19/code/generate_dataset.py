"""Pembangkit Dataset Evaluasi & Explainability Model Telemetri Armada Sensor
Menghasilkan 500 observasi lintas 10 perangkat dan 50 siklus waktu.
"""

import pathlib
import numpy as np
import pandas as pd

def generate_evaluation_dataset():
    np.random.seed(42)
    n_devices = 10
    cycles_per_device = 50
    records = []

    for dev_idx in range(1, n_devices + 1):
        dev_id = f"DEV_{dev_idx:02d}"
        # Efek acak individual tiap perangkat (device random effect)
        dev_bias = np.random.normal(0.0, 0.5)

        for cycle in range(1, cycles_per_device + 1):
            # Tren degradasi kumulatif seiring berjalannya siklus waktu
            wear = (cycle / cycles_per_device) * 1.5 + dev_bias * 0.3
            noise = np.random.normal(0.0, 0.4)

            vibration = 0.5 + 0.8 * wear + np.random.normal(0.0, 0.15)
            temperature = 45.0 + 18.0 * wear + dev_bias * 2.0 + np.random.normal(0.0, 2.5)
            pressure = 101.3 - 4.5 * wear + np.random.normal(0.0, 1.2)
            power_draw = 3.2 + 1.2 * wear + np.random.normal(0.0, 0.25)
            acoustic = 55.0 + 12.0 * wear + np.random.normal(0.0, 3.0)
            operating_hours = (cycle - 1) * 24 + np.random.randint(10, 24)

            # Target Regresi: Skor Risiko Kerusakan (0 s.d. 100)
            raw_risk = (
                15.0
                + 25.0 * (vibration - 0.5)
                + 0.8 * (temperature - 45.0)
                + 12.0 * (power_draw - 3.2)
                + 0.5 * (acoustic - 55.0)
                + noise * 5.0
            )
            risk_score = float(np.clip(raw_risk, 0.0, 100.0))

            # Target Klasifikasi Biner: Status Kerusakan (0 = Normal, 1 = Fault)
            prob_fault = 1.0 / (1.0 + np.exp(-(risk_score - 52.0) / 8.0))
            is_fault = int(np.random.binomial(1, prob_fault))

            # Target Klasifikasi Multikelas
            if not is_fault:
                category = "Normal"
            elif temperature > 68.0:
                category = "Overheating_Fault"
            else:
                category = "Mechanical_Degradation"

            records.append({
                "device_id": dev_id,
                "cycle_id": cycle,
                "operating_hours": operating_hours,
                "vibration_amplitude_g": round(vibration, 4),
                "operating_temp_c": round(temperature, 2),
                "pressure_kpa": round(pressure, 2),
                "power_draw_kw": round(power_draw, 4),
                "acoustic_noise_db": round(acoustic, 2),
                "failure_risk_score": round(risk_score, 2),
                "failure_status": is_fault,
                "failure_category": category
            })

    df = pd.DataFrame(records)
    out_dir = pathlib.Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sensor_fleet_eval_telemetry.csv"
    df.to_csv(out_file, index=False)

    print("======================================================================")
    print("PEMBANGKITAN DATASET EVALUASI MODEL CHAPTER 19 SELESAI")
    print("======================================================================")
    print(f"Berkas tersimpan di   : {out_file}")
    print(f"Dimensi data          : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Jumlah Perangkat Unik : {df['device_id'].nunique()} ({df['device_id'].unique().tolist()})")
    print(f"Distribusi Status     : 0 (Normal) = {(df['failure_status'] == 0).sum()}, 1 (Fault) = {(df['failure_status'] == 1).sum()} ({(df['failure_status'] == 1).mean()*100:.1f}%)")
    print(f"Distribusi Kategori   :\n{df['failure_category'].value_counts().to_string()}")
    print("======================================================================")

if __name__ == "__main__":
    generate_evaluation_dataset()

