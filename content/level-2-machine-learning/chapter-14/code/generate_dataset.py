"""
Script pembangkit dataset klasifikasi kerusakan peralatan industri:
equipment_fault_records.csv
Chapter 14: Supervised Learning — Classification Algorithms
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_classification_dataset(csv_path: Path, n_samples: int = 350, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 4 Kelas Status Operasional:
    # 0: NORMAL (~68%)
    # 1: BEARING_FAULT (~15%)
    # 2: OVERHEATING (~11%)
    # 3: ELECTRICAL_SURGE (~6%)
    class_probs = [0.68, 0.15, 0.11, 0.06]
    labels = np.random.choice([0, 1, 2, 3], size=n_samples, p=class_probs)
    
    records = []
    for lbl in labels:
        if lbl == 0:  # NORMAL
            temp = np.random.normal(55.0, 5.0)
            vib = np.random.normal(1.8, 0.4)
            acoustic = np.random.normal(48.0, 4.0)
            voltage_fluc = np.random.normal(1.5, 0.5)
            pressure = np.random.normal(5.0, 0.4)
            current = np.random.normal(12.0, 1.2)
        elif lbl == 1:  # BEARING_FAULT (Getaran tinggi, akustik bising)
            temp = np.random.normal(68.0, 6.0)
            vib = np.random.normal(4.8, 0.7)
            acoustic = np.random.normal(72.0, 6.0)
            voltage_fluc = np.random.normal(2.2, 0.8)
            pressure = np.random.normal(5.2, 0.5)
            current = np.random.normal(14.5, 1.5)
        elif lbl == 2:  # OVERHEATING (Suhu ekstrem panas, tekanan meningkat)
            temp = np.random.normal(88.0, 7.0)
            vib = np.random.normal(2.8, 0.5)
            acoustic = np.random.normal(58.0, 5.0)
            voltage_fluc = np.random.normal(3.0, 1.0)
            pressure = np.random.normal(7.2, 0.8)
            current = np.random.normal(16.0, 1.8)
        else:  # ELECTRICAL_SURGE (Fluktuasi voltase liar, lonjakan arus mendadak)
            temp = np.random.normal(64.0, 8.0)
            vib = np.random.normal(2.4, 0.6)
            acoustic = np.random.normal(65.0, 7.0)
            voltage_fluc = np.random.normal(8.5, 1.8)
            pressure = np.random.normal(5.1, 0.5)
            current = np.random.normal(24.0, 3.5)
            
        fault_binary = 0 if lbl == 0 else 1
        fault_name = ["NORMAL", "BEARING_FAULT", "OVERHEATING", "ELECTRICAL_SURGE"][lbl]
        
        records.append({
            "temperature_c": round(float(temp), 2),
            "vibration_rms": round(float(np.clip(vib, 0.5, 8.0)), 2),
            "acoustic_db": round(float(acoustic), 1),
            "voltage_fluctuation_pct": round(float(np.clip(voltage_fluc, 0.2, 15.0)), 2),
            "pressure_bar": round(float(pressure), 2),
            "current_amp": round(float(current), 2),
            "fault_detected": fault_binary,
            "fault_type_id": lbl,
            "fault_type_name": fault_name
        })
        
    df = pd.DataFrame(records)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    out_csv = Path(__file__).resolve().parent / "data" / "equipment_fault_records.csv"
    df = generate_classification_dataset(out_csv)
    print(f"[OK] Berhasil menghasilkan {len(df)} rekaman klasifikasi di: {out_csv}")
    print(f"Distribusi Biner (fault_detected): 0={sum(df['fault_detected']==0)}, 1={sum(df['fault_detected']==1)} ({df['fault_detected'].mean()*100:.1f}%)")
    print("Distribusi Multiclass:")
    print(df["fault_type_name"].value_counts().to_string())

