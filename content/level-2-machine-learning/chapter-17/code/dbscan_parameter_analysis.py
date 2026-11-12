"""Analisis Sensitivitas Hyperparameter DBSCAN (eps dan min_samples)
Mengevaluasi pengaruh radius tetangga (eps) dan ambang batas titik inti (min_samples)
terhadap jumlah klaster dan proporsi titik derau (noise).
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def run_parameter_grid():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = StandardScaler().fit_transform(df[features])

    print("=== EKSPLORASI GRID PARAMETER DBSCAN ===")
    print("Mengevaluasi interaksi eps in [0.75, 1.75] dan min_samples in [4, 10]:\n")
    print(f"{'Nilai eps':<10} | {'min_samples':<12} | {'Jumlah Klaster':<15} | {'Titik Noise':<12} | {'Persentase Noise':<16}")
    print("-" * 75)

    eps_candidates = [0.8, 1.0, 1.2, 1.5, 1.8]
    min_samples_candidates = [4, 6, 8]

    for eps in eps_candidates:
        for ms in min_samples_candidates:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            pct_noise = (n_noise / len(X)) * 100

            print(f"eps={eps:<5.1f} | ms={ms:<8} | {n_clusters:<15} | {n_noise:<12} | {pct_noise:5.1f}%")

    print("-" * 75)
    print("Prinsip Pemilihan Hyperparameter:")
    print("1. eps terlalu kecil : Kepadatan sulit terpenuhi, mayoritas data dianggap noise (Under-clustering).")
    print("2. eps terlalu besar : Klaster terpisah menyatu menjadi 1 super-klaster raksasa (Over-merging).")
    print("3. min_samples      : Umumnya dipilih >= D + 1 (di mana D adalah dimensi fitur, di sini D=6 -> min_samples >= 7).")

if __name__ == "__main__":
    run_parameter_grid()

