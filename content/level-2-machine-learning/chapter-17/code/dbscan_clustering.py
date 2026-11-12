"""Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
Mendemonstrasikan pembentukan klaster berbasis kepadatan, deteksi noise/outliers,
dan identifikasi titik Core vs Border vs Noise.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def run_dbscan():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = StandardScaler().fit_transform(df[features])

    print("=== DENSITY-BASED CLUSTERING (DBSCAN) ===")
    print("Parameter Konfigurasi: eps = 1.15, min_samples = 6\n")

    db = DBSCAN(eps=1.15, min_samples=6)
    labels = db.fit_predict(X)

    # Identifikasi titik Core, Border, dan Noise
    core_mask = np.zeros_like(labels, dtype=bool)
    core_mask[db.core_sample_indices_] = True

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    n_core = len(db.core_sample_indices_)
    n_border = len(labels) - n_core - n_noise

    print("Hasil Pembentukan Klaster DBSCAN:")
    print(f"  - Jumlah Klaster Terbentuk : {n_clusters} klaster")
    print(f"  - Total Core Points        : {n_core} titik ({n_core/len(X)*100:.1f}%)")
    print(f"  - Total Border Points      : {n_border} titik ({n_border/len(X)*100:.1f}%)")
    print(f"  - Total Noise / Outliers   : {n_noise} titik ({n_noise/len(X)*100:.1f}% data teridentifikasi derau/anomali)")

    print("\nDistribusi Sampel Tiap Label:")
    unique_labels = sorted(list(set(labels)))
    for lbl in unique_labels:
        count = np.sum(labels == lbl)
        name = f"Klaster {lbl}" if lbl != -1 else "NOISE / ANOMALI (Label -1)"
        print(f"  - {name:<26}: {count} sampel")

    print("\nKeunggulan Kunci DBSCAN vs K-Means:")
    print("1. Tidak memerlukan spesifikasi jumlah klaster k di awal.")
    print("2. Mampu menemukan bentuk klaster sembarang (non-spherical / arbitrary shape).")
    print("3. Memiliki penanganan eksplisit terhadap data pencilan/derau (outliers tidak dipaksa masuk klaster).")

if __name__ == "__main__":
    run_dbscan()

