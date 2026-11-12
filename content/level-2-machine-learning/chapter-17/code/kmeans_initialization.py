"""Analisis Inisialisasi K-Means: Random Initialization vs K-Means++
Mendemonstrasikan risiko optimum lokal pada inisialisasi acak
dan keunggulan stabilitas inisialisasi probabilistik K-Means++.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_initialization_comparison():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = StandardScaler().fit_transform(df[features])

    print("=== PERBANDINGAN INISIALISASI: RANDOM VS K-MEANS++ ===")
    print("Menjalankan 20 kali percobaan independen (n_init = 1) untuk menguji stabilitas:\n")

    n_runs = 20
    inertias_random = []
    iters_random = []
    inertias_pp = []
    iters_pp = []

    for seed in range(n_runs):
        # 1. Random Init
        km_rand = KMeans(n_clusters=3, init="random", n_init=1, max_iter=100, random_state=seed)
        km_rand.fit(X)
        inertias_random.append(km_rand.inertia_)
        iters_random.append(km_rand.n_iter_)

        # 2. K-Means++ Init
        km_pp = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=100, random_state=seed)
        km_pp.fit(X)
        inertias_pp.append(km_pp.inertia_)
        iters_pp.append(km_pp.n_iter_)

    print("Metrik Evaluasi               | Inisialisasi Random     | Inisialisasi K-Means++")
    print("-" * 65)
    print(f"Rata-rata Inertia (WCSS)      | {np.mean(inertias_random):10.2f}            | {np.mean(inertias_pp):10.2f}")
    print(f"Standar Deviasi Inertia       | {np.std(inertias_random):10.2f}            | {np.std(inertias_pp):10.2f} (Lebih Stabil)")
    print(f"Inertia Terburuk (Opt. Lokal) | {np.max(inertias_random):10.2f}            | {np.max(inertias_pp):10.2f}")
    print(f"Inertia Terbaik (Optimum)     | {np.min(inertias_random):10.2f}            | {np.min(inertias_pp):10.2f}")
    print(f"Rata-rata Jumlah Iterasi      | {np.mean(iters_random):10.1f} iterasi      | {np.mean(iters_pp):10.1f} iterasi")
    print("-" * 65)

    print("\nPrinsip Aljabar K-Means++ (Arthur & Vassilvitskii, 2007):")
    print("1. Centroid pertama dipilih secara acak merata dari dataset.")
    print("2. Setiap centroid berikutnya dipilih dengan peluang proporsional terhadap")
    print("   kuadrat jarak terpendek ke centroid yang sudah ada: P(x) = D(x)^2 / sum(D(x)^2).")
    print("3. Menjamin centroid awal tersebar saling berjauhan, secara drastis menekan optimum lokal.")

if __name__ == "__main__":
    run_initialization_comparison()

