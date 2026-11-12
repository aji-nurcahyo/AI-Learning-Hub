"""Visualisasi Manifold Non-Linier Menggunakan t-SNE (t-Distributed Stochastic Neighbor Embedding)
Mengeksplorasi pengaruh hyperparameter perplexity dan preservasi struktur lingkungan lokal.
"""

import pathlib
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def run_tsne():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = [
        "signal_strength_dbm",
        "packet_loss_pct",
        "latency_ms",
        "temperature_c",
        "voltage_v",
        "energy_consumption_mah"
    ]
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    perplexities = [5, 15, 30, 50]

    print("======================================================================")
    print("ANALISIS REDUKSI DIMENSI NON-LINIER: t-SNE (MANIFOLD LEARNING)")
    print("======================================================================")
    print(f"{'Perplexity':<14} {'KL Divergence':<16} {'Waktu Eksekusi (ms)':<22} {'Jumlah Iterasi'}")
    print("----------------------------------------------------------------------")

    for perp in perplexities:
        t0 = time.perf_counter()
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            max_iter=1000,
            learning_rate="auto",
            init="pca",
            random_state=42
        )
        Z_tsne = tsne.fit_transform(X_scaled)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"{perp:<14} {tsne.kl_divergence_:<16.4f} {elapsed:<22.2f} {tsne.n_iter_}")

    print("======================================================================")
    print("INSIGHT & KARAKTERISTIK T-SNE:")
    print("1. Distribusi Peluang: Menggunakan distribusi Gaussian pada ruang asal (tinggi)")
    print("   dan distribusi t-Student 1 derajat kebebasan pada ruang proyeksi (2D)")
    print("   untuk mengatasi Crowding Problem (masalah penumpukan titik).")
    print("2. Preservasi: Sangat unggul menjaga struktur lokal (tetangga dekat),")
    print("   namun jarak global antar-klaster tidak memiliki makna fisik terukur.")
    print("3. Batasan Teknis: Tidak memiliki fungsi pemetaan inversi (.inverse_transform())")
    print("   dan tidak dapat mentransformasi data sampel baru secara langsung (.transform()).")

if __name__ == "__main__":
    run_tsne()
