"""Eksplorasi Algoritma UMAP (Uniform Manifold Approximation and Projection)
Penanganan ketergantungan opsional umap-learn secara elegan dan tangguh.
"""

import pathlib
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_umap_demo():
    print("======================================================================")
    print("ANALISIS REDUKSI DIMENSI MANIFOLD: UMAP (UNIFORM MANIFOLD APPROXIMATION)")
    print("======================================================================")

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

    try:
        import umap
        print("[INFO] Pustaka umap-learn terdeteksi. Menjalankan reduksi dimensi...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        print(f"Bentuk Embedding UMAP 2D: {embedding.shape}")
        print("Sampel 3 baris koordinat proyeksi UMAP:")
        print(embedding[:3])
    except ImportError:
        print("[STATUS KETERSEDIAAN MODUL]: Pustaka 'umap-learn' belum terpasang.")
        print("Lingkungan virtual saat ini beroperasi pada Scikit-Learn murni.")
        print("----------------------------------------------------------------------")
        print("PANDUAN PEMASANGAN OPSIONAL:")
        print("  pip install umap-learn")
        print("----------------------------------------------------------------------")
        print("PENJELASAN TEORETIS FUNDAMENTAL UMAP (McInnes et al., 2018):")
        print("1. Landasan Matematika: Berakar pada topologi aljabar, teori himpunan fuzzy")
        print("   (fuzzy simplicial sets), dan geometri Riemannian.")
        print("2. Preservasi Ganda: Berbeda dari t-SNE yang hanya menjaga struktur lokal,")
        print("   UMAP secara konsisten menjaga struktur lokal sekaligus struktur global")
        print("   (hubungan jarak relatif antar-kelompok besar).")
        print("3. Kecepatan Komputasi: Jauh lebih cepat dari t-SNE standar;")
        print("   skalabilitas komputasi O(N^1.14) vs O(N^2) pada t-SNE.")
        print("4. Dukungan Transformasi Sampel Baru: UMAP mendukung fungsi .transform(X_new)")
        print("   sehingga dapat diintegrasikan ke dalam pipeline produksi.")

    print("======================================================================")

if __name__ == "__main__":
    run_umap_demo()

