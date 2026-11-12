"""Evaluasi Kualitas Klasterisasi Tanpa Label Ground-Truth
Mendemonstrasikan metrik validasi internal (Silhouette, Davies-Bouldin, Calinski-Harabasz)
serta perbandingannya dengan metrik eksternal (ARI, NMI).
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

def run_clustering_evaluation():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = StandardScaler().fit_transform(df[features])

    # Profil laten hanya digunakan untuk verifikasi metrik eksternal demonstrasi
    latent_true = df["latent_profile"]

    print("=== EVALUASI METRIK KLASTERISASI INTERNAL VS EKSTERNAL ===")
    print("Mengevaluasi K-Means untuk k in [2, 6]:\n")
    print(f"{'Nilai k':<8} | {'Silhouette (Tinggi)':<17} | {'Davies-Bouldin (Rendah)':<22} | {'Calinski-Harabasz (Tinggi)':<26} | {'ARI (Eksternal)':<16}")
    print("-" * 88)

    for k in range(2, 7):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X)

        # Metrik Internal (Murni tanpa label)
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)

        # Metrik Eksternal (Hanya jika ground-truth tersedia)
        ari = adjusted_rand_score(latent_true, labels)

        print(f"k = {k:<4} | {sil:10.4f}      | {db:15.4f}     | {ch:18.2f}       | {ari:12.4f}")

    print("-" * 88)
    print("\nDefinisi & Karakteristik Metrik Internal:")
    print("1. Silhouette Score [-1, +1]      : Membandingkan jarak intra-klaster a(i) vs inter-klaster b(i). Nilai tinggi = klaster padat & terpisah jauh.")
    print("2. Davies-Bouldin Index [0, inf)  : Rasio jumlah dispersi klaster terhadap jarak antar-centroid. Nilai rendah = klaster kompak & terisolasi baik.")
    print("3. Calinski-Harabasz Score [0, inf): Rasio dispersi antar-klaster (Between) terhadap dalam-klaster (Within). Nilai tinggi = geometri klaster terdefinisi kuat.")
    print("\nCatatan Kritis:")
    print("Pada skenario Unsupervised Learning di dunia industri nyata, metrik eksternal (seperti ARI dan NMI)")
    print("TIDAK PERNAH DAPAT DIHITUNG karena tidak ada ground-truth label yang tersedia.")

if __name__ == "__main__":
    run_clustering_evaluation()
