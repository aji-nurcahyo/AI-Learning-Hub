"""Scikit-Learn KMeans: Pemodelan Produksi & Interpretasi Centroid Domain
Mendemonstrasikan k-means++ initialization, inspeksi profil fisik centroid,
dan interpretasi segmentasi operasional sensor nirkabel.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_kmeans_sklearn():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=== SCIKIT-LEARN K-MEANS CLUSTERING (k = 3) ===")

    kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    df_result = df.copy()
    df_result["cluster_id"] = labels

    print(f"Total Iterasi Konvergensi : {kmeans.n_iter_}")
    print(f"Nilai Akhir Inertia (WCSS): {kmeans.inertia_:.2f}")

    print("\nDistribusi Sampel per Klaster:")
    print(df_result["cluster_id"].value_counts().sort_index())

    # Kembalikan koordinat centroid ke skala satuan fisik asli
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    df_centroids = pd.DataFrame(centroids_orig, columns=features)
    df_centroids.index.name = "Cluster"

    print("\nProfil Nilai Rata-rata Centroid (Satuan Fisik Asli):")
    print(df_centroids.round(2).to_string())

    print("\nInterpretasi Domain Rekayasa Sensor:")
    for cid, row in df_centroids.iterrows():
        if row["temperature_c"] > 45.0 or row["packet_loss_pct"] > 8.0:
            desc = "Simpul Kritis / Degradasi Termal (Suhu tinggi, paket loss tinggi, baterai turun)"
        elif row["energy_consumption_mah"] > 70.0 or row["latency_ms"] > 60.0:
            desc = "Simpul Beban Tinggi / Transmisi Aktif (Daya tinggi, latensi menengah)"
        else:
            desc = "Simpul Normal / Mode Hemat Energi (Sinyal prima, latensi rendah, adem)"
        print(f"  - Klaster {cid}: {desc}")

    print("\nCatatan Kritis:")
    print("Label ID klaster (0, 1, 2) bersifat arbitrer dan dapat bertukar antar-eksekusi.")
    print("Nama kategori di atas adalah hasil interpretasi domain manusia, bukan keluaran otomatis algoritma.")

if __name__ == "__main__":
    run_kmeans_sklearn()

