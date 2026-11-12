"""Hierarchical Agglomerative Clustering (Klasterisasi Hirarki Aglomeratif)
Mendemonstrasikan pendekatan bottom-up merging dan perbandingan metode linkage:
Ward, Complete, Average, dan Single Linkage.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

def run_hierarchical():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = StandardScaler().fit_transform(df[features])

    print("=== HIERARCHICAL AGGLOMERATIVE CLUSTERING (k = 3) ===")
    print(f"Data: {len(X)} sampel observasi sensor\n")

    linkages = {
        "ward": "Meminimalkan pertambahan varians total WCSS saat penggabungan (Asumsi Euclidean)",
        "complete": "Jarak maksimum antar-dua titik di kedua klaster (Mencegah chaining)",
        "average": "Rata-rata jarak seluruh pasangan titik antar-dua klaster",
        "single": "Jarak minimum (titik terdekat) antar-dua klaster (Rentan fenomena chaining)"
    }

    print(f"{'Metode Linkage':<16} | {'Silhouette Score':<18} | {'Davies-Bouldin (Rendah=Baik)':<28} | {'Distribusi Ukuran Klaster'}")
    print("-" * 92)

    for link, desc in linkages.items():
        agg = AgglomerativeClustering(n_clusters=3, linkage=link)
        labels = agg.fit_predict(X)

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)

        unique, counts = np.unique(labels, return_counts=True)
        dist_str = ", ".join([f"K{k}:{c}" for k, c in zip(unique, counts)])

        print(f"{link:<16} | {sil:14.4f}   | {db:22.4f}       | {dist_str}")

    print("-" * 92)
    print("Takeaways Perilaku Linkage:")
    print("1. Ward Linkage menghasilkan klaster paling seimbang dan kompak (paling mendekati K-Means).")
    print("2. Single Linkage rentan 'chaining effect' di mana titik-titik membentuk satu rantai panjang")
    print("   dan menyisakan klaster-klaster berukuran sangat kerdil (outlier chaining).")

if __name__ == "__main__":
    run_hierarchical()

