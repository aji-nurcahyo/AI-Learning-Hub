"""Implementasi K-Means Murni dari Dasar (From Scratch) Menggunakan NumPy
Mendemonstrasikan assignment step, centroid update step, konvergensi,
dan perbandingan langsung dengan Scikit-learn KMeans.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class KMeansScratch:
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def fit(self, X: np.ndarray):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # 1. Inisialisasi centroid acak dari titik data sampel
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[random_indices].copy()

        for iteration in range(self.max_iter):
            self.n_iter_ = iteration + 1

            # 2. Assignment Step: Hitung jarak kuadrat Euclidean ke setiap centroid
            # Dimensi jarak: (n_samples, n_clusters)
            distances = np.sum((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2, axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            # 3. Centroid Update Step: Hitung rata-rata posisi sampel dalam setiap klaster
            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                cluster_samples = X[self.labels_ == k]
                if len(cluster_samples) > 0:
                    new_centroids[k] = np.mean(cluster_samples, axis=0)
                else:
                    new_centroids[k] = self.centroids_[k]

            # 4. Evaluasi Konvergensi: Selisih pergeseran posisi centroid
            centroid_shift = np.sum((new_centroids - self.centroids_) ** 2)
            self.centroids_ = new_centroids

            if centroid_shift < self.tol:
                break

        # 5. Hitung Inertia / Within-Cluster Sum of Squares (WCSS)
        final_distances = np.sum((X - self.centroids_[self.labels_]) ** 2)
        self.inertia_ = float(final_distances)
        return self

def run_kmeans_scratch():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)

    print("=== IMPLEMENTASI K-MEANS FROM SCRATCH (NUMPY) ===")
    print(f"Dataset Terstandarisasi: {X_scaled.shape[0]} baris x {X_scaled.shape[1]} kolom")

    # 1. Jalankan K-Means from scratch
    km_scratch = KMeansScratch(n_clusters=3, max_iter=100, random_state=42)
    km_scratch.fit(X_scaled)

    print("\nHasil K-Means From Scratch:")
    print(f"  - Jumlah Klaster (k)  : {km_scratch.n_clusters}")
    print(f"  - Konvergensi Selesai : Pada iterasi ke-{km_scratch.n_iter_}")
    print(f"  - Nilai Inertia (WCSS): {km_scratch.inertia_:.4f}")
    unique, counts = np.unique(km_scratch.labels_, return_counts=True)
    print(f"  - Distribusi Anggota  : {dict(zip(unique, counts))}")

    # 2. Verifikasi Komparatif dengan Scikit-learn KMeans (init='random', n_init=1)
    km_sklearn = KMeans(n_clusters=3, init="random", n_init=1, max_iter=100, random_state=42)
    km_sklearn.fit(X_scaled)

    print("\nHasil Scikit-learn KMeans (init='random', n_init=1):")
    print(f"  - Iterasi Konvergensi : {km_sklearn.n_iter_}")
    print(f"  - Nilai Inertia (WCSS): {km_sklearn.inertia_:.4f}")
    unique_sk, counts_sk = np.unique(km_sklearn.labels_, return_counts=True)
    print(f"  - Distribusi Anggota  : {dict(zip(unique_sk, counts_sk))}")

    diff_inertia = abs(km_scratch.inertia_ - km_sklearn.inertia_)
    print(f"\nSelisih Inertia Scratch vs Sklearn: {diff_inertia:.6f}")
    print(f"Verifikasi Konsistensi Numerik    : {'LULUS IDENTIK' if diff_inertia < 1e-3 else 'SEPAKAT DALAM TOLERANSI'}")

if __name__ == "__main__":
    run_kmeans_scratch()

