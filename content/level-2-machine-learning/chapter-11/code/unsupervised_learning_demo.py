"""
Chapter 11: Introduction to Machine Learning
Script 4: unsupervised_learning_demo.py

Tujuan:
Mengenalkan paradigma Unsupervised Learning:
- Tidak ada target atau label y (Unlabeled Data).
- Algoritma bertugas menemukan struktur tersembunyi, klaster (kelompok),
  atau pola intrinsik dari matriks fitur X itu sendiri.
- Menggunakan K-Means Clustering pada data profil konsumsi energi 2D.
"""

import numpy as np
from sklearn.cluster import KMeans


def main():
    print("=" * 65)
    print("DEMO: UNSUPERVISED LEARNING (K-MEANS CLUSTERING)")
    print("=" * 65)

    # Data profil konsumsi 12 gedung kampus tanpa label
    # Fitur: [Konsumsi Daya Siang Hari (kWh), Konsumsi Daya Malam Hari (kWh)]
    X = np.array([
        # Kelompok 1: Gedung Kuliah (Tinggi siang, Rendah malam)
        [150.0, 20.0], [165.0, 25.0], [140.0, 18.0], [155.0, 22.0],
        # Kelompok 2: Gedung Laboratorium Komputasi (Tinggi siang, Tinggi malam)
        [220.0, 180.0], [210.0, 195.0], [235.0, 190.0], [225.0, 185.0],
        # Kelompok 3: Gudang & Fasilitas Umum (Rendah siang, Rendah malam)
        [30.0, 15.0], [25.0, 12.0], [35.0, 20.0], [28.0, 14.0]
    ])

    print(f"Total Sampel Tanpa Label: {len(X)} gedung")
    print("Sampel Fitur X (Siang vs Malam):")
    print(X[:4])

    # Inisialisasi K-Means dengan k=3 klaster
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    print("\nHasil Klasterisasi Tanpa Bimbingan Label:")
    for i, (sample, cl) in enumerate(zip(X, cluster_labels)):
        print(f"  Gedung #{i+1:02d} [Siang: {sample[0]:5.1f} kWh, Malam: {sample[1]:5.1f} kWh] -> Klaster {cl}")

    print("\nTitik Pusat Klaster (Cluster Centroids):")
    for cl_idx, centroid in enumerate(kmeans.cluster_centers_):
        print(f"  Pusat Klaster {cl_idx}: Siang = {centroid[0]:.2f} kWh, Malam = {centroid[1]:.2f} kWh")

    print("\nInersia Klaster (Sum of Squared Distances):", round(kmeans.inertia_, 2))
    print("\n--- Kesimpulan Penting ---")
    print("Unsupervised learning menemukan struktur alami kelompok data")
    print("berdasarkan kedekatan jarak geometris tanpa memerlukan intervensi manusia")
    print("untuk menandai label jawaban yang benar terlebih dahulu.")
    print("=" * 65)


if __name__ == "__main__":
    main()

