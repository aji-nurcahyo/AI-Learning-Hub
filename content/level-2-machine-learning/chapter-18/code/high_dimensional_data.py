"""Eksplorasi Fenomena Kutukan Dimensi (Curse of Dimensionality)
Mendemonstrasikan konsentrasi jarak dan volume hiperkubus berdimensi tinggi.
"""

import numpy as np

def demonstrate_distance_concentration():
    np.random.seed(42)
    n_samples = 200
    dimensions = [2, 5, 10, 50, 100, 500, 1000]

    print("======================================================================")
    print("DEMONSTRASI KONSENTRASI JARAK DALAM RUANG BERDIMENSI TINGGI")
    print("======================================================================")
    print(f"{'Dimensi (D)':<14} {'Jarak Min':<14} {'Jarak Max':<14} {'Rasio Min/Max':<16} {'Selisih (Max-Min)/Min'}")
    print("----------------------------------------------------------------------")

    for d in dimensions:
        # Pembangkitan sampel seragam dalam hiperkubus unit [0, 1]^D
        X = np.random.uniform(0, 1, size=(n_samples, d))
        # Menghitung matriks jarak pairwise Euclidean
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Mengabaikan jarak diagonal diri sendiri (jarak = 0)
        np.fill_diagonal(dist_matrix, np.nan)
        d_min = np.nanmin(dist_matrix)
        d_max = np.nanmax(dist_matrix)
        ratio = d_min / d_max
        rel_diff = (d_max - d_min) / d_min

        print(f"{d:<14} {d_min:<14.4f} {d_max:<14.4f} {ratio:<16.4f} {rel_diff:.4f}")

    print("======================================================================")
    print("KESIMPULAN TEORETIS:")
    print("1. Semakin tinggi dimensi D, rasio (d_min / d_max) mendekati 1.0.")
    print("2. Jarak ke tetangga terdekat menjadi hampir sama dengan jarak ke titik terjauh.")
    print("3. Konsep 'ketetanggaan lokal' kehilangan diskriminasi metrik (Curse of Dimensionality).")

if __name__ == "__main__":
    demonstrate_distance_concentration()

