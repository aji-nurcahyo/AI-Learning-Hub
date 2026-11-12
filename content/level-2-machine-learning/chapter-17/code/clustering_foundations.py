"""Fondasi Pembelajaran Tanpa Pengawasan (Unsupervised Learning) & Clustering
Mendemonstrasikan perbedaan mendasar antara Klasifikasi (Supervised) dan Klasterisasi (Unsupervised),
matriks kedekatan jarak, dan dampak standarisasi fitur.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def run_foundations():
    data_path = pathlib.Path(__file__).parent / "data" / "wireless_sensor_telemetry.csv"
    df = pd.read_csv(data_path)

    features = ["signal_strength_dbm", "packet_loss_pct", "latency_ms", "temperature_c", "voltage_v", "energy_consumption_mah"]
    X = df[features]

    print("=== FONDASI UNSUPERVISED LEARNING & CLUSTERING ===")
    print(f"Total Sampel Data: {len(X)} observasi")
    print(f"Dimensi Fitur (D): {len(features)} variabel kontinu")
    print("Catatan: Dalam Unsupervised Learning, algoritma HANYA menerima matriks X;")
    print("TIDAK ADA label target (y) yang diberikan selama proses pemodelan.\n")

    print("1. Ringkasan Statistik Rentang Fitur Mentah:")
    for col in features:
        print(f"   {col:<25}: Min = {X[col].min():6.2f} | Max = {X[col].max():6.2f} | Range = {X[col].max() - X[col].min():6.2f}")

    print("\n2. Penskalaan Fitur (StandardScaler):")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("   Data berhasil distandarisasi ke Z-Score (Mean ~ 0.0, Std ~ 1.0).")
    print("   Penskalaan mencegah variabel berentang besar (Latensi: rentang ~350)")
    print("   mendominasi variabel berentang sempit (Voltase: rentang ~1.6) secara sepihak.")

    # 3. Matriks Jarak Pasangan (Pairwise Distance) 5 Sampel Awal
    subset = X_scaled[:4]
    diff = subset[:, np.newaxis, :] - subset[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    print("\n3. Contoh Matriks Jarak Euclidean (4 Sampel Terstandarisasi Pertama):")
    print("        " + "  ".join([f"Sampel {i}" for i in range(4)]))
    for i, row in enumerate(dist_matrix):
        row_str = "  ".join([f"{val:8.3f}" for val in row])
        print(f"Sampel {i}: {row_str}")

    print("\nKesimpulan Konseptual:")
    print("Clustering bertugas mengelompokkan sampel sehingga sampel dalam satu klaster")
    print("memiliki jarak dekat (kemiripan tinggi), sedangkan sampel antar-klaster berjauhan.")

if __name__ == "__main__":
    run_foundations()

