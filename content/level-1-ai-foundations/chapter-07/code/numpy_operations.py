"""NumPy Operations: Vectorization, Broadcasting & Linear Algebra.

Chapter 07: Python Ecosystem for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan:
1. Operasi Tervektorisasi (Vectorized Element-wise Arithmetic)
2. Mekanisme Broadcasting (Ekspansi Dimensi Otomatis)
3. Agregasi Sepanjang Sumbu (Axis-wise Reductions: axis=0 vs axis=1)
4. Perkalian Matriks Formal menggunakan Operator @
5. Perhitungan Norma dan Aljabar Linear (np.linalg)
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def run_numpy_operations_demo():
    print("=" * 70)
    print("DEMO 2: NUMPY OPERATIONS — VECTORIZATION, BROADCASTING & LINEAR ALGEBRA")
    print("=" * 70)

    # 1. Operasi Tervektorisasi
    features = np.array([10.0, 20.0, 30.0, 40.0])
    print("--- 1. Operasi Tervektorisasi (Tanpa Perulangan for) ---")
    print("Data Fitur Asli :", features)
    print("Skalasi x 2.5   :", features * 2.5)
    print("Kuadrat (** 2)  :", features ** 2)
    print("Logaritma Natural:", np.log(features))
    print("Operasi vektor dieksekusi di level binary C dengan SIMD instruction set!\n")

    # 2. Mekanisme Broadcasting
    # Misalkan ada 4 sampel data sensor dengan 3 fitur masing-masing (Suhu, Kelembaban, Getaran)
    sensor_matrix = np.array([
        [22.0, 45.0, 0.15],
        [24.0, 50.0, 0.20],
        [21.5, 42.0, 0.12],
        [28.5, 55.0, 0.85]
    ])  # Shape (4, 3)

    # Vektor offset kalibrasi (1 per fitur)
    calibration_offsets = np.array([-1.0, 2.0, -0.05])  # Shape (3,)

    # Broadcasting otomatis mereplikasi offsets ke seluruh 4 baris!
    calibrated_data = sensor_matrix + calibration_offsets

    print("--- 2. Mekanisme Broadcasting ---")
    print(f"Matriks Sensor (Shape {sensor_matrix.shape}):\n", sensor_matrix)
    print(f"Offset Kalibrasi (Shape {calibration_offsets.shape}):", calibration_offsets)
    print("Hasil Kalibrasi Otomatis (Broadcasting):\n", calibrated_data, "\n")

    # 3. Agregasi Sepanjang Sumbu (Axis-wise Aggregations)
    # axis=0: Mereduksi baris -> Menghasilkan ringkasan per fitur (kolom)
    # axis=1: Mereduksi kolom -> Menghasilkan ringkasan per sampel (baris)
    mean_per_feature = sensor_matrix.mean(axis=0)
    std_per_feature = sensor_matrix.std(axis=0)
    max_per_sample = sensor_matrix.max(axis=1)

    print("--- 3. Agregasi Berbasis Sumbu (Axis Reductions) ---")
    print(f"Mean per Fitur (axis=0)       : {mean_per_feature}")
    print(f"  -> Rata-rata Suhu={mean_per_feature[0]:.2f}, Kelembaban={mean_per_feature[1]:.2f}, Getaran={mean_per_feature[2]:.2f}")
    print(f"Standar Deviasi (axis=0)      : {std_per_feature}")
    print(f"Nilai Maksimum per Sampel (axis=1): {max_per_sample}\n")

    # Standarisasi Z-Score Tervektorisasi: (X - mean) / std
    z_scores = (sensor_matrix - mean_per_feature) / std_per_feature
    print("Matriks Z-Score Hasil Standarisasi:\n", np.round(z_scores, 3), "\n")

    # 4. Perkalian Matriks Formal menggunakan Operator @
    # Matriks Fitur (4 sampel x 3 fitur) dikalikan Matriks Bobot (3 fitur x 2 kelas probabilitas)
    weights = np.array([
        [0.4, -0.2],
        [0.1, 0.5],
        [1.2, -0.8]
    ])  # Shape (3, 2)

    # Hasil berukuran (4, 2)
    layer_output = sensor_matrix @ weights

    print("--- 4. Perkalian Matriks (Feedforward Layer) ---")
    print(f"Dimensi X ({sensor_matrix.shape}) @ Dimensi W ({weights.shape}) = Output ({layer_output.shape}):")
    print(np.round(layer_output, 2), "\n")

    # 5. Modul Aljabar Linear (np.linalg)
    u = np.array([3.0, 4.0])
    l2_norm = np.linalg.norm(u)
    print("--- 5. Aljabar Linear Lanjutan ---")
    print(f"Vektor u : {u}, Norma L2 (Panjang) = {l2_norm:.2f}")

    # Menyelesaikan Sistem Persamaan Linear A * x = b:
    # 2x + y = 8
    # x + 3y = 14
    mat_A = np.array([[2.0, 1.0], [1.0, 3.0]])
    vec_b = np.array([8.0, 14.0])
    sol_x = np.linalg.solve(mat_A, vec_b)
    print(f"Solusi Sistem Persamaan Linear A*x = b : x = {sol_x[0]:.2f}, y = {sol_x[1]:.2f}")
    print("Verifikasi A @ sol_x:", mat_A @ sol_x)
    print("=" * 70)


if __name__ == "__main__":
    run_numpy_operations_demo()

