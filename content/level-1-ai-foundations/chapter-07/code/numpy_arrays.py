"""NumPy Fundamentals: Arrays, Shapes, Dimensions & Slicing.

Chapter 07: Python Ecosystem for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan:
1. Pembuatan Array NumPy (dari list, arange, linspace, zeros, ones)
2. Atribut Array (shape, ndim, dtype, size, itemsize)
3. Efisiensi Memori: Python List vs NumPy ndarray
4. Pengindeksan dan Pemotongan (1D, 2D row/column slicing)
5. Boolean Masking (Penyaringan Berbasis Kondisi)
6. Manipulasi Bentuk (Reshaping & Flattening)
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def run_numpy_arrays_demo():
    print("=" * 70)
    print("DEMO 1: NUMPY FUNDAMENTALS — ARRAYS, SHAPES & INDEXING")
    print("=" * 70)

    # 1. Pembuatan Array
    arr_1d = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    arr_2d = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
    zeros_mat = np.zeros((2, 3))
    linspace_vec = np.linspace(0.0, 1.0, 5)  # 5 titik terdistribusi merata dari 0 ke 1

    print("--- 1. Pembuatan & Karakteristik Array ---")
    print(f"Array 1D         : {arr_1d}")
    print(f"Shape: {arr_1d.shape}, Dimensi: {arr_1d.ndim}, Tipe: {arr_1d.dtype}, Elemen: {arr_1d.size}\n")

    print("Array 2D (Matriks):\n", arr_2d)
    print(f"Shape: {arr_2d.shape} (2 baris x 3 kolom), Dimensi: {arr_2d.ndim}, Tipe: {arr_2d.dtype}\n")

    print(f"Linspace Vector  : {linspace_vec}\n")

    # 2. Efisiensi Memori: List Python vs NumPy Array
    py_list = list(range(10000))
    np_arr = np.arange(10000, dtype=np.int32)
    # Di Python 64-bit, list menyimpan pointer objek (~8 byte/pointer + overhead PyObject)
    # NumPy menyimpan contiguous C-array (4 byte per int32)
    py_list_bytes = sys.getsizeof(py_list) + sum(sys.getsizeof(i) for i in py_list[:100]) * 100
    np_arr_bytes = np_arr.nbytes
    print("--- 2. Efisiensi Memori Komputasi ---")
    print(f"Ukuran 10.000 integer pada Python List (est) : ~{py_list_bytes / 1024:.1f} KB")
    print(f"Ukuran 10.000 integer pada NumPy ndarray (raw): {np_arr_bytes / 1024:.1f} KB")
    print("NumPy menyimpan data dalam contiguous memory buffer C yang sangat padat dan cepat!\n")

    # 3. Pengindeksan & Pemotongan (Indexing & Slicing)
    grid = np.array([
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [31, 32, 33, 34]
    ])
    print("--- 3. Pengindeksan & Slicing 2D ---")
    print("Matriks Grid (3x4):\n", grid)
    print(f"Elemen Baris 1, Kolom 2 (grid[1, 2]) : {grid[1, 2]}")
    print(f"Baris Pertama Lengkap (grid[0, :])   : {grid[0, :]}")
    print(f"Kolom Terakhir Lengkap (grid[:, -1]) : {grid[:, -1]}")
    print("Sub-matriks 2x2 Tengah (grid[0:2, 1:3]):\n", grid[0:2, 1:3], "\n")

    # 4. Boolean Masking (Sangat krusial untuk filtering data AI)
    mask = grid > 20
    print("--- 4. Boolean Masking ---")
    print("Kondisi grid > 20:\n", mask)
    print("Nilai yang memenuhi kondisi (grid[grid > 20]):\n", grid[mask], "\n")

    # 5. Reshaping
    flat_data = np.arange(1, 13)  # 12 elemen
    reshaped_3x4 = flat_data.reshape((3, 4))
    reshaped_6x2 = flat_data.reshape((6, 2))
    print("--- 5. Reshaping Array ---")
    print(f"Data Awal (12 elemen) : {flat_data}")
    print("Bentuk Baru (3x4):\n", reshaped_3x4)
    print("Bentuk Baru (6x2):\n", reshaped_6x2)
    print("Ravel (Kembali ke 1D):", reshaped_3x4.ravel())
    print("=" * 70)


if __name__ == "__main__":
    run_numpy_arrays_demo()

