"""Matrix Operations and Linear Transformations for AI.

Chapter 06: Probability, Statistics & Linear Algebra for AI
Artificial Intelligence Learning Hub - ITERA

Skrip ini mendemonstrasikan operasi matriks fundamental dari awal:
1. Validasi Dimensi dan Bentuk (Matrix Shape & Dimensions)
2. Penjumlahan Matriks & Perkalian Skalar
3. Transposisi Matriks (Transpose A^T)
4. Perkalian Matriks-Vektor (A * x)
5. Perkalian Matriks-Matriks (A * B) beserta Syarat Kompatibilitas Dimensi
6. Perbedaan Perkalian Element-wise (Hadamard) vs Perkalian Matriks Formal
7. Transformasi Linear 2D (Penskalaan dan Refleksi)
"""

import sys
from typing import List, Tuple

sys.stdout.reconfigure(encoding="utf-8")

Matrix = List[List[float]]
Vector = List[float]


def get_shape(matrix: Matrix) -> Tuple[int, int]:
    """Mengembalikan dimensi matriks (jumlah_baris, jumlah_kolom)."""
    if not matrix or not matrix[0]:
        return (0, 0)
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    for r in matrix:
        if len(r) != num_cols:
            raise ValueError("Bentuk matriks tidak konsisten (ragged matrix).")
    return (num_rows, num_cols)


def matrix_add(a: Matrix, b: Matrix) -> Matrix:
    """Menghitung penjumlahan matriks A + B."""
    shape_a = get_shape(a)
    shape_b = get_shape(b)
    if shape_a != shape_b:
        raise ValueError(f"Dimensi tidak cocok untuk penjumlahan: {shape_a} != {shape_b}")
    return [[a[i][j] + b[i][j] for j in range(shape_a[1])] for i in range(shape_a[0])]


def matrix_transpose(a: Matrix) -> Matrix:
    """Menghitung transpose A^T di mana elemen (i, j) menjadi (j, i)."""
    rows, cols = get_shape(a)
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def matrix_vector_mult(a: Matrix, x: Vector) -> Vector:
    """Menghitung perkalian matriks-vektor A * x.
    
    A berukuran (m x k), x berukuran (k) -> menghasilkan vektor berukuran (m).
    """
    rows_a, cols_a = get_shape(a)
    if cols_a != len(x):
        raise ValueError(f"Dimensi tidak cocok: Matriks berkolom {cols_a}, sedangkan panjang vektor {len(x)}")
    return [sum(a[i][k] * x[k] for k in range(cols_a)) for i in range(rows_a)]


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    """Menghitung perkalian matriks A * B.
    
    A berukuran (m x k), B berukuran (k x n) -> Hasil (m x n).
    """
    rows_a, cols_a = get_shape(a)
    rows_b, cols_b = get_shape(b)
    if cols_a != rows_b:
        raise ValueError(
            f"Dimensi dalam tidak cocok untuk perkalian: A({rows_a}x{cols_a}) * B({rows_b}x{cols_b}). "
            f"Kolom A ({cols_a}) harus sama dengan Baris B ({rows_b})!"
        )
    result = []
    for i in range(rows_a):
        row_res = []
        for j in range(cols_b):
            cell_sum = sum(a[i][k] * b[k][j] for k in range(cols_a))
            row_res.append(cell_sum)
        result.append(row_res)
    return result


def hadamard_product(a: Matrix, b: Matrix) -> Matrix:
    """Menghitung perkalian per elemen (Hadamard / Element-wise Product A ⊙ B)."""
    shape_a = get_shape(a)
    shape_b = get_shape(b)
    if shape_a != shape_b:
        raise ValueError(f"Dimensi harus identik untuk Hadamard product: {shape_a} != {shape_b}")
    return [[a[i][j] * b[i][j] for j in range(shape_a[1])] for i in range(shape_a[0])]


def print_matrix(name: str, mat: Matrix):
    """Mencetak matriks dengan format rapi."""
    rows, cols = get_shape(mat)
    print(f"Matriks {name} ({rows}x{cols}):")
    for row in mat:
        print("  [" + ", ".join(f"{val:>8.2f}" for val in row) + "]")
    print()


def run_matrix_experiments():
    print("=" * 70)
    print("EKSPERIMEN 5: OPERASI MATRIKS & TRANSFORMASI LINEAR")
    print("=" * 70)

    # Matriks Fitur Data (3 sampel x 2 fitur)
    data_x = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]

    # Matriks Bobot Linear Layer AI (2 input x 3 neuron output)
    weights_w = [
        [0.5, 1.0, -0.5],
        [1.5, -2.0, 0.0]
    ]

    print_matrix("Fitur Input X", data_x)
    print_matrix("Bobot Layer W", weights_w)

    print("--- 1. Transposisi Matriks ---")
    x_t = matrix_transpose(data_x)
    print_matrix("Transpose X^T", x_t)

    print("--- 2. Perkalian Matriks (Feedforward AI Layer: Y = X * W) ---")
    y_output = matrix_multiply(data_x, weights_w)
    print_matrix("Output Aktivasi Y (3x3)", y_output)

    print("--- 3. Pembedaan Hadamard vs Perkalian Matriks ---")
    mat_p = [[1.0, 2.0], [3.0, 4.0]]
    mat_q = [[5.0, 6.0], [7.0, 8.0]]
    print_matrix("P", mat_p)
    print_matrix("Q", mat_q)

    hadamard_res = hadamard_product(mat_p, mat_q)
    matmul_res = matrix_multiply(mat_p, mat_q)
    print_matrix("Element-wise (Hadamard) P ⊙ Q", hadamard_res)
    print_matrix("Matrix Multiplication P * Q", matmul_res)

    print("--- 4. Transformasi Linear Geometris 2D ---")
    # Transformasi rotasi 90 derajat berlawanan jarum jam:
    # [ [0, -1],
    #   [1,  0] ]
    rot_90 = [[0.0, -1.0], [1.0, 0.0]]
    point = [3.0, 2.0]
    transformed = matrix_vector_mult(rot_90, point)
    print(f"Titik Asal        : (x={point[0]}, y={point[1]})")
    print(f"Titik Hasil Rotasi: (x={transformed[0]}, y={transformed[1]}) (x' = -y = -2, y' = x = 3)")
    print("=" * 70)


if __name__ == "__main__":
    run_matrix_experiments()

