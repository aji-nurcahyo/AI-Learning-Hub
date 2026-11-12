"""
Chapter 21 - Neural Network Fundamentals
Script 8: Validasi Bentuk Matriks & Kompatibilitas Dimensi Tensor (Matrix Shape Demo)
Memvalidasi aturan perkalian matriks (M x K) @ (K x P) -> (M x P), penambahan bias broadcasting,
serta pelacakan dimensi tensor pada berbagai variasi ukuran batch (Batch Sizes: 1, 16, 64, 128).
"""

import numpy as np

def validate_tensor_shapes(batch_size, layer_dims):
    """
    Mensimulasikan dan melacak bentuk matriks forward pass.
    layer_dims: [D_in, D_h1, D_h2, ..., D_out]
    """
    print(f"\n--- PENGUJIAN UKURAN BATCH: N = {batch_size} ---")
    
    # 1. Matriks Masukan X: (N, D_in)
    D_in = layer_dims[0]
    X = np.random.randn(batch_size, D_in)
    print(f"Bentuk Matriks Masukan X: {X.shape} (N = {batch_size}, D_in = {D_in})")
    
    A_prev = X
    for l in range(1, len(layer_dims)):
        n_in = layer_dims[l-1]
        n_out = layer_dims[l]
        
        # Bobot W^[l] berukuran (n_in, n_out)
        W = np.random.randn(n_in, n_out) * 0.1
        # Bias b^[l] berukuran (1, n_out)
        b = np.zeros((1, n_out))
        
        # Kombinasi linier: Z^[l] = A_prev @ W^[l] + b^[l]
        # Aturan dimensi: (N, n_in) @ (n_in, n_out) -> (N, n_out)
        Z = np.dot(A_prev, W) + b
        
        # Aktivasi non-linier tidak mengubah dimensi matriks
        A = np.maximum(0.0, Z)
        
        print(f"  Layer {l}: A_prev {A_prev.shape} @ W {W.shape} + b {b.shape} -> Z {Z.shape} -> A {A.shape}")
        
        # Validasi dimensi
        assert Z.shape == (batch_size, n_out), f"Kesalahan dimensi pada Layer {l}!"
        assert A.shape == (batch_size, n_out), f"Kesalahan dimensi aktivasi pada Layer {l}!"
        A_prev = A
        
    print(f"-> Luaran Akhir Y_hat Berhasil Divalidasi: {A_prev.shape}")

def run_matrix_shape_suite():
    print("=" * 85)
    print("CHAPTER 21: VALIDASI BENTUK MATRIKS & DIMENSI TENSOR PADA BERBAGAI BATCH SIZE")
    print("=" * 85)
    
    # Topologi: 10 input -> 32 hidden1 -> 16 hidden2 -> 3 output (Multiclass)
    architecture = [10, 32, 16, 3]
    print(f"Spesifikasi Arsitektur Jaringan: {architecture}")
    print("Aturan Aljabar Linier: Matriks A (N x K) @ Matriks B (K x P) -> Matriks C (N x P)")
    print("Vektor Bias (1 x P) ditambahkan ke setiap baris matriks hasil melalui mekanisme Broadcasting.")
    
    batch_sizes = [1, 16, 64, 128]
    for b_size in batch_sizes:
        validate_tensor_shapes(b_size, architecture)
        
    print("\n" + "=" * 85)
    print("KESIMPULAN VALIDASI:")
    print("1. Dimensi internal (K) harus selalu cocok sempurna antara kolom A dan baris W.")
    print("2. Dimensi baris pertama selalu merefleksikan ukuran batch (N).")
    print("3. Seluruh bentuk matriks terverifikasi konsisten di seluruh variasi batch.")
    print("=" * 85)

if __name__ == "__main__":
    np.random.seed(42)
    run_matrix_shape_suite()

