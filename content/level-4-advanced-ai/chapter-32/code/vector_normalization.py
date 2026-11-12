"""
vector_normalization.py
Normalisasi Vektor (L2 Normalization) dan Kesetaraan Aljabar:
- Transformasi vektor menjadi Vektor Satuan (Unit Vector: ||u||_2 = 1.0)
- Pembuktian matematis: Pada vektor ternormalisasi L2,
  1. Dot Product IDENTIK DENGAN Cosine Similarity: <u_hat, v_hat> = cos(theta)
  2. Jarak Euclidean berbanding lurus langsung dengan Cosine Distance:
     ||u_hat - v_hat||^2 = 2 * (1 - cos(theta))
- Rasionalisasi mengapa database vektor (FAISS/Milvus) menggunakan inner product pada vektor ternormalisasi
"""

import numpy as np

def l2_normalize(matrix_or_vector, eps=1e-12):
    """Normalisasi L2 sepanjang sumbu fitur terakhir."""
    norm = np.linalg.norm(matrix_or_vector, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return matrix_or_vector / norm

def main():
    print("=== Normalisasi Vektor L2 & Pembuktian Aljabar Kesetaraan Metrik ===")
    
    # Dua vektor acak dengan magnitudo dan orientasi berbeda
    u = np.array([3.0, 4.0, 0.0])    # Norm: 5.0
    v = np.array([1.0, 2.0, 2.0])    # Norm: 3.0
    
    print(f"Vektor u Asli : {u} | Norm: {np.linalg.norm(u):.4f}")
    print(f"Vektor v Asli : {v} | Norm: {np.linalg.norm(v):.4f}")
    
    # 1. Hitung Cosine Similarity Langsung dari Vektor Asli
    cos_sim_direct = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    print(f"\n[1] Cosine Similarity Langsung (Vektor Asli) : {cos_sim_direct:.6f}")
    
    # 2. Normalisasi L2 ke Unit Vectors
    u_norm = l2_normalize(u)
    v_norm = l2_normalize(v)
    
    print(f"\n[2] Vektor Satuan Ternormalisasi:")
    print(f"    u_norm : {u_norm} | Norm: {np.linalg.norm(u_norm):.6f}")
    print(f"    v_norm : {v_norm} | Norm: {np.linalg.norm(v_norm):.6f}")
    
    # 3. Pembuktian Teorema 1: Dot Product Vektor Satuan = Cosine Similarity
    dot_normalized = np.dot(u_norm, v_norm)
    print(f"\n[3] Dot Product Vektor Ternormalisasi        : {dot_normalized:.6f}")
    diff_theorem1 = abs(cos_sim_direct - dot_normalized)
    print(f"    Selisih Teorema 1 (Harus 0)             : {diff_theorem1:.8e}")
    assert diff_theorem1 < 1e-12, "Teorema 1 gagal terbukti!"
    print("    -> Terbukti: Dot Product pada vektor satuan adalah Cosine Similarity murni!")
    
    # 4. Pembuktian Teorema 2: Relasi Euclidean Distance & Cosine Distance
    # ||u_hat - v_hat||^2 = 2 * (1 - cos_sim)
    euc_dist_norm = np.linalg.norm(u_norm - v_norm)
    euc_squared = euc_dist_norm ** 2
    theoretical_euc_squared = 2.0 * (1.0 - dot_normalized)
    
    print(f"\n[4] Euclidean Distance Kuadrat Vektor Satuan : {euc_squared:.6f}")
    print(f"    Nilai Teoretis (2 * (1 - cos_sim))       : {theoretical_euc_squared:.6f}")
    diff_theorem2 = abs(euc_squared - theoretical_euc_squared)
    print(f"    Selisih Teorema 2 (Harus 0)              : {diff_theorem2:.8e}")
    assert diff_theorem2 < 1e-12, "Teorema 2 gagal terbukti!"
    print("    -> Terbukti: Pengurutan jarak Euclidean pada vektor satuan IDENTIK 100%")
    print("       dengan pengurutan Cosine Similarity (hanya berbeda skala monotonik)!")
    
    print("\nImplikasi Penting pada Sistem Pencarian Skala Besar (FAISS):")
    print("- Menghitung Cosine Similarity secara naif membutuhkan operasi akar dan pembagian berulang.")
    print("- Dengan pra-normalisasi seluruh dokumen dan kueri (L2 norm), pencarian kesamaan kosinus")
    print("  disederhanakan menjadi operasi matriks perkalian (BLAS GEMM) tercepat: IndexFlatIP.")

if __name__ == "__main__":
    main()

