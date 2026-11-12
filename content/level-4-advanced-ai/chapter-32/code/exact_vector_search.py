"""
exact_vector_search.py
Algoritma Pencarian Tetangga Terdekat Eksak (Exact Brute-Force Nearest Neighbor):
- Implementasi pencarian eksak dari dasar menggunakan aljabar matriks NumPy murni
- Analisis kompleksitas komputasi: O(N * d) per kueri pencarian
- Perbandingan perankingan jarak Euclidean (L2) vs Inner Product (Cosine Similarity)
- Menjadi tolok ukur kebenaran matematis (Ground Truth) untuk mengevaluasi indeks aproksimasi (ANN)
"""

import numpy as np

def exact_search_l2(query_vec, database_matrix, k=3):
    """
    Pencarian eksak berbasis jarak Euclidean terkecil (L2):
    d(q, v_i) = ||q - v_i||_2
    Kompleksitas: O(N * d)
    """
    # Bentuk perbedaan selisih: [N, d]
    diff = database_matrix - query_vec
    # Jarak kuadrat: sum(diff^2) -> [N]
    sq_dists = np.sum(diff ** 2, axis=1)
    dists = np.sqrt(sq_dists)
    
    # Ambil k indeks dengan jarak terendah
    top_k_indices = np.argsort(dists)[:k]
    top_k_dists = dists[top_k_indices]
    return top_k_indices, top_k_dists

def exact_search_cosine(query_vec, database_matrix, k=3):
    """
    Pencarian eksak berbasis kesamaan kosinus terbesar (Cosine Similarity):
    sim(q, v_i) = <q, v_i> / (||q|| * ||v_i||)
    """
    q_norm = query_vec / np.maximum(np.linalg.norm(query_vec), 1e-12)
    db_norm = database_matrix / np.maximum(np.linalg.norm(database_matrix, axis=1, keepdims=True), 1e-12)
    
    # Dot product: [N]
    similarities = db_norm @ q_norm
    # Ambil k indeks dengan kesamaan tertinggi (pengurutan menurun)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_sims = similarities[top_k_indices]
    return top_k_indices, top_k_sims

def main():
    print("=== Exact Nearest Neighbor Search (Brute-Force NumPy) ===")
    np.random.seed(42)
    
    n_docs = 100
    dim = 16
    k = 5
    
    # Basis data vektor sintetis [N=100, d=16]
    database = np.random.randn(n_docs, dim)
    # Kueri pencarian [d=16]
    query = np.random.randn(dim)
    
    print(f"Jumlah Dokumen di Basis Data (N) : {n_docs}")
    print(f"Dimensi Vektor (d)              : {dim}")
    print(f"Target Hasil Teratas (k)        : {k}\n")
    
    # 1. Pencarian Eksak Berbasis L2 (Jarak Terkecil)
    indices_l2, dists_l2 = exact_search_l2(query, database, k=k)
    print("--- [1] Hasil Pencarian Eksak Jarak Euclidean (L2) Terdekat ---")
    print(f"{'Peringkat':<10} {'Indeks Dokumen':<16} {'Jarak Euclidean (L2)':<20}")
    print("-" * 50)
    for rank, (idx, dist) in enumerate(zip(indices_l2, dists_l2), 1):
        print(f"Top-{rank:<6} Indeks #{idx:<10} {dist:.6f}")
        
    # 2. Pencarian Eksak Berbasis Cosine Similarity (Kesamaan Tertinggi)
    indices_cos, sims_cos = exact_search_cosine(query, database, k=k)
    print("\n--- [2] Hasil Pencarian Eksak Cosine Similarity Tertinggi ---")
    print(f"{'Peringkat':<10} {'Indeks Dokumen':<16} {'Cosine Similarity':<20}")
    print("-" * 50)
    for rank, (idx, sim) in enumerate(zip(indices_cos, sims_cos), 1):
        print(f"Top-{rank:<6} Indeks #{idx:<10} {sim:.6f}")
        
    print("-" * 50)
    print("\nAnalisis Kompleksitas:")
    print(f"- Operasi floating-point per kueri : {n_docs * dim * 3:,} FLOPs.")
    print("- Pada skala jutaan vektor (N = 10^7), pencarian brute-force menjadi penghambat latensi (bottleneck).")
    print("- Inilah alasan mengapa teknik Approximate Nearest Neighbor (ANN) diperlukan di lingkungan produksi.")

if __name__ == "__main__":
    main()

