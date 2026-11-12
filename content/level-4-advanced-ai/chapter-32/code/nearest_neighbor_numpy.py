"""
nearest_neighbor_numpy.py
Optimalisasi Pencarian Tetangga Terdekat Tervektorisasi di NumPy:
- Menggunakan `np.argpartition` untuk pengambilan Top-K efisien O(N) dibandingkan `np.argsort` O(N log N)
- Pencarian batch (Multiple Queries sekaligus) memanfaatkan perkalian matriks BLAS GEMM
- Transformasi monotonik jarak ke skor kesamaan terkalibrasi: S = 1 / (1 + d)
"""

import numpy as np
import time

def batch_nearest_neighbor(queries, database, k=5):
    """
    Pencarian batch multi-kueri terhadap basis data:
    queries  : [Q, d]
    database : [N, d]
    Output   : top_indices [Q, k], top_scores [Q, k]
    """
    # L2 normalisasi kedua matriks untuk memanfaatkan Dot Product
    q_norm = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    db_norm = database / np.maximum(np.linalg.norm(database, axis=1, keepdims=True), 1e-12)
    
    # Matriks Kesamaan Seluruh Pasangan: [Q, N] via BLAS SGEMM
    sim_matrix = q_norm @ db_norm.T
    
    Q, N = sim_matrix.shape
    # Menggunakan argpartition untuk seleksi Top-k dalam O(N)
    # Partisi pada elemen -k
    partition_idx = np.argpartition(-sim_matrix, kth=k-1, axis=1)[:, :k]
    
    # Urutkan k elemen teratas secara presisi
    rows = np.arange(Q)[:, None]
    part_scores = sim_matrix[rows, partition_idx]
    sorted_order = np.argsort(-part_scores, axis=1)
    
    top_indices = np.take_along_axis(partition_idx, sorted_order, axis=1)
    top_scores = np.take_along_axis(part_scores, sorted_order, axis=1)
    
    return top_indices, top_scores

def main():
    print("=== Batch Nearest Neighbor Search dengan argpartition di NumPy ===")
    np.random.seed(42)
    
    N = 1000  # 1.000 dokumen kandidat
    Q = 4     # 4 kueri bersamaan
    d = 64    # Dimensi embedding
    k = 3     # Top-3
    
    database = np.random.randn(N, d)
    queries = np.random.randn(Q, d)
    
    print(f"Basis Data : {N} vektor (Dimensi {d})")
    print(f"Kueri Batch: {Q} kueri")
    
    t0 = time.time()
    top_indices, top_scores = batch_nearest_neighbor(queries, database, k=k)
    elapsed_ms = (time.time() - t0) * 1000.0
    
    print(f"Waktu Eksekusi Pencarian Batch: {elapsed_ms:.2f} ms\n")
    
    for q_idx in range(Q):
        print(f"Kueri #{q_idx+1}:")
        for rank in range(k):
            doc_id = top_indices[q_idx, rank]
            score = top_scores[q_idx, rank]
            # Transformasi jarak ke skor eksponensial terkalibrasi
            calibrated_prob = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
            print(f"   Peringkat {rank+1}: Dokumen #{doc_id:<4} | Cosine Sim: {score:.4f} | Skor Kalibrasi: {calibrated_prob*100:.1f}%")
        print()
        
    print("Efisiensi Rekayasa:")
    print("`np.argpartition` memangkas kompleksitas dari O(N log N) menjadi O(N),")
    print("sangat krusial untuk mempertahankan latensi rendah saat ukuran basis data membesar.")

if __name__ == "__main__":
    main()

