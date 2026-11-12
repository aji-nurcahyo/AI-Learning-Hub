"""
faiss_ann_demo.py
Konsep Approximate Nearest Neighbor (ANN) dengan Inverted File Index (IVF):
- Menjelaskan partisi ruang vektor ke dalam sel Voronoi (Voronoi Cells / K-Means Centroids)
- Hiperparameter nlist (jumlah sel partisi) dan nprobe (jumlah sel yang diperiksa saat kueri)
- Trade-off fundamental: Waktu Pencarian (Search Speed) vs Akurasi Temu Kembali (Recall)
- Implementasi mandiri dengan fallback NumPy murni jika pustaka FAISS belum terpasang
"""

import numpy as np
import time

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class PureNumPyIndexIVF:
    """
    Simulasi mandiri Inverted File Index (IVF) menggunakan K-Means dan partisi sel Voronoi.
    """
    def __init__(self, d, nlist=10, nprobe=2):
        self.d = d
        self.nlist = nlist
        self.nprobe = nprobe
        self.centroids = None
        self.inverted_lists = {i: [] for i in range(nlist)} # cell_idx -> list of (doc_id, vector)
        self.ntotal = 0
        
    def train_and_add(self, xb):
        N = len(xb)
        # 1. K-Means sederhana untuk menemukan nlist centroid
        np.random.seed(42)
        init_idx = np.random.choice(N, self.nlist, replace=False)
        self.centroids = xb[init_idx].copy()
        
        # 5 iterasi K-Means cepat untuk memusatkan centroid
        for _ in range(5):
            dists = np.sum((xb[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1)
            assignments = np.argmin(dists, axis=1)
            for c in range(self.nlist):
                members = xb[assignments == c]
                if len(members) > 0:
                    self.centroids[c] = np.mean(members, axis=0)
                    
        # 2. Tambahkan vektor ke inverted list masing-masing sel Voronoi
        dists = np.sum((xb[:, None, :] - self.centroids[None, :, :]) ** 2, axis=-1)
        assignments = np.argmin(dists, axis=1)
        for doc_id, c in enumerate(assignments):
            self.inverted_lists[c].append((doc_id, xb[doc_id]))
        self.ntotal = N

    def search(self, xq, k):
        Q = len(xq)
        all_indices = []
        all_dists = []
        
        for q_idx in range(Q):
            q = xq[q_idx]
            # 1. Temukan nprobe centroid terdekat
            c_dists = np.sum((self.centroids - q) ** 2, axis=1)
            closest_cells = np.argsort(c_dists)[:self.nprobe]
            
            # 2. Pindai hanya dokumen di dalam sel-sel terpilih
            candidates = []
            for cell in closest_cells:
                candidates.extend(self.inverted_lists[cell])
                
            if not candidates:
                all_indices.append([-1] * k)
                all_dists.append([float('inf')] * k)
                continue
                
            cand_ids = np.array([c[0] for c in candidates])
            cand_vecs = np.array([c[1] for c in candidates])
            
            # Hitung jarak eksak hanya pada kandidat
            v_dists = np.sum((cand_vecs - q) ** 2, axis=1)
            top_k_order = np.argsort(v_dists)[:k]
            
            all_indices.append(cand_ids[top_k_order].tolist())
            all_dists.append(v_dists[top_k_order].tolist())
            
        return np.array(all_dists), np.array(all_indices)

def main():
    print("=== Approximate Nearest Neighbor (ANN): Inverted File Index (IVF) ===")
    np.random.seed(42)
    
    d = 16
    N = 2000    # 2.000 vektor dalam basis data
    nlist = 20  # Partisi ke dalam 20 sel Voronoi
    k = 5       # Top-5
    
    xb = np.random.randn(N, d).astype(np.float32)
    xq = np.random.randn(5, d).astype(np.float32) # 5 kueri evaluasi
    
    # 1. Ground Truth Eksak (Exact Brute-Force)
    # Menghitung jarak ke seluruh 2.000 vektor
    exact_top_ids = []
    for q in xq:
        dists = np.sum((xb - q) ** 2, axis=1)
        exact_top_ids.append(set(np.argsort(dists)[:k]))
        
    print(f"Total Dokumen (N)    : {N:,} vektor")
    print(f"Dimensi Vektor (d)   : {d}")
    print(f"Jumlah Sel (nlist)   : {nlist} sel Voronoi")
    print(f"Ground Truth Dihitung: {len(xq)} kueri dievaluasi eksak\n")
    
    # 2. Eksperimen Trade-off nprobe (Akurasi vs Kecepatan)
    print(f"{'Nilai nprobe':<15} {'Vektor Diperiksa (%)':<24} {'Recall@5 Terhadap Eksak':<25} {'Status'}")
    print("-" * 75)
    
    # Uji variasi nprobe dari 1 (sangat cepat, recall rendah) hingga nlist (pencarian penuh)
    for nprobe in [1, 2, 5, 10, 20]:
        ivf_index = PureNumPyIndexIVF(d=d, nlist=nlist, nprobe=nprobe)
        ivf_index.train_and_add(xb)
        
        _, ann_indices = ivf_index.search(xq, k=k)
        
        # Hitung rata-rata Recall@k terhadap hasil eksak
        recalls = []
        for q_idx in range(len(xq)):
            retrieved = set(ann_indices[q_idx])
            ground_truth = exact_top_ids[q_idx]
            intersection = len(retrieved.intersection(ground_truth))
            recalls.append(intersection / k)
            
        avg_recall = np.mean(recalls) * 100.0
        pct_scanned = (nprobe / nlist) * 100.0
        note = "Sangat Cepat" if nprobe == 1 else ("Eksak Sempurna" if nprobe == nlist else "Seimbang")
        print(f"nprobe = {nprobe:<8} {pct_scanned:<24.1f}% {avg_recall:<25.1f}% {note}")
        
    print("-" * 75)
    print("\nPrinsip Utama Algoritma ANN (Inverted File Index):")
    print("1. Dengan memeriksa hanya 25% data (nprobe=5 dari 20), model dapat menangkap mayoritas")
    print("   dokumen yang relevan (Recall tinggi), memangkas waktu komputasi secara signifikan.")
    print("2. Pemilihan `nprobe` adalah keputusan arsitektural: semakin besar nprobe, semakin tinggi")
    print("   recall, namun waktu inferensi mendekati pencarian eksak.")

if __name__ == "__main__":
    main()

