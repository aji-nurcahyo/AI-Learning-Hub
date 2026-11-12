"""
faiss_flat_index.py
Pengenalan Indeks Vektor Datar FAISS (IndexFlatL2 dan IndexFlatIP):
- Struktur indeks datar (Flat Index) tanpa kompresi kehilangan data (Lossless Brute-Force)
- API standar FAISS: `index.add(xb)` dan `D, I = index.search(xq, k)`
- Dukungan fallback otomatis menggunakan NumPy murni jika pustaka faiss-cpu belum terpasang
- Manajemen pemetaan ID vektor ke metadata dokumen aplikasi
"""

import numpy as np

# Cek ketersediaan pustaka FAISS CPU
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class PureNumPyIndexFlat:
    """
    Kelas Fallback Mandiri yang mengimplementasikan antarmuka (API) persis FAISS
    menggunakan aljabar NumPy jika pustaka faiss-cpu tidak tersedia di sistem.
    """
    def __init__(self, d, metric="L2"):
        self.d = d
        self.metric = metric
        self.vectors = None
        self.ntotal = 0
        
    def add(self, xb):
        xb = np.ascontiguousarray(xb, dtype=np.float32)
        if self.vectors is None:
            self.vectors = xb
        else:
            self.vectors = np.vstack([self.vectors, xb])
        self.ntotal = len(self.vectors)
        
    def search(self, xq, k):
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        Q = xq.shape[0]
        if self.metric == "L2":
            # Hitung jarak Euclidean kuadrat ||xq - xb||^2
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * <a, b>
            q_sq = np.sum(xq ** 2, axis=1, keepdims=True)
            db_sq = np.sum(self.vectors ** 2, axis=1, keepdims=True).T
            dot = xq @ self.vectors.T
            sq_dists = np.maximum(q_sq + db_sq - 2 * dot, 0.0)
            
            # Ambil k jarak terendah
            idx = np.argsort(sq_dists, axis=1)[:, :k]
            dists = np.take_along_axis(sq_dists, idx, axis=1)
            return dists, idx
        else:
            # Inner Product (IP): Semakin besar semakin mirip
            sims = xq @ self.vectors.T
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
            return scores, idx

def create_index(d, metric="L2"):
    if HAS_FAISS:
        if metric == "L2":
            return faiss.IndexFlatL2(d), "FAISS C++ Backend (IndexFlatL2)"
        else:
            return faiss.IndexFlatIP(d), "FAISS C++ Backend (IndexFlatIP)"
    else:
        return PureNumPyIndexFlat(d, metric=metric), "NumPy Fallback Engine (FAISS-Compatible API)"

def main():
    print("=== Indeks Vektor Datar FAISS (IndexFlatL2 & IndexFlatIP) ===")
    np.random.seed(42)
    
    d = 8      # Dimensi vektor
    N = 10     # 10 dokumen
    k = 3      # Top-3
    
    # Metadata dokumen terpisah (FAISS hanya menyimpan vektor & indeks numerik)
    metadata_store = {
        0: "DOC-01: Jaringan Syaraf Tiruan",
        1: "DOC-02: Propagasi Balik & Gradient Descent",
        2: "DOC-03: Convolutional Neural Networks",
        3: "DOC-04: Deteksi Objek YOLO",
        4: "DOC-05: Representasi Vektor Word2Vec",
        5: "DOC-06: Self-Attention Transformer",
        6: "DOC-07: Large Language Models",
        7: "DOC-08: Adaptasi LoRA",
        8: "DOC-09: Transfer Learning Dasar",
        9: "DOC-10: Retrieval-Augmented Generation"
    }
    
    # Bangun basis data vektor sintetis
    xb = np.random.randn(N, d).astype(np.float32)
    xq = np.random.randn(1, d).astype(np.float32)
    
    # 1. Indeks Flat L2 (Jarak Euclidean Kuadrat)
    index_l2, engine_l2 = create_index(d, metric="L2")
    print(f"[1] Membangun Indeks L2 menggunakan: {engine_l2}")
    index_l2.add(xb)
    print(f"    Total vektor tersimpan dalam indeks: {index_l2.ntotal}")
    
    D_l2, I_l2 = index_l2.search(xq, k)
    print("\n    Hasil Pencarian IndexFlatL2 (Jarak Terendah):")
    for rank in range(k):
        v_id = int(I_l2[0, rank])
        dist = float(D_l2[0, rank])
        print(f"    Top-{rank+1}: ID #{v_id} | Jarak L2^2: {dist:.4f} | {metadata_store[v_id]}")
        
    # 2. Indeks Flat IP (Inner Product / Cosine Similarity pada Vektor Satuan)
    print("\n[2] Membangun Indeks Flat IP (Cosine Similarity via Normalisasi L2):")
    # Normalisasi vektor ke satuan L2
    xb_norm = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    xq_norm = xq / np.linalg.norm(xq, axis=1, keepdims=True)
    
    index_ip, engine_ip = create_index(d, metric="IP")
    index_ip.add(xb_norm)
    
    D_ip, I_ip = index_ip.search(xq_norm, k)
    print("    Hasil Pencarian IndexFlatIP (Cosine Similarity Tertinggi):")
    for rank in range(k):
        v_id = int(I_ip[0, rank])
        sim = float(D_ip[0, rank])
        print(f"    Top-{rank+1}: ID #{v_id} | Cosine Sim: {sim:.4f} | {metadata_store[v_id]}")
        
    print("\nArsitektur Penyimpanan Database Vektor:")
    print("1. FAISS secara default hanya menyimpan koordinat vektor `xb` dan ID baris `0 s/d N-1`.")
    print("2. Teks asli, judul, dan data atribut lainnya wajib disimpan pada database terpisah (Metadata Store),")
    print("   lalu digabungkan kembali (*hydration*) berdasarkan ID hasil pencarian FAISS.")

if __name__ == "__main__":
    main()

