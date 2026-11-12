"""
semantic_search.py
Pipa Pencarian Semantik End-to-End (Semantic Search Pipeline):
1. Memuat koleksi dokumen dari data/ai_telecom_corpus.json
2. Vektorisasi dokumen ke dalam representasi dense berdimensi 32
3. Normalisasi vektor ke vektor satuan L2
4. Membangun indeks pencarian vektor
5. Menerima kueri bahasa alami pengguna, mengekstrak embedding kueri
6. Melakukan pencarian Top-k kesamaan kosinus dan merekonstruksi teks dokumen (Metadata Hydration)
Catatan: Alur berhenti pada tahap temu kembali (Retrieval), tanpa proses pembangkitan teks (RAG ada di Bab 33).
"""

import json
import os
import numpy as np
import re

class DeterministicSentenceEncoder:
    """Encoder kalimat deterministik mandiri."""
    def __init__(self, embedding_dim=32):
        self.embedding_dim = embedding_dim
        self.hash_space = 2048
        np.random.seed(42)
        proj = np.random.randn(self.hash_space, embedding_dim)
        q, _ = np.linalg.qr(proj)
        self.projection_matrix = q[:, :embedding_dim]

    def encode(self, text):
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = clean.split()
        if not tokens:
            return np.zeros(self.embedding_dim)
        sparse = np.zeros(self.hash_space)
        for t in tokens:
            sparse[hash(t) % self.hash_space] += 1.0
            # Tambahkan bi-gram karakter
            for i in range(len(t) - 1):
                bg = t[i:i+2]
                sparse[hash(bg) % self.hash_space] += 0.5
        vec = np.log1p(sparse) @ self.projection_matrix
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "ai_telecom_corpus.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("=== Pipeline Pencarian Semantik Dokumen AI & Telekomunikasi ===")
    data = load_data()
    docs = data["documents"]
    
    # Inisialisasi Encoder
    encoder = DeterministicSentenceEncoder(embedding_dim=32)
    
    # 1. Bangun Index Vektor Seluruh Dokumen
    print(f"\n[1] Melakukan Indeksasi {len(docs)} Dokumen ke Ruang Vektor...")
    doc_vectors = np.array([encoder.encode(d["title"] + " " + d["text"]) for d in docs])
    # Pastikan normalisasi L2
    doc_vectors = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    print(f"    Basis Data Vektor Terbentuk: {doc_vectors.shape} (Vektor Ternormalisasi L2)")
    
    # 2. Uji Beberapa Kueri Pencarian
    test_queries = [
        "Bagaimana cara kerja pencarian kesamaan semantik menggunakan vektor?",
        "Protokol komunikasi nirkabel pengiriman data sensor industri",
        "Deteksi kerusakan dini pada mesin turbin pabrik"
    ]
    
    print("\n[2] Menjalankan Pencarian Kueri:")
    k = 3
    
    for q_idx, q_text in enumerate(test_queries, 1):
        print(f"\n" + "=" * 80)
        print(f"KUERI #{q_idx}: \"{q_text}\"")
        print("=" * 80)
        
        # Vektorisasi kueri
        q_vec = encoder.encode(q_text)
        q_vec = q_vec / np.linalg.norm(q_vec)
        
        # Hitung skor kesamaan kosinus via dot product pada vektor ternormalisasi
        scores = doc_vectors @ q_vec
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        for rank, idx in enumerate(top_k_indices, 1):
            doc = docs[idx]
            score = scores[idx]
            print(f"Peringkat {rank} [Skor: {score:.4f}] - {doc['id']}: {doc['title']} ({doc['category']})")
            print(f"   Kutipan Teks: \"{doc['text']}\"")
            print(f"   Kata Kunci  : {', '.join(doc['tags'])}")
            
    print("\n" + "=" * 80)
    print("Kesimpulan: Pencarian semantik berhasil mengidentifikasi dokumen yang relevan")
    print("berdasarkan kesamaan makna konseptual di ruang laten vektor.")

if __name__ == "__main__":
    main()

