"""
text_tfidf_baseline.py
Sistem Temu Kembali Leksikal Berbasis TF-IDF (Sparse Retrieval Baseline):
- Memuat korpus dokumen akademik dari data/ai_telecom_corpus.json
- Membangun matriks jarang (Sparse Matrix) menggunakan TfidfVectorizer Scikit-learn
- Menghitung kecocokan leksikal kueri terhadap seluruh dokumen
- Mengukur batas kemampuan pencarian berbasis kata kunci murni (Exact Match)
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_corpus():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(base_dir, "data", "ai_telecom_corpus.json")
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"], data["benchmark_queries"]

def main():
    print("=== Sparse Retrieval Baseline: TF-IDF Search Engine ===")
    docs, queries = load_corpus()
    print(f"Total Dokumen dalam Korpus: {len(docs)}")
    print(f"Total Kueri Uji Penilaian : {len(queries)}\n")
    
    # 1. Bangun Vektor TF-IDF dari Teks Dokumen
    doc_texts = [d["text"] for d in docs]
    doc_ids = [d["id"] for d in docs]
    
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(doc_texts)
    
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"[1] Matriks TF-IDF Dibangun:")
    print(f"    Bentuk Matriks : {tfidf_matrix.shape} ({len(docs)} dokumen x {vocab_size} kata unik)")
    print(f"    Kepadatan Data : {(tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}% (Sparse)")
    
    # 2. Uji Pencarian pada Kueri Benchmark
    print("\n[2] Menjalankan Pencarian Kueri Benchmark (Top-3):")
    k = 3
    
    for q in queries[:3]:
        q_text = q["query_text"]
        relevant_ids = set(q["relevant_doc_ids"])
        
        # Vektorisasi kueri
        q_vec = vectorizer.transform([q_text])
        # Hitung kesamaan kosinus terhadap seluruh dokumen
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        # Ambil top-k indeks tertinggi
        top_indices = np.argsort(scores)[::-1][:k]
        
        print(f"\nKueri: \"{q_text}\"")
        print(f"Target Dokumen Relevan: {list(relevant_ids)}")
        print(f"{'Peringkat':<10} {'Doc ID':<10} {'Skor TF-IDF':<14} {'Status Relevan':<15} {'Judul Dokumen'}")
        print("-" * 75)
        
        hits = 0
        for rank, idx in enumerate(top_indices, 1):
            d_id = doc_ids[idx]
            d_title = docs[idx]["title"]
            score = scores[idx]
            is_rel = d_id in relevant_ids
            if is_rel:
                hits += 1
            rel_str = "RELEVAN" if is_rel else "TIDAK"
            print(f"Top-{rank:<6} {d_id:<10} {score:<14.4f} {rel_str:<15} {d_title}")
            
    print("\nKelemahan Pencarian Leksikal (Vocabulary Mismatch):")
    print("Jika kueri menggunakan sinonim atau parafrasa yang kata persisnya tidak muncul di dokumen,")
    print("skor TF-IDF akan bernilai 0.0000 meskipun secara makna semantik dokumen tersebut sangat relevan.")

if __name__ == "__main__":
    main()

