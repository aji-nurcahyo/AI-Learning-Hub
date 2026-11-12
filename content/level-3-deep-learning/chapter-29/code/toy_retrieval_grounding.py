"""
Chapter 29 - Large Language Models & Generative AI
Script 11: Retrieval-Augmented Generation (RAG) & Grounding Concept
Mendemonstrasikan pipeline RAG sederhana: Temu-kembali dokumen rujukan (Retrieval),
penyusunan prompt bersumber (Context Augmentation), dan pembangkitan berdasar fakta (Grounded Generation).
"""

import math
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ToyKnowledgeBase:
    """Basis pengetahuan dokumen internal mini berlisensi domain akademik ITERA."""
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.corpus = [doc["content"] for doc in documents]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[Dict[str, str], float]]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        ranked_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in ranked_indices:
            results.append((self.documents[idx], float(scores[idx])))
        return results


def construct_rag_prompt(query: str, retrieved_doc: Dict[str, str]) -> str:
    """Menyusun prompt berakar konteks (grounded prompt)."""
    return f"""Anda adalah asisten AI yang jujur. Jawablah pertanyaan pengguna HANYA berdasarkan konteks dokumen rujukan yang disediakan berikut ini. Jika informasi tidak terdapat dalam dokumen, jawab secara tegas bahwa informasi tidak ditemukan.

[DOKUMEN RUJUKAN: ID {retrieved_doc['id']}]
Sumber: {retrieved_doc['source']}
Konten: "{retrieved_doc['content']}"

[PERTANYAAN PENGGUNA]
"{query}"

[JAWABAN BERDASARKAN DOKUMEN DENGAN ATRIBUSI SUMBER]"""


def main():
    print("=" * 70)
    print("CHAPTER 29: RETRIEVAL-AUGMENTED GENERATION (RAG) & GROUNDING")
    print("=" * 70)

    # Basis pengetahuan mini
    docs = [
        {
            "id": "DOC-001",
            "source": "Buku Panduan Akademik ITERA 2024",
            "content": "Institut Teknologi Sumatera (ITERA) didirikan pada tanggal 6 Oktober 2014 berdasarkan Peraturan Presiden Republik Indonesia Nomor 124 Tahun 2014."
        },
        {
            "id": "DOC-002",
            "source": "Katalog Laboratorium AI ITERA",
            "content": "Laboratorium Kecerdasan Buatan ITERA menyelenggarakan praktikum deep learning, visi komputer, dan pemrosesan bahasa alami menggunakan workstation komputasi GPU dan klaster CPU."
        },
        {
            "id": "DOC-003",
            "source": "Pedoman Perpustakaan ITERA",
            "content": "Perpustakaan pusat ITERA menyediakan akses ribuan jurnal internasional terindeks Scopus dan IEEE Xplore untuk seluruh sivitas akademika."
        }
    ]

    kb = ToyKnowledgeBase(docs)
    print(f"Total Dokumen dalam Basis Pengetahuan: {len(docs)}")

    query = "Kapan ITERA didirikan dan apa dasar hukum pendiriannya?"
    print(f"\nPertanyaan Pengguna: '{query}'")

    # 1. Tahap Retrieval (Temu-Kembali Bukti Fakta)
    results = kb.retrieve(query, top_k=1)
    best_doc, sim_score = results[0]

    print("\n[1. HASIL RETRIEVAL DOKUMEN RELEVAN]")
    print(f"Dokumen Terpilih : {best_doc['id']} (Skor Kemiripan Kosinus TF-IDF: {sim_score:.4f})")
    print(f"Sumber Rujukan   : {best_doc['source']}")
    print(f"Kutipan Dokumen  : \"{best_doc['content']}\"")

    # 2. Tahap Augmentation (Penyusunan Grounded Prompt)
    rag_prompt = construct_rag_prompt(query, best_doc)
    print("\n[2. GROUNDED PROMPT DENGAN BUKTI KONTEKS]")
    print("-" * 65)
    print(rag_prompt)
    print("-" * 65)

    # 3. Simulasi Pembangkitan: Grounded vs Ungrounded (Halusinasi)
    print("\n[3. PERBANDINGAN OUTPUT MODEL: GROUNDED VS UNGROUNDED]")
    print("\nA. Generasi Tanpa RAG (Ungrounded - Rawan Halusinasi Parametrik):")
    print("   'ITERA kemungkinan didirikan sekitar tahun 2010 atau 2012 oleh pemerintah daerah Lampung.'")
    print("   -> Status: HALUSINASI FAKTUAL (Tahun dan dasar hukum salah).")

    print("\nB. Generasi Berbasis RAG (Grounded - Faktual & Teratribusi):")
    print(f"   'Berdasarkan {best_doc['source']} ({best_doc['id']}), Institut Teknologi Sumatera didirikan pada tanggal 6 Oktober 2014 sesuai Perpres RI Nomor 124 Tahun 2014.'")
    print("   -> Status: FAKTUAL LENGKAP dengan atribusi sumber yang dapat diaudit manusia.")

    print("\n[RANGKUMAN TEORETIS]")
    print("1. RAG memisahkan memori pengetahuan (basis data eksternal) dari mesin penalaran (LLM).")
    print("2. Grounding membatasi ruang hipotesis model sehingga hanya bersandar pada dokumen yang disajikan.")
    print("3. Batasan RAG: Jika tahap retrieval gagal menemukan dokumen relevan, model tetap dapat menghasilkan jawaban yang salah (Retrieval Failure).")


if __name__ == "__main__":
    main()

