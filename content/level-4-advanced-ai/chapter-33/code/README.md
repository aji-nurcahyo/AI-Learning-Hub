# Chapter 33: Retrieval-Augmented Generation — Code Catalog

Koleksi script Python mandiri, deterministik, dan dapat dijalankan secara offline 100% pada CPU untuk mempelajari arsitektur Retrieval-Augmented Generation (RAG).

## Daftar Script

1. `document_ingestion.py`: Memuat dokumen dari JSON, ekstraksi metadata, pencatatan rekam jejak sumber (*source provenance*), deduplikasi berbasis hash, dan sanitasi teks.
2. `document_chunking.py`: Strategi chunking teks (Fixed-Size, Recursive, Sentence-based) dengan pelestarian metadata segmen (`chunk_id`, rentang karakter, judul dokumen induk).
3. `chunk_overlap_analysis.py`: Eksperimen trade-off ukuran chunk vs overlap terhadap redundansi teks dan mitigasi *boundary clipping*.
4. `rag_vector_index.py`: Indeks vektor dalam memori berbasis matriks NumPy yang kompatibel dengan FAISS `IndexFlatIP` melalui normalisasi L2 untuk Cosine Similarity.
5. `sparse_retrieval.py`: Retrieval leksikal BM25 Okapi dengan *inverted index* dan penghitungan IDF standar.
6. `dense_retrieval.py`: Retrieval semantik berbasis vektor padat dilengkapi ambang batas kemiripan (*similarity threshold*) dan filter metadata kategori.
7. `hybrid_retrieval.py`: Penggabungan hasil retrieval leksikal (BM25) dan semantik (vektor) menggunakan *Reciprocal Rank Fusion* (RRF).
8. `reranking_demo.py`: Cross-scoring reranker heuristik untuk menilai ulang kandidat teratas menggunakan kecocokan frasa eksak, penyelarasan judul, dan kerapatan kata kunci.
9. `context_builder.py`: Pembentuk konteks prompt terstruktur dengan pembatasan anggaran token/karakter (*context budget*), isolasi delimiter, dan pemetaan nomor sitasi.
10. `grounded_answer_demo.py`: Generator jawaban berbasis bukti deterministik offline dengan deteksi abstensi jika konteks tidak mencukupi (bukan inferensi generatif LLM).
11. `citation_validation.py`: Verifikasi otomatis keabsahan sitasi `[Doc: <id>]`, deteksi sitasi halusinasi, dan pemeriksaan kalimat tanpa bukti pendukung.
12. `rag_evaluation.py`: Pengujian terpisah untuk evaluasi retrieval (Precision@k, Recall@k, Hit Rate@k, MRR) dan evaluasi generatif (Groundedness, Faithfulness, Citation Correctness).
13. `rag_pipeline.py`: Arsitektur pipeline RAG modular menyeluruh yang mengintegrasikan seluruh tahapan dari ingest hingga audit sitasi.
14. `rag_security_checks.py`: Pertahanan keamanan lapis jamak (*defense-in-depth*), pemindaian muatan *indirect prompt injection*, dan *Access Control List* (ACL) level dokumen.

## Cara Menjalankan

Seluruh script dapat dijalankan secara individual menggunakan virtual environment proyek:

```bash
# Jalankan pipeline utuh
python content/level-4-advanced-ai/chapter-33/code/rag_pipeline.py

# Jalankan evaluasi benchmark
python content/level-4-advanced-ai/chapter-33/code/rag_evaluation.py

# Jalankan pengujian keamanan
python content/level-4-advanced-ai/chapter-33/code/rag_security_checks.py
```

