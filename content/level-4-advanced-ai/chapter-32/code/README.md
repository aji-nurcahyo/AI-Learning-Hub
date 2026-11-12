# Panduan Berkas Kode Chapter 32: Embeddings & Vector Search

Katalog ini berisi 14 skrip Python mandiri yang mendemonstrasikan operasi aljabar vektor, representasi teks ke vektor (*Embeddings*), normalisasi L2, pencarian tetangga terdekat eksak (*Exact Search*), arsitektur indeks FAISS dan aproksimasi (*ANN*), serta evaluasi sistem temu kembali (*Information Retrieval*). Seluruh skrip dapat dieksekusi secara instan pada CPU komputer standar tanpa ketergantungan kartu grafis (GPU).

---

## 1. Daftar dan Deskripsi Skrip

| No | Nama Berkas | Topik Utama | Kompleksitas / Dependensi |
|---|---|---|:---:|
| 1 | `embedding_basics.py` | Perbedaan representasi jarang (Sparse One-Hot) vs representasi padat (Dense Embeddings) | NumPy, CPU |
| 2 | `vector_similarity.py` | Operasi aljabar: Dot Product, Euclidean Distance, Cosine Similarity, dan Cosine Distance | NumPy, CPU |
| 3 | `cosine_similarity.py` | Stabilitas numerik, penanganan eksplisit vektor nol, dan penjepitan interval [-1.0, 1.0] | NumPy, CPU |
| 4 | `vector_normalization.py` | Normalisasi L2, unit vectors, dan pembuktian Dot Product = Cosine Similarity | NumPy, CPU |
| 5 | `text_tfidf_baseline.py` | Sistem temu kembali leksikal TF-IDF sebagai baseline pencarian kata kunci | Scikit-learn, CPU |
| 6 | `sentence_embedding_demo.py` | Pembangkitan representasi kalimat deterministik mandiri tanpa model unduhan raksasa | NumPy, CPU |
| 7 | `exact_vector_search.py` | Algoritma pencarian tetangga terdekat eksak $\mathcal{O}(N \cdot d)$ berbasis jarak L2 dan Cosine | NumPy, CPU |
| 8 | `nearest_neighbor_numpy.py` | Optimasi pencarian batch tervektorisasi dengan `np.argpartition` dan kalibrasi skor | NumPy, CPU |
| 9 | `faiss_flat_index.py` | Pengenalan struktur indeks FAISS `IndexFlatL2` dan `IndexFlatIP` dengan fallback NumPy | FAISS / NumPy, CPU |
| 10 | `faiss_ann_demo.py` | Konsep Inverted File Index (IVF), sel Voronoi, serta trade-off $n_{\text{probe}}$ vs Recall | NumPy / FAISS, CPU |
| 11 | `semantic_search.py` | Pipa pencarian semantik end-to-end atas korpus dokumen akademik di `data/` | NumPy, CPU |
| 12 | `retrieval_metrics.py` | Formulasi dan perhitungan numerik Precision@k, Recall@k, Hit Rate@k, dan MRR | NumPy, CPU |
| 13 | `sparse_dense_comparison.py` | Komparasi empiris performa TF-IDF vs Dense Retrieval pada kueri benchmark | Scikit-learn, NumPy |
| 14 | `vector_search_benchmark.py` | Tolok ukur skalabilitas latensi per kueri dan konsumsi memori ($N = 100 \to 10.000$) | NumPy, CPU |

---

## 2. Instruksi Eksekusi

Jalankan seluruh skrip secara individual melalui terminal menggunakan Python 3.10+:

```bash
# Contoh menjalankan pencarian semantik end-to-end
python content/level-4-advanced-ai/chapter-32/code/semantic_search.py

# Contoh menjalankan komparasi sparse vs dense
python content/level-4-advanced-ai/chapter-32/code/sparse_dense_comparison.py
```

Seluruh skrip menggunakan seed acak deterministik (`np.random.seed(42)`) untuk menjamin reproduksibilitas hasil numerik 100%.

