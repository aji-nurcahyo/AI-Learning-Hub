# Panduan Praktik Laboratorium: Embeddings & Vector Search

Selamat datang di panduan laboratorium **Chapter 32: Embeddings & Vector Search**. Modul ini memandu Anda mempraktikkan secara langsung matematika representasi vektor (**Dense Embeddings**), metrik kesamaan aljabar (**Cosine Similarity, Dot Product, Euclidean Distance**), normalisasi L2, pencarian tetangga terdekat (**Exact Search & Approximate Nearest Neighbor/ANN**), arsitektur indeks FAISS, serta evaluasi sistem temu kembali (**Information Retrieval Evaluation**) berbasis **Python**, **NumPy**, dan **Scikit-learn** pada jalur komputasi CPU.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- **Python**: Versi 3.10+ (Teruji pada Python 3.12.10).
- **Pustaka Wajib**: `numpy>=2.0.0` (Teruji 2.5.4), `scikit-learn>=1.3.0` (Teruji 1.9.1).
- **Pustaka Opsional**: `faiss-cpu` (Jika tidak terpasang, sistem secara otomatis beralih ke *Pure NumPy FAISS-Compatible Engine* tanpa memutus alur eksekusi).
- **Akselerator Perangkat Keras**: Bersifat opsional. Seluruh 14 berkas skrip praktik dirancang khusus agar dapat dieksekusi secara instan (< 15 detik untuk 14 skrip) pada CPU komputer standar tanpa ketergantungan kartu grafis (GPU), tanpa API berbayar, dan tanpa model unduhan raksasa.

### Struktur Direktori Bab 32:
```
content/level-4-advanced-ai/chapter-32/
├── code/
│   ├── README.md
│   ├── embedding_basics.py
│   ├── vector_similarity.py
│   ├── cosine_similarity.py
│   ├── vector_normalization.py
│   ├── text_tfidf_baseline.py
│   ├── sentence_embedding_demo.py
│   ├── exact_vector_search.py
│   ├── nearest_neighbor_numpy.py
│   ├── faiss_flat_index.py
│   ├── faiss_ann_demo.py
│   ├── semantic_search.py
│   ├── retrieval_metrics.py
│   ├── sparse_dense_comparison.py
│   └── vector_search_benchmark.py
├── data/
│   └── ai_telecom_corpus.json
├── experiments/
│   └── results_ch32.json
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_32_companion.ipynb
├── slides/
│   ├── build_slides.py
│   └── deck.json
├── index.md
├── practice.md
├── metadata.json
└── audit.md
```

---

## 2. Empat Belas Tahapan Praktik Terpadu

### Langkah 1: Eksplorasi Representasi Sparse vs Dense
Jalankan skrip dasar representasi vektor kata:
```bash
python content/level-4-advanced-ai/chapter-32/code/embedding_basics.py
```
**Fokus Pengamatan**:
- Amati bagaimana representasi One-Hot menghasilkan vektor ortogonal murni (dot product = 0) meskipun dua kata memiliki kedekatan semantik kuat (*jaringan* dan *syaraf*).
- Bandingkan dengan representasi dense di mana kedekatan makna tercermin dalam skor kesamaan kosinus positif tinggi.

### Langkah 2: Operasi Aljabar Metrik Kesamaan
Jalankan skrip eksplorasi metrik aljabar:
```bash
python content/level-4-advanced-ai/chapter-32/code/vector_similarity.py
```
**Fokus Pengamatan**:
- Bandingkan perilaku Dot Product, Euclidean Distance, dan Cosine Similarity pada vektor kolinear, ortogonal, dan berlawanan arah.
- Perhatikan bahwa pada vektor kolinear dengan magnitudo berbeda, Cosine Similarity bernilai tepat 1.0000, membuktikan kekebalannya terhadap skala panjang vektor.

### Langkah 3: Stabilitas Numerik & Penanganan Vektor Nol
Jalankan skrip pengujian edge-cases:
```bash
python content/level-4-advanced-ai/chapter-32/code/cosine_similarity.py
```
**Fokus Pengamatan**:
- Pelajari mekanisme proteksi `ZeroDivisionError` saat menjumpai kueri atau dokumen kosong (vektor nol).
- Pahami fungsi penjepitan (*clamping*) ke rentang $[-1.0, 1.0]$ untuk mencegah anomali floating-point.

### Langkah 4: Pembuktian Teorema Normalisasi L2
Jalankan skrip verifikasi kesetaraan aljabar:
```bash
python content/level-4-advanced-ai/chapter-32/code/vector_normalization.py
```
**Fokus Pengamatan**:
- Buktikan bahwa Dot Product pada vektor satuan ($\|\hat{\mathbf{u}}\|=1, \|\hat{\mathbf{v}}\|=1$) identik persis dengan Cosine Similarity murni ($\hat{\mathbf{u}} \cdot \hat{\mathbf{v}} = \cos(\theta)$).
- Buktikan bahwa kuadrat jarak Euclidean berbanding lurus langsung dengan Cosine Distance ($\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|^2 = 2(1 - \cos(\theta))$).

### Langkah 5: Baseline Temu Kembali Leksikal TF-IDF
Jalankan skrip pencarian leksikal berbasis kata kunci:
```bash
python content/level-4-advanced-ai/chapter-32/code/text_tfidf_baseline.py
```
**Fokus Pengamatan**:
- Amati konstruksi matriks TF-IDF jarang (*sparse matrix*) atas korpus 20 dokumen akademik.
- Identifikasi masalah *vocabulary mismatch*: kueri gagal menemukan dokumen relevan jika kata kunci persis tidak muncul dalam teks.

### Langkah 6: Pembangkitan Dense Sentence Embeddings Mandiri
Jalankan skrip ekstraksi representasi kalimat deterministik:
```bash
python content/level-4-advanced-ai/chapter-32/code/sentence_embedding_demo.py
```
**Fokus Pengamatan**:
- Pelajari teknik ekstraksi representasi subword n-gram dan proyeksi ortogonal berdimensi 32 pada CPU tanpa model unduhan raksasa.
- Periksa matriks korelasi semantik antar kalimat dari berbagai kategori topik.

### Langkah 7: Algoritma Pencarian Tetangga Terdekat Eksak
Jalankan skrip pencarian brute-force di NumPy:
```bash
python content/level-4-advanced-ai/chapter-32/code/exact_vector_search.py
```
**Fokus Pengamatan**:
- Amati kompleksitas waktu $\mathcal{O}(N \cdot d)$ per kueri.
- Bandingkan perankingan jarak terkecil (L2) vs kesamaan tertinggi (Cosine).

### Langkah 8: Optimalisasi Pencarian Batch Tervektorisasi
Jalankan skrip akselerasi `argpartition`:
```bash
python content/level-4-advanced-ai/chapter-32/code/nearest_neighbor_numpy.py
```
**Fokus Pengamatan**:
- Pahami mengapa `np.argpartition` memangkas kompleksitas perankingan dari $\mathcal{O}(N \log N)$ menjadi $\mathcal{O}(N)$ pada basis data berukuran besar.
- Amati kalibrasi skor kesamaan ke rentang probabilitas terkalibrasi $[0\%, 100\%]$.

### Langkah 9: Indeks Vektor Datar FAISS (IndexFlatL2 & IndexFlatIP)
Jalankan skrip struktur indeks FAISS standar:
```bash
python content/level-4-advanced-ai/chapter-32/code/faiss_flat_index.py
```
**Fokus Pengamatan**:
- Pahami pemisahan arsitektural: FAISS hanya menyimpan ID numerik dan vektor, sementara teks dan judul disimpan di metadata store terpisah (*Hydration*).
- Amati mekanisme fallback otomatis ke NumPy jika FAISS C++ backend tidak tersedia.

### Langkah 10: Approximate Nearest Neighbor (ANN) dengan Inverted File Index
Jalankan simulasi partisi sel Voronoi:
```bash
python content/level-4-advanced-ai/chapter-32/code/faiss_ann_demo.py
```
**Fokus Pengamatan**:
- Analisis pengaruh hiperparameter $n_{\text{probe}}$ terhadap Recall dan persentase data yang dipindai.
- Buktikan bahwa dengan memindai hanya 25% data ($n_{\text{probe}}=5$ dari 20 sel), sistem dapat menangkap mayoritas dokumen relevan dengan pemangkasan waktu komputasi yang signifikan.

### Langkah 11: Pipa Pencarian Semantik End-to-End
Jalankan pipeline pencarian semantik atas korpus akademik nyata:
```bash
python content/level-4-advanced-ai/chapter-32/code/semantic_search.py
```
**Fokus Pengamatan**:
- Amati alur utuh: Kueri Teks $\to$ Vektor $\to$ Normalisasi L2 $\to$ Dot Product Index $\to$ Top-3 Reconstructed Documents.

### Langkah 12: Kalkulasi Metrik Evaluasi Temu Kembali
Jalankan skrip formulasi metrik IR:
```bash
python content/level-4-advanced-ai/chapter-32/code/retrieval_metrics.py
```
**Fokus Pengamatan**:
- Pelajari cara menghitung Precision@k, Recall@k, Hit Rate@k, dan Mean Reciprocal Rank (MRR) secara analitis dan numerik.

### Langkah 13: Komparasi Empiris Sparse vs Dense Retrieval
Jalankan evaluasi komparatif kedua paradigma pada data benchmark:
```bash
python content/level-4-advanced-ai/chapter-32/code/sparse_dense_comparison.py
```
**Fokus Pengamatan**:
- Analisis tabel metrik komparasi (@k=3) antara TF-IDF dan Dense Retrieval.
- Pahami mengapa arsitektur industri mengadopsi *Hybrid Retrieval* untuk memadukan kekuatan kedua metode.

### Langkah 14: Tolok Ukur Skalabilitas Waktu & Memori
Jalankan skrip profiling latensi:
```bash
python content/level-4-advanced-ai/chapter-32/code/vector_search_benchmark.py
```
**Fokus Pengamatan**:
- Amati pertumbuhan waktu komputasi linier $\mathcal{O}(N)$ dari $N=100$ hingga $N=10.000$ pada CPU.
- Catat konsumsi memori vektor mentah dalam satuan megabyte (MB).

---

## 3. Latihan Bertingkat (3-Tier Exercises)

### Latihan Tingkat 1: Pemahaman Konseptual & Aljabar
1. **Soal 1.1**: Diberikan dua vektor representasi dokumen:
   $$\mathbf{u} = [1.0, \, 1.0, \, 0.0], \quad \mathbf{v} = [0.0, \, 2.0, \, 2.0]$$
   a. Hitung Dot Product $\mathbf{u} \cdot \mathbf{v}$.
   b. Hitung panjang norma Euclidean $\|\mathbf{u}\|_2$ dan $\|\mathbf{v}\|_2$.
   c. Hitung Cosine Similarity $\cos(\theta)$ dan Cosine Distance $d_{\text{cos}}$.
2. **Soal 1.2**: Mengapa pencarian menggunakan `IndexFlatIP` pada pustaka FAISS hanya menghasilkan Cosine Similarity yang valid jika seluruh vektor telah dinormalisasi L2 terlebih dahulu?

*Kunci Jawaban*:
- **1.1**:
  a. $\mathbf{u} \cdot \mathbf{v} = (1.0 \times 0.0) + (1.0 \times 2.0) + (0.0 \times 2.0) = 2.0$.
  b. $\|\mathbf{u}\|_2 = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2} \approx 1.4142$. $\|\mathbf{v}\|_2 = \sqrt{0^2 + 2^2 + 2^2} = \sqrt{8} \approx 2.8284$.
  c. $\cos(\theta) = \frac{2.0}{\sqrt{2} \times \sqrt{8}} = \frac{2.0}{\sqrt{16}} = \frac{2.0}{4.0} = 0.5000$.  
     Cosine Distance $d_{\text{cos}} = 1.0 - 0.5000 = 0.5000$.
- **1.2**: `IndexFlatIP` mengeksekusi operasi Inner Product (Dot Product) murni $\mathbf{u} \cdot \mathbf{v} = \sum u_i v_i$. Dari definisi $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$, jika $\|\mathbf{u}\|_2 = 1.0$ dan $\|\mathbf{v}\|_2 = 1.0$, maka penyebut bernilai tepat $1.0$, sehingga $\mathbf{u} \cdot \mathbf{v} \equiv \cos(\theta)$. Tanpa normalisasi, hasil `IndexFlatIP` akan terdistorsi oleh panjang magnitudo vektor.

### Latihan Tingkat 2: Implementasi Pemrograman NumPy
**Soal 2.1**: Tuliskan fungsi Python `compute_mrr(retrieved_ids_batch, relevant_ids_batch)` yang menerima list of list dokumen terambil dan dokumen relevan, lalu mengembalikan nilai Mean Reciprocal Rank (MRR) dalam format float.

*Kunci Jawaban*:
```python
def compute_mrr(retrieved_ids_batch, relevant_ids_batch):
    rr_list = []
    for retrieved, relevant in zip(retrieved_ids_batch, relevant_ids_batch):
        rel_set = set(relevant)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in rel_set:
                rr = 1.0 / rank
                break
        rr_list.append(rr)
    return float(np.mean(rr_list)) if rr_list else 0.0
```

### Latihan Tingkat 3: Analisis Rekayasa & Keamanan
**Soal 3.1**: Sebuah startup layanan hukum ingin membangun mesin pencari berkas perkara pengadilan berbasis pencarian vektor.
a. Diskusikan risiko keamanan dan privasi terkait *Embedding Inversion Attack*.
b. Rancang arsitektur pencarian hibrida (*Hybrid Search*) yang memadukan keunggulan leksikal (nomor pasal / undang-undang) dan semantik (analisis kasus serupa).

*Kunci Jawaban*:
a. **Risiko Privasi & Inversion Attack**:
   - Vektor embeddings bukan bentuk anonimisasi atau enkripsi satu arah. Penyerang dengan model de-vektorisasi (*Vec2Text*) dapat merekonstruksi nama terdakwa, nomor identitas, atau isi klausa rahasia dari koordinat vektor.
   - **Mitigasi**: Lakukan sanitasi data pribadi (PII redaction) sebelum teks diproses oleh model embedding, terapkan enkripsi pada penyimpanan basis data vektor, dan batasi kontrol akses (*role-based access control*).
b. **Arsitektur Hybrid Search**:
   - **Tahap 1 (Dual-Retrieval)**: Jalankan dua indeks paralel untuk setiap kueri pengguna:
     1. Indeks Leksikal (BM25 / TF-IDF) untuk menangkap nomor undang-undang dan pasal persis (*exact matches*).
     2. Indeks Vektor Dense (FAISS) untuk menangkap konteks kasus dan analogi preseden hukum (*semantic matches*).
   - **Tahap 2 (Reciprocal Rank Fusion - RRF)**: Gabungkan skor dari kedua jalur menggunakan formula:
     $$\text{RRF Score}(d) = \sum_{m \in \{\text{lexical}, \text{dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$
   - Hasil akhir mengurutkan berkas yang memiliki kecocokan pasal persis sekaligus konteks kasus yang paling dekat.

---

## 4. Mini Project: Building a Semantic Search System for AI Learning Materials

### 4.1 Sasaran Proyek Mini
Mahasiswa diminta membangun sistem temu kembali informasi semantik mandiri menggunakan korpus materi pembelajaran kecerdasan buatan dan telekomunikasi (`data/ai_telecom_corpus.json`):
1. Memuat dan memvalidasi struktur korpus 20 dokumen akademik.
2. Membangun representasi teks ke vektor padat (*Dense Embeddings*).
3. Menerapkan normalisasi L2 dan membangun struktur indeks pencarian tetangga terdekat (*Exact Search* atau *FAISS-Compatible Flat Index*).
4. Menjalankan evaluasi kueri benchmark dan menghitung metrik kuantitatif: Precision@3, Recall@3, Hit Rate@3, dan Mean Reciprocal Rank (MRR).
5. Membandingkan performa terhadap baseline leksikal TF-IDF.
6. Menyusun analisis kesalahan (*error analysis*) pada kueri yang gagal menempati peringkat 1, serta menyertakan dokumentasi privasi data.

---

## 5. Rubrik Penilaian Proyek Mini (Skala 100 Poin)

| Dimensi Penilaian | Bobot | Kriteria Sempurna (Full Credit) | Kriteria Cukup (Partial Credit) | Kriteria Kurang (Insufficient) |
|---|:---:|---|---|---|
| **1. Pemahaman Masalah & Korpus Data** | 15 Poin | Korpus data dipahami mendalam; skema dokumen dan kueri uji terpetakan rapi; batasan temu kembali dijelaskan. | Korpus dimuat namun pemahaman skema metadata kueri kurang lengkap. | Data tidak dimuat; deskripsi korpus kabur atau tidak terstruktur. |
| **2. Representasi Vektor & Normalisasi L2** | 25 Poin | Ekstraksi embedding padat berhasil; normalisasi L2 diterapkan benar; pembuktian kesetaraan aljabar terverifikasi. | Embedding terbentuk namun normalisasi L2 diabaikan atau menimbulkan zero division. | Ekstraksi vektor gagal; perhitungan aljabar salah; kode error. |
| **3. Implementasi Indeks & Pencarian** | 25 Poin | Pipa pencarian Top-k end-to-end berfungsi sempurna; rekonstruksi metadata dokumen bersih; penanganan kueri stabil. | Pencarian berfungsi namun metadata dokumen tidak terekonstruksi dengan benar (*missing hydration*). | Indeks pencarian gagal dibangun; output tidak relevan. |
| **4. Evaluasi Temu Kembali (IR Metrics)** | 25 Poin | Menghitung Precision@k, Recall@k, Hit Rate@k, dan MRR secara presisi; tabel komparasi sparse vs dense tersedia mendalam. | Metrik evaluasi dihitung namun terbatas pada salah satu indikator tanpa analisis perbandingan. | Tidak ada perhitungan metrik kuantitatif; klaim subjektif murni. |
| **5. Dokumentasi, Privasi & Referensi** | 10 Poin | Kode modular dan terdokumentasi; analisis risiko privasi embeddings dibahas; menyertakan $\ge 2$ referensi akademik terverifikasi. | Kode dapat dijalankan namun dokumentasi minim; referensi kurang lengkap. | Tidak ada dokumentasi; referensi fiktif atau tidak relevan. |
| **Total Skor Maksimal** | **100 Poin** | **Standar Rekayasa Temu Kembali Informasi Modern** | — | — |

