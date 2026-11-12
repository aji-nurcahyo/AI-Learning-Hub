# Panduan Praktik Laboratorium: Retrieval-Augmented Generation (RAG)
**Artificial Intelligence Learning Hub — Level 4 (Chapter 33)**  
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## 1. DESKRIPSI DAN PRASYARAT PRAKTIKUM

Praktikum ini dirancang untuk memberikan pemahaman operasional dan keterampilan implementasi nyata dalam membangun arsitektur **Retrieval-Augmented Generation (RAG)** modular yang efisien, dapat diuji secara objektif, serta beroperasi 100% mandiri (*offline*) pada CPU tanpa ketergantungan pada API berbayar atau unduhan model berbobot raksasa.

### Prasyarat Pengetahuan
- Pemahaman aljabar linier dan representasi vektor berdimensi tinggi (Chapter 32).
- Pemahaman mekanisme atensi pada Transformer dan arsitektur model bahasa (Chapter 28 & 29).
- Pemrograman Python tingkat menengah: manipulasi struktur data dictionary, pemrosesan teks ekspresi reguler (regex), dan komputasi array NumPy.

### Lingkungan Eksekusi
- **Bahasa:** Python 3.10+ (Direkomendasikan Python 3.12).
- **Pustaka Utama:** `numpy`, `scikit-learn` (opsional), pustaka standar Python (`hashlib`, `re`, `json`, `math`, `collections`).
- **Akselerasi:** Berjalan penuh pada CPU standar (RAM $\ge$ 4 GB).

---

## 2. PANDUAN LANGKAH DEMI LANGKAH EKSEKUSI

Praktikum ini tersusun atas 14 script Python mandiri di direktori `content/level-4-advanced-ai/chapter-33/code/`:

### Langkah 1: Ingestion Dokumen & Pelacakan Provenance
```bash
python content/level-4-advanced-ai/chapter-33/code/document_ingestion.py
```
- **Tujuan:** Memuat berkas korpus `data/ai_course_corpus.json`, mengekstraksi metadata, mencatat *source provenance*, membersihkan karakter kontrol non-printable, dan mendeteksi duplikat menggunakan *cryptographic hashing* SHA-256.
- **Hasil Verifikasi:** 20 dokumen akademik dimuat dan divalidasi dengan atribut identitas unik (`doc_id`).

### Langkah 2: Pemotongan Teks (Chunking) & Pelestarian Metadata
```bash
python content/level-4-advanced-ai/chapter-33/code/document_chunking.py
```
- **Tujuan:** Membandingkan teknik pemotongan *Fixed-Size* dan *Recursive Hierarchical*.
- **Hasil Verifikasi:** Teks dipecah menjadi unit-unit semantik mandiri dengan metadata rentang karakter (`start_char`, `end_char`) dan judul dokumen induk tetap terjaga.

### Langkah 3: Analisis Trade-off Chunk Overlap & Redundansi
```bash
python content/level-4-advanced-ai/chapter-33/code/chunk_overlap_analysis.py
```
- **Tujuan:** Mengukur secara kuantitatif dampak rasio overlap ($0\%$, $10\%$, $25\%$, $50\%$) terhadap faktor redundansi penyimpanan dan risiko dilusi informasi pada *context window*.
- **Hasil Verifikasi:** Mengonfirmasi bahwa overlap $10\% - 25\%$ merupakan rentang optimal untuk mempertahankan kontinuitas antarsegmen tanpa redundansi berlebih.

### Langkah 4: Pembuatan Indeks Vektor Berbasis L2 Normalization
```bash
python content/level-4-advanced-ai/chapter-33/code/rag_vector_index.py
```
- **Tujuan:** Mengonversi teks chunk ke representasi vektor padat ($d=64$) dan mengindeksnya menggunakan matriks NumPy yang kompatibel dengan FAISS `IndexFlatIP`.
- **Hasil Verifikasi:** Pencarian inner product menghasilkan skor kemiripan kosinus persis berkat normalisasi L2 ($\|\hat{\mathbf{v}}\|_2 = 1.0$).

### Langkah 5: Retrieval Leksikal BM25 (Sparse)
```bash
python content/level-4-advanced-ai/chapter-33/code/sparse_retrieval.py
```
- **Tujuan:** Membangun *inverted index* dan menghitung skor BM25 Okapi berbasis frekuensi term dan *inverse document frequency* (IDF).
- **Hasil Verifikasi:** Menemukan dokumen yang memuat kata kunci eksak dan singkatan teknis (\"IoU\", \"NMS\", \"YOLO\") dengan presisi tinggi.

### Langkah 6: Retrieval Semantik Vektor Padat (Dense) dengan Filtering
```bash
python content/level-4-advanced-ai/chapter-33/code/dense_retrieval.py
```
- **Tujuan:** Menguji retrieval semantik yang dilengkapi mekanisme *similarity threshold* ($\tau = 0.30$) dan penyaringan kategori dokumen.
- **Hasil Verifikasi:** Dokumen di luar kategori target atau query di luar domain disaring secara otomatis sebelum tahap generasi.

### Langkah 7: Pencarian Hibrida Menggunakan Reciprocal Rank Fusion (RRF)
```bash
python content/level-4-advanced-ai/chapter-33/code/hybrid_retrieval.py
```
- **Tujuan:** Menggabungkan peringkat dari BM25 dan Dense Vector Index menggunakan formula:
  $$RRF(d) = \sum_{m \in \{\text{sparse}, \text{dense}\}} \frac{1}{60 + r_m(d)}$$
- **Hasil Verifikasi:** Mengeliminasi kebutuhan kalibrasi bobot antarmetrik dan menghasilkan daftar kandidat retrieval yang seimbang.

### Langkah 8: Neural Cross-Scoring Reranker
```bash
python content/level-4-advanced-ai/chapter-33/code/reranking_demo.py
```
- **Tujuan:** Menilai ulang kandidat teratas menggunakan evaluasi interaksi atensi silang (kecocokan n-gram frasa eksak, penyelarasan judul, dan kerapatan kata kunci).
- **Hasil Verifikasi:** Mempromosikan potongan dokumen yang memuat jawaban paling spesifik ke peringkat teratas.

### Langkah 9: Konstruksi Konteks & Manajemen Anggaran Token
```bash
python content/level-4-advanced-ai/chapter-33/code/context_builder.py
```
- **Tujuan:** Menyusun prompt RAG terstruktur di bawah batasan kapasitas karakter tertentu, menduplikasi teks yang mirip, dan memisahkan instruksi menggunakan delimiter `<RETRIEVED_CONTEXT>`.
- **Hasil Verifikasi:** Tercipta prompt bersih beranggaran ketat dengan pemetaan ID dokumen yang jelas.

### Langkah 10: Generator Jawaban Berbasis Bukti Deterministik
```bash
python content/level-4-advanced-ai/chapter-33/code/grounded_answer_demo.py
```
- **Tujuan:** Menghasilkan jawaban ekstraktif berbasis bukti dan mendemonstrasikan penolakan menjawab (*abstention*) pada query tanpa bukti pendukung.
- **Catatan Penting:** Modul ini adalah baseline aturan deterministik offline untuk pengujian reproduksibilitas ilmiah, bukan inferensi generatif LLM.

### Langkah 11: Audit dan Validasi Keabsahan Sitasi
```bash
python content/level-4-advanced-ai/chapter-33/code/citation_validation.py
```
- **Tujuan:** Memvalidasi bahwa setiap tag sitasi `[Doc: <id>]` merujuk ke dokumen yang benar-benar ada dalam konteks dan memuat bukti tekstual yang relevan.
- **Hasil Verifikasi:** Mendeteksi sitasi hantu (*phantom citations*) dan kalimat tanpa referensi (*unsupported claims*).

### Langkah 12: Evaluasi Kuantitatif Terpisah (Retrieval vs Generasi)
```bash
python content/level-4-advanced-ai/chapter-33/code/rag_evaluation.py
```
- **Tujuan:** Mengevaluasi pipeline terhadap 6 query uji benchmark:
  - Metrik Retrieval: Precision@3, Recall@3, Hit Rate@3, Mean Reciprocal Rank (MRR).
  - Metrik Generasi: Groundedness, Faithfulness, Citation Correctness.
- **Hasil Verifikasi:** Memperoleh nilai kuantitatif yang terdokumentasi rapi.

### Langkah 13: Eksekusi Pipeline RAG Menyeluruh
```bash
python content/level-4-advanced-ai/chapter-33/code/rag_pipeline.py
```
- **Tujuan:** Menjalankan pipeline modular lengkap dari ingestion hingga audit sitasi dalam satu alur eksekusi terpadu.

### Langkah 14: Pertahanan Keamanan dan Deteksi Indirect Prompt Injection
```bash
python content/level-4-advanced-ai/chapter-33/code/rag_security_checks.py
```
- **Tujuan:** Memindai dokumen adversarial yang memuat teks manipulatif penyusup (*indirect prompt injection*) dan menegakkan izin akses level dokumen (*Access Control Lists*).

---

## 3. LATIHAN BERTINGKAT (3 TIERS)

### Tingkat Dasar: Pemahaman & Replikasi
1. **Latihan 1.1:** Ubah ukuran potongan teks pada `document_chunking.py` menjadi $chunk\_size = 150$ dan $chunk\_overlap = 25$. Amati dan catat berapa jumlah total chunk yang dihasilkan untuk dokumen `doc_rag_01`.
   *Solusi Mandiri:* Dengan ukuran chunk yang lebih kecil, jumlah potongan akan meningkat dari 3 menjadi 5 potongan, menghasilkan granularitas pencarian yang lebih tinggi namun berisiko memotong kalimat panjang.
2. **Latihan 1.2:** Pada `context_builder.py`, lakukan pengujian dengan menetapkan `max_context_chars = 300`. Apa yang terjadi pada chunk kedua yang memiliki skor relevansi tinggi jika total panjang melebihi anggaran?
   *Solusi Mandiri:* Chunk kedua akan dipotong (*truncated*) atau diabaikan sepenuhnya untuk mencegah pelanggaran anggaran context window model.

### Tingkat Menengah: Eksperimen & Optimasi
1. **Latihan 2.1:** Pada `hybrid_retrieval.py`, ubah konstanta penghalus $k$ pada formula RRF dari $k=60$ menjadi $k=10$ dan $k=100$. Bandingkan pengaruhnya terhadap bobot peringkat dokumen teratas.
   *Panduan:* Nilai $k$ yang lebih kecil memberikan penalti yang jauh lebih drastis kepada peringkat di bawah 1, membuat dokumen peringkat pertama sangat dominan. Nilai $k=60$ adalah standar heuristik yang meratakan pengaruh antar-peringkat.
2. **Latihan 2.2:** Tambahkan aturan pemfilteran berbasis tag kata kunci (`tags`) pada `dense_retrieval.py` sehingga pencarian hanya mempertimbangkan dokumen yang memiliki tag `"evaluation"` atau `"security"`.

### Tingkat Mahir: Desain & Implementasi Lanjutan
1. **Latihan 3.1:** Rancang sebuah fungsi detektor kontradiksi (*contradiction detector*) pada `citation_validation.py` yang mendeteksi apabila dua dokumen yang diambil menyajikan klaim yang bertentangan (misalnya mengenai tanggal rilis atau definisi teknis berlawanan).
2. **Latihan 3.2:** Integrasikan model embedding lokal berbasis Sentence-Transformers (jika terpasang) atau perluas representasi vektor berdimensi $d=128$ pada `rag_vector_index.py` untuk menguji peningkatan nilai Recall@3 pada benchmark query.

---

## 4. PANDUAN MINI PROJECT

### Judul Proyek
**RAG-Based Question Answering for AI Learning Materials**

### Deskripsi Tugas
Mahasiswa diminta merancang dan membangun purwarupa sistem tanya-jawab cerdas berbasis **Retrieval-Augmented Generation (RAG)** yang mampu menjawab pertanyaan seputar materi perkuliahan kecerdasan buatan secara terverifikasi dengan rujukan dokumen sumber yang sah, serta memiliki mekanisme pertahanan terhadap dokumen berbahaya (*adversarial prompt injection*).

### Luaran yang Wajib Dikumpulkan
1. **Laporan Teknis (PDF/Markdown):**
   - Ringkasan arsitektur sistem dan diagram alur data.
   - Analisis trade-off chunk size dan overlap.
   - Hasil evaluasi kuantitatif (Precision@3, Recall@3, MRR, Groundedness).
   - Analisis studi kasus kegagalan retrieval dan mitigasi keamanannya.
2. **Kode Sumber (Python):**
   - Script pipeline yang dapat direproduksi secara mandiri (*reproducible*).
   - Berkas korpus data dan skenario query pengujian.
3. **Log Eksekusi & Audit:**
   - Bukti log eksekusi pengujian sitasi dan penolakan menjawab (*abstention*).

---

## 5. RUBRIK PENILAIAN (100 POIN)

| Komponen Penilaian | Bobot | Kriteria Pemenuhan Standar |
| :--- | :---: | :--- |
| **1. Problem & Corpus Understanding** | **15 Poin** | Memahami arsitektur RAG, keterbatasan pengetahuan parametrik LLM, pemilihan dokumen korpus yang representatif, dan pencatatan *source provenance* yang konsisten. |
| **2. Ingestion & Retrieval Engineering** | **25 Poin** | Implementasi strategi chunking hierarkis dengan overlap yang proporsional, integrasi pencarian hibrida (BM25 + Dense Vektor) melalui RRF, serta tahap reranking yang berfungsi efektif. |
| **3. Context Construction & Grounded Response** | **25 Poin** | Manajemen anggaran konteks prompt yang ketat, sintesis jawaban yang bersandar pada bukti faktual, penegakan atribusi sitasi `[Doc: <id>]`, dan perilaku abstensi yang tepat saat bukti tidak mencukupi. |
| **4. Evaluation & Security Defense** | **25 Poin** | Pemisahan evaluasi metrik retrieval (Precision@k, Recall@k, MRR) dan generasi (Groundedness, Faithfulness), audit keabsahan sitasi, serta pengujian pertahanan terhadap *indirect prompt injection* dan ACL dokumen. |
| **5. Code Quality & Documentation** | **10 Poin** | Kebersihan kode, modularitas arsitektur, ketiadaan dependensi eksternal tak terdokumentasi, instruksi eksekusi yang jelas, dan reproduksibilitas 100% pada CPU. |

---

## 6. DAFTAR PUSTAKA & REFERENSI AKADEMIK

1. **Lewis, P., et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems (NeurIPS 2020), 33, 9459-9474.
2. **Robertson, S., & Zaragoza, H. (2009).** *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 333-389.
3. **Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).** *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* Proceedings of the 32nd International ACM SIGIR Conference, 758-759.
4. **Gao, Y., et al. (2023).** *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv preprint arXiv:2312.10997.
5. **National Institute of Standards and Technology (NIST). (2023).** *Artificial Intelligence Risk Management Framework (AI RMF 1.0).* NIST Trustworthy and Responsible AI.

