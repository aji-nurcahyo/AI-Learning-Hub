# Bab 32: Embeddings & Vector Search

## Representasi Vektor Semantik, Aljabar Kesamaan, Indeks FAISS, dan Evaluasi Temu Kembali Informasi

---

## 1. Hakikat Representasi Vektor (Embeddings)

Dalam rekayasa sistem kecerdasan buatan, sebagian besar informasi yang dikonsumsi dan dihasilkan manusia berwujud data tak terstruktur (*unstructured data*): teks bahasa alami, rekaman audio percakapan, citra medis, hingga sinyal telemetri sensor industri. Komputer dan prosesor numerik pada dasarnya tidak memiliki kemampuan bawaan untuk memahami makna simbolik dari kata atau gambar secara langsung. Komputer hanya mampu mengeksekusi operasi aritmatika dan aljabar linier pada skalar, vektor, dan matriks.

**Embeddings** adalah fungsi pemetaan matematis yang mentranslasikan entitas diskrit atau tak terstruktur ke dalam titik-titik koordinat pada ruang vektor berdimensi kontinu yang padat (*dense vector space*):
$$f: \mathcal{O} \to \mathbb{R}^d, \quad d \ll |\mathcal{V}|$$

Di mana $\mathcal{O}$ adalah objek masukan (misalnya sebuah kata, kalimat, atau dokumen), $d$ adalah dimensi ruang representasi (biasanya bernilai antara $32$ hingga $1536$), dan $|\mathcal{V}|$ adalah ukuran ruang simbolik asal.

```
+---------------------------------------------------------------------------------------------------------+
|                                  KONSEP TRANSFORMASI KE RUANG VEKTOR                                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       DATA TAK TERSTRUKTUR                                       RUANG VEKTOR PADAT (DENSE)             |
|                                                                                                         |
|       "Jaringan Syaraf Tiruan" ───┐                             Dimensi 2 (Domain AI)                   |
|                                   │                                      ▲                              |
|       "Deep Learning"          ───┼──> [ ENCODER EMBEDDING ] ──>        │   * "Jaringan Syaraf"        |
|                                   │        (Neural Network)              │  * "Deep Learning"           |
|       "Antena Gelombang 5G"    ───┘                                      │                              |
|                                                                          │        * "Antena 5G"         |
|                                                                          └──────────────────────►       |
|                                                                                    Dimensi 1 (Hardware) |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 1.1 Representasi Jarang (*Sparse*) vs Representasi Padat (*Dense*)
Secara historis, temu kembali informasi (*Information Retrieval*) bertumpu pada representasi jarang (*Sparse Representation*):
- **One-Hot Encoding & Bag-of-Words**: Setiap kata direpresentasikan oleh vektor ortogonal berukuran kosakata penuh $|\mathcal{V}|$ ($> 50.000$ dimensi) di mana hanya ada satu angka $1$ dan sisanya angka $0$. Akibatnya, perkalian titik antara dua kata apa pun yang berbeda selalu menghasilkan nilai nol:
  $$\mathbf{v}_{\text{kucing}} \cdot \mathbf{v}_{\text{anjing}} = 0$$
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Memberikan bobot statistik leksikal pada kemunculan kata penting, namun tetap menderita akibat masalah ketidakcocokan kosakata (*Vocabulary Mismatch Problem*). Jika kueri pencarian menggunakan kata "dokter" sementara dokumen menulis "tenaga medis", sistem leksikal gagal menemukan keterkaitan tersebut.

Sebaliknya, **Dense Embeddings** memampatkan seluruh informasi ke dalam ruang kontinu berdimensi tetap $d$. Setiap dimensi merepresentasikan fitur semantik laten yang dipelajari secara otomatis, sehingga konsep-konsep serupa secara alami mengelompok saling berdekatan.

```
+---------------------------------------------------------------------------------------------------------+
|                                 KOMPARASI REKAYASA: SPARSE VS DENSE                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Karakteristik               Sparse (TF-IDF / BM25)               Dense Embeddings (Vector Search)     |
|   ─────────────               ──────────────────────               ────────────────────────────────     |
|   Dimensi Vektor              Sangat Tinggi (|V| > 50.000)         Rendah / Padat (d = 32 s/d 1536)     |
|   Nilai Elemen                Mayoritas 0.0 (Jarang)               Bilangan Riil Kontinu (Padat)        |
|   Pencocokan Semantik         Gagal pada sinonim / parafrasa       Unggul menangkap makna tersirat      |
|   Pencocokan Istilah Unik     Sangat Akurat (Nomor part, akronim)  Terkadang kabur pada entitas mirip   |
|   Struktur Indeks             Inverted Index Leksikal              Vector Index (Flat, IVF, HNSW)       |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Fondasi Matematika Vektor & Metrik Kesamaan

Untuk menentukan apakah dua dokumen memiliki kesamaan topik atau apakah sebuah dokumen relevan dengan kueri pengguna, sistem menghitung jarak atau sudut antara dua vektor representasi $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$.

```
                    VEKTOR REFERENSI u
                           ▲
                           │ \
                           │   \  Sudut Theta (Orientasi Murni -> Cosine Similarity)
                           │     \
                           │       ▼  VEKTOR v
                           │        \
                           │         \
                           │          \ Jarak Garis Lurus (Euclidean Distance)
                           │           \
                           └────────────────────────►
```

### 2.1 Dot Product (Inner Product)
Perkalian titik antara dua vektor $\mathbf{u}$ dan $\mathbf{v}$ didefinisikan sebagai penjumlahan hasil kali elemen-elemennya:
$$\mathbf{u} \cdot \mathbf{v} = \langle \mathbf{u}, \, \mathbf{v} \rangle = \sum_{i=1}^d u_i v_i$$
Secara geometris, hubungan ini dapat dinyatakan sebagai:
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|_2 \, \|\mathbf{v}\|_2 \cos(\theta)$$
Dot Product menggabungkan dua informasi sekaligus: sudut relatif $\theta$ dan panjang magnitudo kedua vektor. Operasi ini merupakan operasi paling efisien pada tingkat arsitektur CPU/GPU modern melalui instruksi perangkat keras SIMD/BLAS GEMM.

### 2.2 Euclidean Distance (Jarak L2)
Mengukur jarak geometris terpendek antara dua ujung vektor:
$$d_{\text{Euc}}(\mathbf{u}, \, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$
Dalam perankingan sistem temu kembali, jarak Euclidean berbanding terbalik dengan kemiripan: **semakin kecil jarak, semakin mirip kedua dokumen** (*Ascending Order*).

### 2.3 Cosine Similarity & Cosine Distance
Mengukur nilai kosinus dari sudut $\theta$ di antara kedua vektor:
$$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \, \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$
- Bernilai $+1.0$: Searah sempurna ($\theta = 0^\circ$).
- Bernilai $0.0$: Saling tegak lurus / ortogonal ($\theta = 90^\circ$).
- Bernilai $-1.0$: Berlawanan arah murni ($\theta = 180^\circ$).

**Cosine Distance** didefinisikan sebagai komplemen jarak:
$$d_{\text{cos}}(\mathbf{u}, \, \mathbf{v}) = 1.0 - \cos(\theta) \in [0.0, \, 2.0]$$

Cosine similarity memiliki keunggulan fundamental: **kebal terhadap panjang teks**. Dokumen singkat (1 paragraf) dan dokumen panjang (10 halaman) yang membahas topik yang sama akan menghasilkan orientasi sudut vektor yang hampir identik, meskipun vektor dokumen panjang memiliki magnitudo lebih besar.

---

## 3. Teorema Normalisasi L2 & Kesetaraan Aljabar

Dalam perancangan basis data vektor skala industri (*vector database*), menghitung akar kuadrat dan pembagian norma untuk setiap kueri secara berulang menimbulkan penalti latensi komputasi yang besar. Solusi standar industri adalah melakukan **Pra-Normalisasi L2** (*L2 Normalization*).

Sebuah vektor dinormalisasi menjadi **Vektor Satuan** (*Unit Vector*):
$$\hat{\mathbf{u}} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2} \implies \|\hat{\mathbf{u}}\|_2 = 1.0$$

### 3.1 Teorema Kesetaraan Dot Product dan Cosine Similarity
Jika dua vektor telah dinormalisasi L2 ($\|\hat{\mathbf{u}}\|_2 = 1$ dan $\|\hat{\mathbf{v}}\|_2 = 1$), maka penyebut pada rumus Cosine Similarity bernilai tepat $1.0$:
$$\cos(\theta) = \frac{\hat{\mathbf{u}} \cdot \hat{\mathbf{v}}}{\|\hat{\mathbf{u}}\|_2 \|\hat{\mathbf{v}}\|_2} = \frac{\hat{\mathbf{u}} \cdot \hat{\mathbf{v}}}{1.0 \times 1.0} \equiv \hat{\mathbf{u}} \cdot \hat{\mathbf{v}}$$
**Kesimpulan 1**: Pada vektor satuan, menghitung Cosine Similarity cukup dilakukan melalui satu operasi perkalian titik (Dot Product) murni tanpa pembagian sama sekali!

### 3.2 Teorema Relasi Jarak Euclidean dan Cosine Distance
Kuadrat jarak Euclidean antara dua vektor satuan dapat dijabarkan secara aljabar:
$$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = \|\hat{\mathbf{u}}\|_2^2 + \|\hat{\mathbf{v}}\|_2^2 - 2 (\hat{\mathbf{u}} \cdot \hat{\mathbf{v}}) = 1 + 1 - 2 \cos(\theta) = 2 (1 - \cos(\theta)) = 2 \cdot d_{\text{cos}}$$
$$\implies \|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2 = \sqrt{2 \cdot d_{\text{cos}}}$$
**Kesimpulan 2**: Pada vektor satuan, pengurutan jarak Euclidean terdekat menghasilkan urutan peringkat yang **IDENTIK 100%** dengan pengurutan Cosine Similarity tertinggi! Inilah alasan mengapa modul seperti `IndexFlatIP` pada FAISS menjadi tulang punggung pencarian kesamaan kosinus berkecepatan tinggi.

---

## 4. Pencarian Tetangga Terdekat Eksak (*Exact Nearest Neighbor Search*)

Algoritma paling murni dalam temu kembali vektor adalah **Exact Brute-Force Search**.

```
+---------------------------------------------------------------------------------------------------------+
|                                    MEKANISME PENCARIAN VEKTOR EKSAK                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       Kueri q [1 x d]                                            Basis Data Vektor X [N x d]            |
|             │                                                                │                          |
|             v                                                                v                          |
|       [ Normalisasi L2 ]                                            [ Normalisasi L2 ]                  |
|             │                                                                │                          |
|             └───────────────────────────────┬────────────────────────────────┘                          |
|                                             │                                                           |
|                                             v                                                           |
|                           [ Perkalian Matriks BLAS: q @ X.T ] ──> Vektor Skor [1 x N]                   |
|                                             │                                                           |
|                                             v                                                           |
|                           [ Seleksi Top-K via argpartition ]  ──> [ Peringkat Dokumen Teratas ]         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 4.1 Kompleksitas Komputasi
- **Kompleksitas Waktu**: $\mathcal{O}(N \cdot d)$ per kueri.
- **Kompleksitas Memori**: $\mathcal{O}(N \cdot d)$ untuk menyimpan seluruh koordinat vektor floating-point di memori utama.
- **Jaminan Kualitas**: Menghasilkan akurasi 100% (*Ground Truth*) tanpa pernah melewatkan tetangga terdekat sejati.

### 4.2 Optimalisasi `np.argpartition`
Pada basis data berukuran puluhan ribu, melakukan pengurutan penuh (*full sort*) via `np.argsort` memiliki kompleksitas $\mathcal{O}(N \log N)$. Pengambilan Top-$k$ dapat diakselerasi menjadi $\mathcal{O}(N)$ menggunakan algoritma pemilihan partisi linier:
```python
# Ambil k elemen terbesar tanpa mengurutkan N-k elemen sisanya:
top_k_indices = np.argpartition(-scores, kth=k-1)[:k]
# Urutkan hanya k elemen terpilih:
sorted_indices = top_k_indices[np.argsort(-scores[top_k_indices])]
```

---

## 5. Indeks Aproksimasi (*Approximate Nearest Neighbor - ANN*)

Ketika basis data bertumbuh melampaui jutaan vektor ($N > 10^6$), pencarian eksak $\mathcal{O}(N \cdot d)$ mulai menimbulkan latensi yang tidak dapat diterima untuk layanan interaktif real-time ($> 100$ ms). **Approximate Nearest Neighbor (ANN)** hadir sebagai kompromi rekayasa: mengorbankan sedikit akurasi temu kembali (Recall turun $1-3\%$) demi akselerasi kecepatan hingga $10\times$ sampai $100\times$.

```
+---------------------------------------------------------------------------------------------------------+
|                                 ARSITEKTUR INDEKS INVERTED FILE (IVF)                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       RUANG VEKTOR PENUH (N Dokumen)                                  STRUKTUR INVERTED LIST            |
|                                                                                                         |
|       ┌──────────┬──────────┬──────────┐                              Sel #1 ──> [Doc 2, Doc 15, ...]   |
|       │  Sel #1  │  Sel #2  │  Sel #3  │                              Sel #2 ──> [Doc 4, Doc 88, ...]   |
|       │    * C1  │    * C2  │    * C3  │                              Sel #3 ──> [Doc 7, Doc 91, ...]   |
|       ├──────────┼──────────┼──────────┤                              ...                               |
|       │  Sel #4  │  Sel #5  │  Sel #6  │                                                                |
|       │    * C4  │  * Q * C5│    * C6  │ ──> Saat kueri Q tiba di dekat Centroid C5:                   |
|       └──────────┴──────────┴──────────┘     Sistem HANYA memindai Sel #5 (dan nprobe tetangga)!       |
|            (Partisi Sel Voronoi via K-Means) Tidak perlu memindai seluruh N dokumen di ruang vektor!    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 5.1 Inverted File Index (IVF) & Sel Voronoi
IVF mengelompokkan ruang vektor ke dalam $n_{\text{list}}$ sel menggunakan algoritma K-Means (*Centroids*). Setiap vektor didaftarkan ke dalam *inverted list* sel terdekatnya.

Saat kueri pencarian berlangsung:
1. Sistem menghitung jarak kueri hanya ke $n_{\text{list}}$ titik centroid.
2. Memilih $n_{\text{probe}}$ centroid terdekat ($n_{\text{probe}} \ll n_{\text{list}}$).
3. Hanya memindai vektor-vektor yang berada di dalam sel-sel terpilih tersebut.

Rasio pemindaian data berkurang drastis menjadi $\approx \frac{n_{\text{probe}}}{n_{\text{list}}}$. Pengembang dapat mengatur trade-off secara dinamis: menaikkan $n_{\text{probe}}$ untuk menaikkan Recall, atau menurunkannya untuk latensi minimum.

### 5.2 Hierarchical Navigable Small World (HNSW)
HNSW memodelkan ruang vektor sebagai graf multi-lapisan bertingkat:
- **Lapisan Teratas**: Memiliki koneksi graf berjarak jauh (*express highway*) untuk melompati ruang semantik dengan cepat.
- **Lapisan Terbawah**: Memiliki koneksi lokal padat untuk penyisiran presisi tinggi.
HNSW saat ini merupakan algoritma ANN paling populer di industri karena menawarkan kurva latensi-recall terbaik tanpa memerlukan tahapan *training* awal seperti K-Means pada IVF.

---

## 6. Fondasi Pustaka FAISS (Facebook AI Similarity Search)

Dikembangkan oleh Meta Fundamental AI Research (Johnson et al., 2019), **FAISS** adalah pustaka standar industri untuk pencarian kesamaan vektor berkecepatan tinggi yang dioptimalkan untuk CPU dan GPU.

### 6.1 Keluarga Indeks Utama FAISS
1. `IndexFlatL2`: Indeks datar eksak berbasis kuadrat jarak Euclidean ($\|u-v\|^2$).
2. `IndexFlatIP`: Indeks datar eksak berbasis Inner Product ($u \cdot v$). Sangat cepat untuk vektor ternormalisasi L2.
3. `IndexIVFFlat`: Indeks aproksimasi berbasis partisi sel Voronoi K-Means.
4. `IndexHNSWFlat`: Indeks aproksimasi berbasis navigasi graf multi-lapisan.

### 6.2 Pemisahan Vektor & Rekonstruksi Metadata (*Hydration*)
Karakteristik penting dalam perancangan arsitektur FAISS:
> **Peringatan Rekayasa**: Objek indeks FAISS hanya menyimpan tensor numerik dan ID baris bilangan bulat `0, 1, ..., N-1`. FAISS **tidak pernah menyimpan string teks, judul dokumen, atau metadata bisnis**.

Aplikasi bertanggung jawab mengelola basis data metadata terpisah (*Document Store*). Saat FAISS mengembalikan ID hasil pencarian, aplikasi melakukan rekonsiliasi (*Hydration*) untuk mengambil teks asli dokumen sebelum disajikan ke pengguna.

---

## 7. Pipa Lengkap Pencarian Vektor (Vector Search Pipeline)

Alur kerja sistem temu kembali semantik diimplementasikan secara terstruktur melalui 8 langkah rekayasa:

```
    [1. DOKUMEN MENTAH] ──> Teks artikel, regulasi, materi pembelajaran
             │
             v
    [2. PREPROCESSING]   ──> Pembersihan teks, penanganan whitespace, chunking
             │
             v
    [3. ENCODER MODEL]   ──> Ekstraksi dense embedding [d = 32]
             │
             v
    [4. L2 NORMALISASI]  ──> Transformasi ke vektor satuan (||v|| = 1.0)
             │
             v
    [5. VECTOR INDEX]    ──> Indeksasi ke basis data vektor (IndexFlatIP / FAISS)
             │
             ▲
             │ (Kueri Pengguna)
             │
    [6. QUERY ENCODER]   ──> Vektorisasi & L2 normalisasi kueri pencarian
             │
             v
    [7. SIMILARITY TOP-K]──> Eksekusi pencarian kesamaan kosinus
             │
             v
    [8. METADATA HYDRATE]──> Pengambilan teks dokumen berdasarkan ID & penyajian hasil
```

*Catatan Ruang Lingkup*: Alur sistem berhenti tepat pada perolehan dokumen relevan (*Information Retrieval*). Penggabungan hasil retrieval dengan generator teks berbasis Large Language Models (LLM) merupakan domain arsitektur **Retrieval-Augmented Generation (RAG)** yang dibahas secara mendalam pada Chapter 33.

---

## 8. Metrik Evaluasi Sistem Temu Kembali (Information Retrieval Metrics)

Kualitas sistem pencarian tidak dapat diukur menggunakan akurasi klasifikasi standar karena perankingan dokumen memiliki dimensi posisi (*positional rank*).

```
+---------------------------------------------------------------------------------------------------------+
|                                    TAKSONOMI METRIK EVALUASI RETRIEVAL                                  |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       HASIL PENCARIAN SISTEM (Top-3)                     GROUND TRUTH DOKUMEN RELEVAN                   |
|       Peringkat 1: DOC-11 (Relevan)                      - DOC-11                                       |
|       Peringkat 2: DOC-04 (Tidak Relevan)                - DOC-05                                       |
|       Peringkat 3: DOC-05 (Relevan)                                                                     |
|                                                                                                         |
|       Kalkulasi:                                                                                        |
|       - Precision@3 = 2 dokumen relevan / 3 dokumen terambil = 66.67%                                   |
|       - Recall@3    = 2 dokumen relevan / 2 total dokumen relevan = 100.00%                             |
|       - Hit Rate@3  = 1.0 (Karena minimal ada 1 dokumen relevan di top-3)                               |
|       - Reciprocal Rank (RR) = 1 / 1 (Dokumen relevan pertama ada di peringkat 1) = 1.0000             |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 8.1 Precision@k
Proporsi dokumen terambil pada kedalaman $k$ yang terbukti relevan secara faktual:
$$\text{Precision@k} = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{k}$$

### 8.2 Recall@k
Proporsi dari keseluruhan himpunan dokumen relevan yang berhasil ditemukan dalam daftar teratas $k$:
$$\text{Recall@k} = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}$$

### 8.3 Hit Rate@k (Hit@k)
Indikator biner yang bernilai $1.0$ jika minimal ada satu dokumen relevan yang tertangkap dalam daftar top-$k$, dan $0.0$ jika gagal total:
$$\text{Hit Rate@k} = \begin{cases} 1.0, & \text{jika } |\text{Retrieved}_k \cap \text{Relevant}| > 0 \\ 0.0, & \text{lainnya} \end{cases}$$

### 8.4 Mean Reciprocal Rank (MRR)
Metrik paling krusial untuk mengukur kualitas perankingan. Menghitung rata-rata dari kebalikan posisi peringkat (*reciprocal rank*) dokumen relevan pertama yang dijumpai:
$$\text{RR}(q) = \frac{1}{\text{rank}_q}, \quad \text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{RR}(q)$$
Sistem yang menempatkan dokumen relevan di peringkat 1 menghasilkan $\text{MRR} = 1.0$, sedangkan sistem yang menempatkannya di peringkat 2 menghasilkan $\text{MRR} = 0.5$.

---

## 9. Aspek Keamanan, Privasi & Bias Representasi Vektor

Penerapan pencarian vektor di industri menuntut pertimbangan etika dan keamanan data yang ketat:

### 9.1 Serangan Rekonstruksi Teks (*Embedding Inversion Attack*)
Banyak praktisi pemula keliru menganggap bahwa mengubah teks rahasia (seperti rekam medis atau data perbankan) menjadi vektor angka otomatis menganonimkan data tersebut.
> **Peringatan Keamanan**: Vektor embeddings bukanlah fungsi hash satu arah (*one-way cryptographic hash*). Riset terbaru (Morris et al., 2023; Vec2Text) membuktikan bahwa teks asli dapat direkonstruksi kembali dari vektor embeddings dengan tingkat akurasi linguistik yang tinggi.

**Kaidah Rekayasa**: Data pribadi sensitif (*Personally Identifiable Information - PII*) wajib disanitasi atau diredaksi **sebelum** proses ekstraksi embedding dilakukan.

### 9.2 Bias Representasi Semantik
Ruang vektor merefleksikan asosiasi kata yang terdapat dalam data pelatihannya. Jika data pelatihan mengandung stereotipe sosial atau bias gender, model embedding akan menempatkan kata-kata profesi tertentu lebih dekat ke gender tertentu di ruang laten. Audit kesamaan kosinus antar kelompok demografi wajib dilakukan untuk mendeteksi disparitas representasi.

---

## 10. Latihan Bertingkat (3-Tier Exercises)

### Latihan Tingkat 1: Pemahaman Aljabar Vektor
1. **Soal**: Diberikan kueri $q = [0.6, \, 0.8]$ dan dokumen $d_1 = [0.8, \, 0.6]$ serta $d_2 = [0.0, \, 1.0]$.
   a. Buktikan apakah kedua vektor dokumen telah memenuhi syarat vektor satuan L2!
   b. Hitung Cosine Similarity kueri $q$ terhadap $d_1$ dan $d_2$. Dokumen manakah yang menempati peringkat 1?
2. **Soal**: Mengapa kueri pencarian yang hanya berisi kata sambung atau karakter kosong dapat memicu `ZeroDivisionError` pada implementasi Cosine Similarity naif? Bagaimana cara mitigasinya?

*Pembahasan*:
1. a. $\|d_1\| = \sqrt{0.8^2 + 0.6^2} = \sqrt{0.64 + 0.36} = 1.0$. $\|d_2\| = \sqrt{0^2 + 1^2} = 1.0$. Kedua vektor terbukti merupakan vektor satuan.
   b. $\text{Sim}(q, d_1) = (0.6 \times 0.8) + (0.8 \times 0.6) = 0.48 + 0.48 = 0.9600$.  
      $\text{Sim}(q, d_2) = (0.6 \times 0.0) + (0.8 \times 1.0) = 0.8000$.  
      Dokumen $d_1$ menempati peringkat 1 karena memiliki kesamaan kosinus tertinggi ($0.9600 > 0.8000$).
2. Kueri kosong menghasilkan vektor dengan semua elemen nol ($\|q\|_2 = 0$). Pembagian dengan norma nol memicu *ZeroDivisionError*. Mitigasinya adalah mendeteksi norma vektor dengan nilai ambang $\epsilon = 10^{-12}$ dan mengembalikan skor kesamaan netral $0.0$.

### Latihan Tingkat 2: Implementasi Pemrograman NumPy
**Soal**: Tuliskan fungsi Python `top_k_cosine_search(query_vec, database_matrix, k=3)` yang menerima vektor kueri dan matriks basis data, menormalisasi keduanya ke L2, lalu mengembalikan indeks $k$ dokumen teratas.

*Pembahasan*:
```python
import numpy as np

def top_k_cosine_search(query_vec, database_matrix, k=3):
    # 1. Normalisasi L2
    q_norm = query_vec / np.maximum(np.linalg.norm(query_vec), 1e-12)
    db_norm = database_matrix / np.maximum(np.linalg.norm(database_matrix, axis=1, keepdims=True), 1e-12)
    # 2. Dot Product Matrix Multiplication
    scores = db_norm @ q_norm
    # 3. Pengambilan Top-k
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices, scores[top_indices]
```

### Latihan Tingkat 3: Desain Arsitektur Sistem Temu Kembali
**Soal**: Sebuah platform e-commerce memiliki 10 juta produk. Pencarian leksikal berbasis kata kunci sering gagal ketika pembeli mencari barang menggunakan bahasa sehari-hari (*"baju pesta malam anggun"*), namun pencarian vektor murni sering salah menampilkan produk saat pembeli memasukkan kode seri suku cadang persis (*"SP-404-MK2"*). Rancang arsitektur temu kembali hibrida (*Hybrid Retrieval*) terpadu untuk menyelesaikan masalah ini.

*Pembahasan*:
- **Arsitektur Hibrida**: Terapkan sistem *Two-Stage Retrieval*:
  1. **Stage 1 (Parallel Retrieval)**:
     - Jalur Leksikal (BM25 / Elasticsearch): Menghasilkan 50 kandidat teratas berbasis kecocokan kata persis dan kode suku cadang.
     - Jalur Vektor Padat (FAISS HNSW): Menghasilkan 50 kandidat teratas berbasis kesamaan semantik deskriptif kueri.
  2. **Stage 2 (Fusion & Re-ranking)**:
     - Gabungkan 100 kandidat menggunakan *Reciprocal Rank Fusion (RRF)*:
       $$\text{RRF Score}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{Dense}}(d)}$$
     - Terapkan model *Cross-Encoder Re-ranker* pada 20 produk teratas hasil fusi untuk menentukan urutan final di antarmuka pengguna.

---

## 11. Tugas Proyek Mandiri, Mini Project, dan Enam Milestone Capstone

### 11.1 Deskripsi Mini Project
Mahasiswa wajib mengimplementasikan proyek rekayasa sistem temu kembali semantik bertajuk:  
**"Building a Semantic Search System for AI Learning Materials"**

### 11.2 Enam Milestone Pelaksanaan
1. **Milestone 1**: Pemuatan dan validasi korpus dokumen akademik (`data/ai_telecom_corpus.json`).
2. **Milestone 2**: Implementasi fungsi ekstraksi *Sentence Embedding* deterministik padat.
3. **Milestone 3**: Pembuktian kesetaraan aljabar normalisasi L2 dan penanganan vektor nol.
4. **Milestone 4**: Pembangunan indeks vektor (*Exact Search* atau *FAISS-Compatible Engine*).
5. **Milestone 5**: Eksekusi pencarian kueri benchmark dan komparasi terhadap baseline leksikal TF-IDF.
6. **Milestone 6**: Perhitungan metrik evaluasi kuantitatif (Precision@3, Recall@3, Hit Rate@3, MRR), diseksi galat (*error analysis*), serta penyusunan panduan mitigasi privasi embeddings.

---

## 12. Rubrik Penilaian Proyek Mini (Skala 100 Poin)

| Dimensi Penilaian | Bobot | Kriteria Sempurna (Full Credit) | Kriteria Cukup (Partial Credit) | Kriteria Kurang (Insufficient) |
|---|:---:|---|---|---|
| **1. Pemahaman Masalah & Korpus Data** | 15 Poin | Korpus data dipahami mendalam; skema dokumen dan kueri benchmark terpetakan rapi; batasan sistem temu kembali diidentifikasi jelas. | Korpus data dimuat namun pemahaman skema metadata kueri kurang lengkap. | Data korpus tidak dimuat; deskripsi masalah kabur atau tidak terstruktur. |
| **2. Representasi Vektor & Normalisasi L2** | 25 Poin | Ekstraksi embedding padat berhasil; normalisasi L2 diterapkan benar; pembuktian aljabar Dot Product = Cosine Similarity terverifikasi. | Embedding terbentuk namun normalisasi L2 diabaikan atau menimbulkan zero division. | Ekstraksi vektor gagal; perhitungan aljabar salah; kode error. |
| **3. Implementasi Indeks & Pencarian** | 25 Poin | Pipa pencarian Top-k end-to-end berfungsi sempurna; rekonstruksi metadata dokumen bersih; penanganan kueri stabil. | Pencarian berfungsi namun metadata dokumen tidak terekonstruksi dengan benar (*missing hydration*). | Indeks pencarian gagal dibangun; output tidak relevan. |
| **4. Evaluasi Temu Kembali (IR Metrics)** | 25 Poin | Menghitung Precision@k, Recall@k, Hit Rate@k, dan MRR secara presisi; tabel komparasi sparse vs dense tersedia mendalam. | Metrik evaluasi dihitung namun terbatas pada salah satu indikator tanpa analisis perbandingan komparatif. | Tidak ada perhitungan metrik kuantitatif; klaim subjektif murni. |
| **5. Dokumentasi, Privasi & Referensi** | 10 Poin | Kode modular dan terdokumentasi rapi; analisis risiko privasi embeddings dibahas; menyertakan $\ge 2$ referensi akademik terverifikasi. | Kode dapat dijalankan namun dokumentasi minim; referensi kurang lengkap. | Tidak ada dokumentasi; referensi fiktif atau tidak relevan. |
| **Total Skor Maksimal** | **100 Poin** | **Standar Rekayasa Temu Kembali Informasi Modern** | — | — |

---

## 13. Ringkasan Bab & Referensi Akademik Terverifikasi

### 13.1 Ringkasan Bab 32
1. **Embeddings** memetakan data teks tak terstruktur ke dalam ruang vektor padat kontinu berdimensi tetap di mana kedekatan geometris mencerminkan kedekatan makna semantik.
2. **Normalisasi L2** mengubah vektor menjadi vektor satuan, membuktikan secara aljabar bahwa Dot Product identik dengan Cosine Similarity dan kuadrat jarak Euclidean berbanding lurus dengan Cosine Distance.
3. **Exact Search** memberikan jaminan presisi mutlak $\mathcal{O}(N \cdot d)$, sedangkan **Approximate Nearest Neighbor (ANN)** seperti IVF dan HNSW mempercepat pencarian skala masif dengan trade-off penurunan recall yang terkendali.
4. **FAISS** mengelola komputasi vektor murni secara terpisah dari metadata aplikasi, mewajibkan mekanisme *hydration* untuk merekonstruksi teks dokumen.
5. **Evaluasi Temu Kembali** diukur secara kuantitatif melalui Precision@k, Recall@k, Hit Rate@k, dan Mean Reciprocal Rank (MRR).

### 13.2 Referensi Akademik Terverifikasi
1. **Johnson, J., Douze, M., & Jégou, H. (2019).** *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data, 7(3), pp. 535–547. [DOI: 10.1109/TBDATA.2019.2921572](https://doi.org/10.1109/TBDATA.2019.2921572)
2. **Malkov, Y. A., & Yashunin, D. A. (2018).** *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), pp. 824–836. [DOI: 10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
3. **Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).** *Efficient Estimation of Word Representations in Vector Space.* In Proceedings of the International Conference on Learning Representations (ICLR 2013). [arXiv:1301.3781](https://arxiv.org/abs/1301.3781)
4. **Reimers, N., & Gurevych, I. (2019).** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019), pp. 3982–3992. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
5. **Manning, C. D., Raghavan, P., & Schütze, H. (2008).** *Introduction to Information Retrieval.* Cambridge University Press, Cambridge, UK. [Cambridge Online](https://nlp.stanford.edu/IR-book/)

