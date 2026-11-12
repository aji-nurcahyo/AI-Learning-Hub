# Chapter 38: Multimodal AI Systems
**Level 4: Applied & Advanced AI Systems — Artificial Intelligence Learning Hub**  
*Created & Developed by Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## PART A — FONDASI MULTIMODAL AI

Kecerdasan buatan klasik umumnya beroperasi dalam silo tunggal (**Unimodal AI**): jaringan konvolusional (*CNN*) hanya melihat piksel, sementara model bahasa hanya membaca deretan token teks. Namun, dunia fisik pada hakikatnya bersifat **multimodal**—informasi hadir serentak melalui penglihatan, bahasa, suara, serta telemetri sensor.

**Multimodal AI** adalah paradigma kecerdasan buatan yang mempelajari representasi terpadu, menyelaraskan makna semantik, dan menggabungkan sinyal dari dua atau lebih modalitas heterogen (*heterogeneous modalities*) untuk melakukan penalaran, klasifikasi, maupun penelusuran lintas modalitas (*cross-modal retrieval*).

### 1. Mengapa Unimodal Saja Gagal? Kasus Sinergi Modalitas
Pada banyak keputusan kritis (seperti inspeksi industri, diagnosis medis, atau pemantauan fasilitas otonom), prediksi yang benar hanya dapat dicapai ketika dua modalitas dibaca bersamaan (*synergistic interaction*):

| Skenario Pengamatan | Sinyal Visual (Kamera) | Sinyal Teks/Sensor | Status Peringatan Benar | Kegagalan Unimodal |
| :--- | :---: | :---: | :---: | :--- |
| Panel Bersih + Suhu Normal | `0.0` (Aman) | `0.0` (Normal) | `0` (Aman) | Kedua cabang setuju `0` |
| Panel Bersih + Laporan *Overheat* | `0.0` (Aman) | `1.0` (Panas) | `1` (Peringatan) | Model visual saja salah (`0`) |
| Percikan Api + Suhu Normal | `1.0` (Bahaya) | `0.0` (Normal) | `1` (Peringatan) | Sensor teks saja salah (`0`) |
| Retakan + Teks *Normal Vibration* | `1.0` (Retak) | `0.0` (Normal) | `1` (Peringatan) | Sinyal teks menipu model teks |

Demonstrasi pada `code/multimodal_foundations.py` memperlihatkan bahwa hanya model yang menggabungkan kedua modalitas mampu mempertahankan akurasi keputusan di seluruh skenario sinergi.

### 2. Taksonomi Modalitas dan Tantangan Representasi
```text
+-------------------------------------------------------------------------+
|                    SPEKTRUM MODALITAS DATA HETEROGEN                    |
+-------------------+------------------------+----------------------------+
| Modalitas         | Struktur Alami         | Sifat Sinyal & Dimensi     |
+-------------------+------------------------+----------------------------+
| Citra (Vision)    | Grid Spasial 2D/3D     | Kontinu, padat piksel (RGB)|
| Teks (Language)   | Sekuens Diskrit 1D     | Simbolik, padat semantik   |
| Audio / Suara     | Gelombang Waktu 1D     | Spektrogram frekuensi      |
| Tabular / IoT     | Vektor Numerik & Waktu | Terstruktur, laju beragam  |
+-------------------+------------------------+----------------------------+
```

---

## PART B — MULTIMODAL REPRESENTATION LEARNING & MODALITY GAP

Sebelum dua modalitas dapat dibandingkan atau dipadukan, masing-masing modalitas harus diubah oleh **Encoder** khususnya menjadi vektor fitur berdimensi tetap di ruang laten yang kompatibel.

### 1. Dual Encoder Ringan pada Proyek Chapter 38
Dalam modul ini, seluruh eksperimen menggunakan korpus pasangan citra–teks sintetis (`code/paired_dataset_generator.py`) berukuran `32x32` RGB yang terdiri dari **4 warna** (`red`, `green`, `blue`, `yellow`) $\times$ **4 bentuk geometris** (`circle`, `square`, `triangle`, `rectangle`) = **16 konsep semantik**, dibagi secara ketat menjadi **`320` pasangan latih** dan **`96` pasangan uji tak terlihat** (*strict unseen split*).

1. **`TinyImageEncoder` (`code/image_encoder.py`, `23,584` parameter):**
   - Menerima citra `(B, 3, 32, 32)`.
   - Melalui 3 blok `Conv2d` + `ReLU` + `MaxPool2d`, menghasilkan peta fitur spasial `4x4` $\to$ diratakan menjadi **`16` token patch spasial** berdimensi `64`, yaitu `(B, 16, 64)` untuk *Cross-Attention*.
   - Melakukan *Global Average Pooling* menjadi vektor global citra `(B, 64)`.

2. **`TinyTextEncoder` (`code/text_encoder.py`, `9,024` parameter):**
   - Menerima indeks token `(B, L)` (panjang `L=2`, misalnya `['red', 'circle']`).
   - Mengubah token melalui `nn.Embedding` + proyeksi linear menjadi representasi sekuens `(B, 2, 64)` dan merata-ratakan token aktif (*masked mean-pooling*) menjadi vektor kalimat `(B, 64)`.

```text
  [Citra 3x32x32] ---> TinyImageEncoder (CNN) ---> Patch (B, 16, 64) atau Global (B, 64)
                                                               |
                                                      Projection + L2 Norm
                                                               v
                                                   Unit Hypersphere (B, 32)
                                                               ^
                                                      Projection + L2 Norm
                                                               |
  [Teks Token 2]  ---> TinyTextEncoder (Emb)  ---> Seq   (B, 2, 64)  atau Pooled (B, 64)
```

### 2. Normalisasi L2 pada Hipersfer Satuan & Modality Gap
Agar kesamaan semantik antara vektor citra $v_i \in \mathbb{R}^d$ dan vektor teks $t_j \in \mathbb{R}^d$ diukur semata-mata oleh arah sudutnya (bukan besarnya norma vektor), keduanya diproyeksikan dan dinormalisasi pada permukaan **hipersfer satuan** (*unit hypersphere*):

$$
\hat{v}_i = \frac{W_v v_i}{\|W_v v_i\|_2}, \qquad \hat{t}_j = \frac{W_t t_j}{\|W_t t_j\|_2}, \qquad \text{sim}(\hat{v}_i, \hat{t}_j) = \hat{v}_i^\top \hat{t}_j \in [-1, 1]
$$

Pada `code/multimodal_embeddings.py`, sebelum dilatih secara kontrastif, rata-rata *cosine similarity* pasangan gambar–teks yang cocok hanya **`0.2650`**. Kesenjangan geometris di mana encoder citra dan encoder teks menempati kerucut ruang laten yang terpisah pada awal inisialisasi inilah yang disebut **Modality Gap**.

---

## PART C — ARSITEKTUR MULTIMODAL FUSION

Ketika tujuannya bukan sekadar penelusuran (*alignment*), melainkan **prediksi klasifikasi gabungan** (misalnya memprediksi 16 kelas konsep secara serentak dari pasangan gambar dan teks), sinyal dari kedua encoder dapat digabungkan pada tahap yang berbeda:

### 1. Early Fusion (Feature-Level Concatenation)
```text
[Citra] ---> ImageEncoder ---> z_img (64) --\
                                             +--> [z_img ; z_txt] (128) --> MLP --> Prediksi
[Teks]  ---> TextEncoder  ---> z_txt (64) --/
```
- **Cara Kerja (`code/early_fusion.py`):** Vektor fitur `z_img` dan `z_txt` langsung digabungkan (`torch.cat`) menjadi vektor `128`-d sebelum masuk ke lapisan linear MLP.
- **Kelebihan:** Mudah diimplementasikan dan mampu menangkap korelasi lintas fitur sejak lapisan awal.
- **Kelemahan Fatal:** Bobot MLP terkopling ketat terhadap keberadaan kedua modalitas. Jika satu modalitas hilang/putus saat inferensi, distribusi aktivasi rusak parah.

### 2. Late Fusion (Decision-Level Ensemble)
```text
[Citra] ---> ImageEncoder ---> MLP_img ---> P_img(y|img) --\
                                                            +--> 0.5*P_img + 0.5*P_txt --> Prediksi
[Teks]  ---> TextEncoder  ---> MLP_txt ---> P_txt(y|txt) --/
```
- **Cara Kerja (`code/late_fusion.py`):** Masing-masing encoder memiliki kepala klasifikasi independen; keputusan akhir adalah rata-rata probabilitas (*decision ensemble*).
- **Kelebihan:** Sangat modular dan tangguh menghadapi kegagalan satu sensor/modalitas (*missing modality resilience*).
- **Kelemahan:** Tidak dapat memodelkan interaksi granular tingkat fitur antara wilayah piksel tertentu dan kata tertentu.

### 3. Intermediate Fusion & Cross-Attention
Fitur mentah diproses oleh masing-masing encoder hingga lapisan menengah (*spatial patches* vs *token embeddings*), lalu berinteraksi di tengah melalui proyeksi bersama, lapisan bilinear, atau mekanisme **Cross-Attention** sebelum kepala klasifikasi akhir.

### 4. Hasil Benchmark Empiris Ketahanan Hilang Modalitas
Eksekusi aktual `code/multimodal_comparison.py` pada set uji `96` sampel tak terlihat memperlihatkan perbedaan perilaku yang tegas:

| Arsitektur Fusi | Kedua Modalitas Hadir | Tanpa Teks (*Image Only*) | Tanpa Citra (*Text Only*) | Karakteristik Utama |
| :--- | :---: | :---: | :---: | :--- |
| **Early Fusion** (`early_fusion.py`) | `100.00%` | `25.00%` | `93.75%` | Sangat rapuh jika cabang dominan hilang |
| **Late Fusion** (`late_fusion.py`) | `100.00%` | **`43.75%`** | **`100.00%`** | Paling tangguh (*degrades gracefully*) |
| **Cross-Attention** (`cross_attention_fusion.py`) | `100.00%` | `25.00%` | `62.50%` | Kaya interaksi spasial, butuh input terpadu |

> [!IMPORTANT]
> Perhatikan mengapa akurasi *Image Only* pada data sintetis geometris berada pada kisaran `25.00%–43.75%` sementara *Text Only* mencapai `62.50%–100.00%`: deskripsi teks `'red circle'` secara deterministik memetakan 1-ke-1 ke kelas konsep tanpa noise piksel, sedangkan citra `32x32` memiliki variasi posisi, skala acak, dan *Gaussian pixel noise* sehingga cabang teks menjadi modalitas dominan bila tidak diatur *modality dropout* saat pelatihan.

---

## PART D — MEKANISME CROSS-ATTENTION MULTIMODAL

Dalam **Cross-Attention**, satu modalitas bertindak sebagai pencari informasi (**Query**, $Q$), sementara modalitas lain bertindak sebagai basis pengetahuan yang dicari (**Key**, $K$, dan **Value**, $V$):

$$
\text{CrossAttention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

Pada `code/cross_attention_fusion.py`, token teks (`L_t = 2`) menjadi **Query**, dan **`16` patch spasial citra (`4x4`)** menjadi **Key** dan **Value**. Artinya, kata `"red"` atau `"triangle"` secara selektif menaruh bobot atensi lebih tinggi pada patch gambar tempat objek berwarna merah atau bersudut segitiga berada.

```text
  Teks Token ("red", "circle")  ---> W_q ---> Query Q (B, 2, D)  ---\
                                                                     +--> Softmax(QK^T / sqrt(d_k)) @ V
  Patch Citra 4x4 (16 patches)  ---> W_k ---> Key   K (B, 16, D) ---/         |
                                \--> W_v ---> Value V (B, 16, D) ------------/
                                                                              v
                                                                 Konteks Visual Terfokus (B, 2, D)
```

### Contoh Hitungan Tensor Verifikasi NumPy vs PyTorch (`cross_attention_fusion.py`)
Misalkan terdapat `2` token teks ($Q \in \mathbb{R}^{2 \times 4}$) dan `3` patch gambar ($K, V \in \mathbb{R}^{3 \times 4}$) dengan $d_k = 4 \implies \sqrt{d_k} = 2.0$:

$$
Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0.5 & 0.5 & 0 & 0 \end{bmatrix}, \quad
\frac{QK^\top}{2.0} = \begin{bmatrix} 1.0 & 0.0 & 0.25 \\ 0.0 & 1.0 & 0.25 \end{bmatrix}
$$

Setelah fungsi `softmax` per baris diterapkan:
- Baris 0 (Token Teks 0): $\alpha_0 = \text{softmax}([1.0, 0.0, 0.25]) = [0.5434,\ 0.1999,\ 0.2567]$, jumlah baris $= 1.0000$.
- Baris 1 (Token Teks 1): $\alpha_1 = \text{softmax}([0.0, 1.0, 0.25]) = [0.1999,\ 0.5434,\ 0.2567]$, jumlah baris $= 1.0000$.
- Hasil `out_np = weights @ V` berukuran `(2, 4)` terverifikasi identik hingga presisi `1e-5` dengan eksekusi `torch.bmm`.

---

## PART E — CONTRASTIVE LEARNING & SYMMETRIC INFONCE LOSS

Untuk membangun sistem penelusuran lintas modalitas tanpa terikat pada himpunan kelas tertutup, kita melatih *Dual Encoder* agar menarik pasangan citra–teks yang cocok berdekatan dan mendorong pasangan tak cocok menjauh menggunakan **Symmetric InfoNCE (Contrastive Loss)**.

Diberikan satu mini-batch berisi $N$ pasangan $\{(I_i, T_i)\}_{i=1}^N$ dengan vektor ternormalisasi L2 $(\hat{v}_i, \hat{t}_i)$ dan parameter suhu (*temperature*) $\tau > 0$, kita membentuk matriks skor logit $S \in \mathbb{R}^{N \times N}$:

$$
S_{i,j} = \frac{\hat{v}_i^\top \hat{t}_j}{\tau}
$$

Loss dihitung secara simetris dari dua arah menggunakan *Cross-Entropy* di mana target benar selalu berada pada **diagonal utama** ($j = i$):

$$
\mathcal{L}_{I \to T} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(S_{i,i})}{\sum_{j=1}^{N} \exp(S_{i,j})}, \qquad
\mathcal{L}_{T \to I} = -\frac{1}{N}\sum_{j=1}^{N} \log \frac{\exp(S_{j,j})}{\sum_{i=1}^{N} \exp(S_{i,j})}
$$

$$
\mathcal{L}_{\text{Symmetric InfoNCE}} = \frac{1}{2}\left(\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}\right)
$$

### Contoh Hitungan Manual Matriks $3 \times 3$ (`code/contrastive_loss.py`)
Dengan `3` pasang vektor ternormalisasi L2 dan suhu $\tau = 0.5$, matriks kesamaan kosinus $I T^\top$ adalah:
- Baris Citra 0: `[0.9000, 0.1000, 0.7071]` $\implies \text{logits}_0 = [1.8000, 0.2000, 1.4142]$
- Baris Citra 1: `[0.4359, 0.9950, 0.7071]` $\implies \text{logits}_1 = [0.8718, 1.9900, 1.4142]$
- Baris Citra 2: `[0.9446, 0.7743, 1.0000]` $\implies \text{logits}_2 = [1.8892, 1.5486, 2.0000]$

Rata-rata *Cross-Entropy* per baris menghasilkan **`Manual I->T Loss = 0.73244`**, per kolom menghasilkan **`Manual T->I Loss = 0.75112`**, sehingga:
$$
\mathcal{L}_{\text{Symmetric}} = \frac{0.73244 + 0.75112}{2} = \mathbf{0.74178}
$$
Nilai ini terverifikasi identik dengan hasil `F.cross_entropy` pada PyTorch (`0.74178`).

> [!WARNING]
> **Jebakan False Negatives dalam Mini-Batch:** Jika satu mini-batch memuat dua pasangan berbeda dari konsep yang sama (misalnya dua sampel `'red circle'`), elemen off-diagonal antar keduanya dianggap sebagai kelas negatif oleh *InfoNCE* dan didorong saling menjauh! Di `clip_miniature.py`, masalah ini dicegah dengan **Unique-Concept Batch Sampling** (setiap batch berisi tepat `16` konsep berbeda).

---

## PART F — ARSITEKTUR CLIP & PENELUSURAN ZERO-SHOT

**CLIP (Contrastive Language–Image Pre-training, Radford et al., 2021)** mempopulerkan arsitektur *Dual Encoder* simetris pada skala `400` juta pasangan citra–teks dari web.

```text
                    MATRIKS SKOR SIMILARITY CLIP (N x N)
                        T_1          T_2          T_3     ...   T_N
                   +------------+------------+------------+-----+------+
   Img_1  ----->   |  (+) I1.T1 |  (-) I1.T2 |  (-) I1.T3 | ... |  (-) |
   Img_2  ----->   |  (-) I2.T1 |  (+) I2.T2 |  (-) I2.T3 | ... |  (-) |
   Img_3  ----->   |  (-) I3.T1 |  (-) I3.T2 |  (+) I3.T3 | ... |  (-) |
    ...            |     ...    |     ...    |     ...    | ... |  ... |
   Img_N  ----->   |     (-)    |     (-)    |     (-)    | ... |  (+) |
                   +------------+------------+------------+-----+------+
```

Untuk **Zero-Shot Classification**, kelas baru yang sebelumnya tidak dilatih dengan kepala klasifikasi tetap cukup dituliskan sebagai deskripsi teks (misalnya `"a photo of a {class}"`), dienkode oleh *Text Encoder* sekali saja, lalu dicocokkan melalui skor *cosine similarity* tertinggi terhadap embedding gambar.

> [!NOTE]
> **Kejujuran Skala Pedagogis:** `MiniatureDualEncoder` pada modul ini menggunakan `36,704` parameter pada himpunan tertutup **16 konsep** sintetis (`32x32` RGB) agar seluruh siklus pelatihan, retrieval, dan ablasi dapat dijalankan ulang dalam hitungan detik di CPU lokal tanpa GPU. Konsep matematisnya identik, tetapi generalisasi ke foto dunia nyata tak terbatas membutuhkan korpus ratusan juta pasangan dan arsitektur Transformer skala besar.

---

## PART G — PELATIHAN MINIATURE DUAL ENCODER DI CPU

Pada `code/clip_miniature.py`, `MiniatureDualEncoder` menggabungkan `TinyImageEncoder` dan `TinyTextEncoder` dengan kepala proyeksi ke ruang laten bersama `32`-d dan suhu $\tau = 0.15$:

- **Total Parameter:** `36,704` parameter (bobot disimpan ke `experiments/clip_miniature.pt`).
- **Dinamika Pelatihan (35 Epoch, Adam lr=3e-3, Unique-Concept Batch):**
  - **Epoch 1 Loss:** `1.6808`
  - **Epoch 35 Loss:** `0.0717`
- **Evaluasi Penyelarasan Ruang Embedding pada `16` Konsep Uji Tak Terlihat:**
  - Rata-rata *Cosine Similarity* Pasangan Positif (diagonal): **`+0.9753`** (mendekati `1.0` sempurna)
  - Rata-rata *Cosine Similarity* Pasangan Negatif (off-diagonal): **`-0.0650`**
  - **Alignment Margin (Pos − Neg):** **`+1.0403`**
  - Akurasi Pencocokan 16-Arah pada Sampel Uji: **`100.00%`**

---

## PART H — EVALUASI CROSS-MODAL RETRIEVAL DUA ARAH

Berbeda dengan klasifikasi akurasi tunggal, sistem pencarian lintas modalitas dinilai menggunakan metrik peringkat (*Information Retrieval*) pada **`96` citra uji tak terlihat** (`6` variasi acak posisi/skala per konsep untuk `16` konsep):

1. **$\text{Recall}@k$:** Persentase kueri di mana pasangan target yang benar muncul di dalam **$k$ peringkat teratas** (Top-$k$).
2. **$\text{Precision}@k$:** Proporsi item di dalam $k$ hasil teratas yang memang relevan dengan kueri (khusus *Text-to-Image* ketika ada `6` gambar relevan per konsep di korpus uji).
3. **Median Rank:** Nilai tengah posisi peringkat pertama ditemukannya jawaban benar (skor ideal = `1.0`).

Hasil evaluasi aktual `code/cross_modal_evaluation.py` (murni pada data uji tak terlihat):

| Arah Penelusuran | Ukuran Kueri -> Kolam Kandidat | `Recall@1` | `Recall@3` | `Recall@5` | `Precision@6` | `Median Rank` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Image-to-Text (`I->T`)** | `96` Citra Uji -> `16` Teks Konsep | `100.00%` | `100.00%` | `100.00%` | — | `1.0` |
| **Text-to-Image (`T->I`)** | `16` Kueri Teks -> `96` Citra Uji | `100.00%` | `100.00%` | `100.00%` | `100.00%` | `1.0` |

Visualisasi matriks kesamaan $16\times 16$ (`code/similarity_matrix.py` $\to$ `experiments/similarity_matrix.png`) memperlihatkan garis diagonal menyala kuat (`+0.9827` untuk pasangan tepat `'red circle'`) dan nilai rendah/negatif di luar diagonal, serta gradasi semantik alami seperti `'red circle'` memiliki kesamaan lebih dekat dengan `'red square'` (`+0.4652`) dibanding konsep yang warna maupun bentuknya berbeda total.

---

## PART I — PIPELINE VISUAL QUESTION ANSWERING (VQA) DETERMINISTIK

Secara arsitektural, **Visual Question Answering (VQA)** memadukan pertanyaan teks bebas bersama gambar masukan untuk menghasilkan jawaban langsung:

```text
[Pertanyaan Teks] ---> Text Parser / Encoder ----\
                                                  +--> Joint Reasoning ---> Jawaban ("blue", "circle", "yes")
[Citra Input]     ---> Vision Feature Inspector -/
```

Script `code/vqa_concept_demo.py` mengimplementasikan **inspektor atribut geometris–warna deterministik**:
1. Menentukan warna dominan melalui rata-rata kanal RGB piksel non-latar (*foreground mask*).
2. Membedakan bentuk berdasarkan **rasio pengisian kotak pembatas** (*bounding-box fill ratio*, $A_{\text{objek}} / A_{\text{bbox}}$): sekitar `1.0` untuk persegi/persegi panjang (dibedakan lewat *aspect ratio*), `~0.785` ($\pi / 4$) untuk lingkaran, dan `~0.5` untuk segitiga.

> [!CAUTION]
> **Batas Jujur Sistem:** Demo pada `vqa_concept_demo.py` merupakan **simulator konsep VQA berbasis aturan geometris** guna mendemonstrasikan mekanisme *multimodal reasoning* secara deterministik di CPU, dan **bukan** model VQA generatif berbasis neural *Vision-Language Model* (VLM) yang mampu menjawab pertanyaan penalaran terbuka di foto dunia nyata.

---

## PART J — TANTANGAN TEKNIS & APLIKASI DUNIA NYATA

### 1. Tantangan Rekayasa Sistem Multimodal
- **Sinkronisasi Temporal & Laju Sampel:** Kamera menangkap `30 fps`, sementara sensor suhu IoT mengambil data setiap `2 detik`, dan log operator datang sporadis. Dibutuhkan *window alignment* sebelum fusi.
- **Ketidakhadiran Modalitas (*Missing Modalities*):** Di lapangan, kamera dapat tertutup debu atau koneksi sensor terputus. Teknik **Modality Dropout** saat pelatihan serta desain **Late / Hybrid Fusion** menjaga layanan tetap aman (*graceful degradation*).
- **Bias & Ketergantungan Modalitas Dominan:** Jika satu modalitas jauh lebih mudah diprediksi (seperti teks bersih pada dataset sintetis kita), jaringan cenderung "malas" mempelajari cabang gambar (*modality collapse*) kecuali diberi *regularization* terpisah.

### 2. Aplikasi Nyata (Konteks Industri, Pertanian Presisi, & IoT)
- **Pemeliharaan Prediktif Industri (IoT + Visi):** Menggabungkan kamera termal inframerah motor listrik dengan sinyal getaran akselerometer dan catatan historis teknisi.
- **Pertanian Presisi:** Mengaitkan citra daun drone multispektral dengan tingkat kelembaban tanah serta laporan hama berbasis teks untuk rekomendasi irigasi dan pemupukan lokal.
- **Sistem Pencarian Katalog E-Commerce & Arsip Medis:** Penelusuran dua arah antara foto radiologi/produk dengan deskripsi klinis atau kueri kata kunci pengguna.

---

## PART K — RINGKASAN, LATIHAN, MINI PROJECT & REFERENSI

### Ringkasan Utama Chapter 38
1. **Multimodal AI** mengatasi keterbatasan silo unimodal melalui penyelaras ruang fitur (*Alignment*) dan penggabungan sinyal (*Fusion*).
2. **Early Fusion** memadukan fitur di awal namun rentan terhadap *missing modality* (`25.00%` saat teks hilang), sedangkan **Late Fusion** paling tangguh (`43.75%` hanya citra; `100.00%` hanya teks), dan **Cross-Attention** memfokuskan token teks ke patch spasial relevan.
3. **Symmetric InfoNCE Loss** memaksimalkan kesamaan kosinus pasangan diagonal di hipersfer satuan, menyingkirkan *Modality Gap* (`0.2650` awal $\to$ `+0.9753` positif setelah 35 epoch, margin `+1.0403`).
4. **Unique-Concept Batch Sampling** sangat krusial pada batch kecil untuk mencegah *false negatives* yang merusak gradien kesamaan.
5. **Cross-Modal Retrieval** dievaluasi dengan metrik peringkat (`Recall@k`, `Precision@k`, `Median Rank`), bukan sekadar akurasi tunggal.

### Latihan Bertingkat & Mini Project
Panduan praktikum langkah demi langkah beserta 16 instruksi eksekusi Python, **5 soal latihan bertingkat (Dasar, Menengah, Lanjut) lengkap dengan pembahasannya**, serta **Mini Project "Building a Miniature Image-Text Retrieval System"** (16 artefak wajib & **Rubrik Penilaian 100 Poin**: `15 + 25 + 25 + 25 + 10`) tersedia lengkap pada menu **Praktik (`practice.md`)** dan notebook `notebooks/chapter_38_companion.ipynb`.

### Referensi Akademik Terverifikasi
1. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). *Multimodal Machine Learning: A Survey and Taxonomy*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). https://arxiv.org/abs/1705.09406
2. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML 2021. https://arxiv.org/abs/2103.00020
3. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. https://arxiv.org/abs/1706.03762
4. Li, J., Li, D., Xiong, C., & Hoi, S. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. ICML 2022. https://arxiv.org/abs/2201.12086
5. Antol, S., Agrawal, A., Lu, J., et al. (2015). *VQA: Visual Question Answering*. ICCV 2015. https://arxiv.org/abs/1505.00468

