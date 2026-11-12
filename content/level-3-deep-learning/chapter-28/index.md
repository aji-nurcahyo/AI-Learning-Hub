# Bab 28: Attention & Transformer Architecture

## Query-Key-Value Attention, Scaled Dot-Product, Multi-Head Attention, Positional Encoding, dan Rancang Bangun Blok Transformer Encoder-Decoder

---

## 1. Pendahuluan & Paradigma Baru Deep Learning

Pada Bab 27, kita telah mempelajari fondasi pemrosesan bahasa alami (*Natural Language Processing*) dan arsitektur sekuensial rekuren seperti Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), dan Gated Recurrent Units (GRU). Meskipun arsitektur rekuren berhasil memodelkan ketergantungan temporal untuk sekuens pendek, pemrosesan sekuensial dari kiri ke kanan (*step-by-step*) menghadirkan dua kelemahan fundamental yang tidak dapat dihindari:
1. **Ketidakmampuan Paralelisasi**: Perhitungan pada langkah waktu $t$ bergantung secara mutlak pada status tersembunyi sebelumnya $\mathbf{h}_{t-1}$, sehingga perangkat keras komputasi masif modern (seperti GPU dan TPU) terpaksa beroperasi secara serial $O(T)$.
2. **Bottleneck Informasi Jalur Jauh**: Informasi dari awal kalimat harus melewati rantai transformasi non-linier yang panjang untuk mencapai akhir kalimat, menyebabkan hilangnya detail semantik halus (*information compression bottleneck*).

Pada tahun 2017, tim peneliti Google Brain menerbitkan makalah seminal *"Attention Is All You Need"* (Vaswani et al., 2017) yang merevolusi lanskap kecerdasan buatan. Mereka mengusulkan arsitektur **Transformer** yang sepenuhnya meninggalkan mekanisme rekuren dan konvolusional, bertumpu sepenuhnya pada **Attention Mechanism** untuk menghubungkan seluruh token dalam sekuens secara simultan dan instan.

```
+---------------------------------------------------------------------------------------------------------+
|                                  PARADIGMA SEKUANSIAL VS PARADIGMA TRANSFORMER                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   A. PARADIGMA REKUREN (RNN / LSTM / GRU)                                                               |
|      Komputasi Serial Bertahap: Waktu Komputasi O(T)                                                    |
|                                                                                                         |
|      [x_1] ──► [ Cell_1 ] ──h_1──► [ Cell_2 ] ──h_2──► [ Cell_3 ] ──h_3──► [ Cell_4 ]                  |
|                     ▲                   ▲                   ▲                   ▲                       |
|                     │                   │                   │                   │                       |
|                   [x_1]               [x_2]               [x_3]               [x_4]                     |
|      * Masalah: Harus menunggu Cell_1 selesai sebelum Cell_2 dapat diproses. Tidak bisa paralel!         |
|      * Informasi x_1 teredam dan pudar saat mencapai Cell_4 (Jalur O(T)).                               |
|                                                                                                         |
|   B. PARADIGMA TRANSFORMER (ALL-TO-ALL SELF-ATTENTION)                                                   |
|      Komputasi Simultan Paralel: Jalur Komputasi O(1)                                                   |
|                                                                                                         |
|                   +-------------------------------------------------------------+                       |
|                   |           MULTI-HEAD SELF-ATTENTION TENSOR ENGINE           |                       |
|                   +-------------------------------------------------------------+                       |
|                     ▲                     ▲                     ▲             ▲                         |
|                     │ (Koneksi Langsung)  │ (Koneksi Langsung)  │             │                         |
|                   [x_1]                 [x_2]                 [x_3]         [x_4]                       |
|      * Keunggulan: Seluruh token diproses serentak O(1) langkah waktu via aljabar matriks.              |
|      * Jarak interaksi antar sembarang pasangan token selalu tepat 1 langkah O(1).                      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Evolusi: Dari Bahdanau Attention ke Self-Attention

Sebelum Transformer lahir, mekanisme atensi pertama kali dirumuskan oleh Bahdanau et al. (2015) sebagai ekstensi pada arsitektur Encoder-Decoder RNN untuk penerjemahan mesin.

### A. Bottleneck Vektor Konteks Statis
Pada model Seq2Seq tradisional (Sutskever et al., 2014), Encoder mereduksi seluruh kalimat masukan dengan panjang sembarang menjadi satu vektor tersembunyi tunggal berdimensi tetap $\mathbf{c} = \mathbf{h}_T$. Meringkas paragraf berisi 50 kata ke dalam vektor 512 angka menyebabkan hilangnya konteks secara masif (*information loss*).

### B. Mekanisme Bahdanau (Additive Attention)
Bahdanau et al. mengusulkan bahwa Decoder tidak perlu bergantung pada satu vektor statis. Pada setiap langkah pembangkitan kata ke-$i$, Decoder menghasilkan vektor konteks dinamis $\mathbf{c}_i$ sebagai penjumlahan berbobot dari seluruh representasi tersembunyi Encoder $(\mathbf{h}_1, \dots, \mathbf{h}_{T_{\text{enc}}})$:

$$\mathbf{c}_i = \sum_{j=1}^{T_{\text{enc}}} \alpha_{ij} \mathbf{h}_j$$

Bobot atensi $\alpha_{ij}$ dihitung melalui normalisasi Softmax dari skor keselarasan (*alignment score*) $e_{ij}$:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_{\text{enc}}} \exp(e_{ik})}, \quad e_{ij} = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{i-1} + \mathbf{U}_a \mathbf{h}_j)$$

di mana $\mathbf{s}_{i-1}$ adalah status Decoder dan $\mathbf{h}_j$ adalah status Encoder.

### C. Lompatan Vaswani et al.: Self-Attention Tanpa RNN
Meskipun Bahdanau Attention memecahkan bottleneck representasi statis, model dasarnya tetaplah RNN yang berjalan serial. Terobosan Vaswani et al. (2017) adalah menyadari bahwa **mekanisme atensi itu sendiri sudah cukup kuat untuk merepresentasikan struktur bahasa**, tanpa memerlukan RNN sama sekali. Konsep ini melahirkan **Self-Attention** (Atensi-Diri), di mana setiap token dalam sebuah kalimat menghitung relasi langsung terhadap seluruh token lainnya dalam kalimat yang sama.

---

## 3. Formulasi Matematika: Query, Key, dan Value

Mekanisme atensi modern diformulasikan berdasarkan abstraksi sistem temu-kembali informasi (*information retrieval*), serupa dengan pencarian di mesin pencari atau basis data relasional.

```
+---------------------------------------------------------------------------------------------------------+
|                                 ANALOGI QUERY, KEY, DAN VALUE (SISTEM PUSTAKA)                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. QUERY (Q)  : Kata pencarian yang Anda ketikkan ("Apa itu neural network?").                        |
|                                                                                                         |
|   2. KEY (K)    : Label atau judul buku pada katalog perpustakaan ("Buku AI", "Buku Memasak").          |
|                   -> Komputer menghitung skor kecocokan (Simpangan / Dot-Product) antara Q dan K.       |
|                                                                                                         |
|   3. VALUE (V)  : Isi konten aktual di dalam buku yang sesuai.                                          |
|                   -> Informasi yang diambil merupakan rata-rata berbobot dari V berdasarkan skor Q & K. |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Proyeksi Ruang Vektor
Diberikan matriks representasi masukan sekuens $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$, di mana $T$ adalah panjang sekuens dan $d_{\text{model}}$ adalah dimensi fitur tiap token. Matriks Query ($\mathbf{Q}$), Key ($\mathbf{K}$), dan Value ($\mathbf{V}$) dibentuk melalui transformasi linier terpelajari:

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

di mana $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ dan $\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$. Dalam implementasi standar, umumnya dipilih $d_k = d_v = d_{\text{model}} / h$ (dengan $h$ adalah jumlah kepala atensi).

```
+---------------------------------------------------------------------------------------------------------+
|                                    PROSES TRANSFORMASI LINIER Q, K, V                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Matriks Masukan X            Bobot Proyeksi                    Matriks Representasi                   |
|                                                                                                         |
|   +---------------+            +---------------+                 +---------------+                      |
|   |               |            |    W_Q        |                 |      Q        |                      |
|   |   ( T x d )   |     x      |   ( d x d_k ) |       =         |   ( T x d_k ) |                      |
|   +---------------+            +---------------+                 +---------------+                      |
|                                                                                                         |
|   +---------------+            +---------------+                 +---------------+                      |
|   |               |            |    W_K        |                 |      K        |                      |
|   |   ( T x d )   |     x      |   ( d x d_k ) |       =         |   ( T x d_k ) |                      |
|   +---------------+            +---------------+                 +---------------+                      |
|                                                                                                         |
|   +---------------+            +---------------+                 +---------------+                      |
|   |               |            |    W_V        |                 |      V        |                      |
|   |   ( T x d )   |     x      |   ( d x d_v ) |       =         |   ( T x d_v ) |                      |
|   +---------------+            +---------------+                 +---------------+                      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Scaled Dot-Product Attention & Analisis Variansi $\sqrt{d_k}$

Setelah matriks $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ terbentuk, fungsi atensi mengevaluasi kesamaan antara setiap vektor Query terhadap seluruh Key melalui operasi perkalian titik (*dot product*):

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

```
+---------------------------------------------------------------------------------------------------------+
|                                PIPELINE SCALED DOT-PRODUCT ATTENTION                                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|          Q                  K                                                                           |
|          │                  │                                                                           |
|          └───► [ MatMul ] ◄─┘  ──► Skor S = Q K^T  (Dimensi: T_q x T_k)                                 |
|                     │                                                                                   |
|                     ▼                                                                                   |
|              [ Scale 1/sqrt(d_k) ]                                                                      |
|                     │                                                                                   |
|                     ▼                                                                                   |
|               [ Mask (Opsional) ]  ──► Gantikan posisi ilegal dengan -inf                               |
|                     │                                                                                   |
|                     ▼                                                                                   |
|                [ Softmax ]         ──► Matriks Bobot Probabilitas A (Baris bernilai total 1.0)          |
|                     │                                                                                   |
|                     ▼                                                                                   |
|          A ───────► [ MatMul ] ◄── V                                                                    |
|                           │                                                                             |
|                           ▼                                                                             |
|                      [ Output O ]  ──► Konteks Terbobot (Dimensi: T_q x d_v)                            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Pembuktian Teoretis: Mengapa Dibagi $\sqrt{d_k}$?
Mengapa hasil kali titik harus diskalakan dengan $\frac{1}{\sqrt{d_k}}$? Pertimbangkan dua vektor acak $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ yang elemen-elemennya berdistribusi independen dengan rata-rata 0 dan variansi 1:

$$\mathbb{E}[q_i] = \mathbb{E}[k_j] = 0, \quad \operatorname{Var}(q_i) = \operatorname{Var}(k_j) = 1$$

Hasil kali titik keduanya adalah skalar:

$$S = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

Nilai ekspektasinya adalah:

$$\mathbb{E}[S] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

Variansinya adalah:

$$\operatorname{Var}(S) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k} \left(\mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2\right) = \sum_{i=1}^{d_k} (1 \cdot 1 - 0) = d_k$$

Artinya, **variansi hasil kali titik bertumbuh secara linier sebanding dengan dimensi $d_k$**. Untuk dimensi besar seperti $d_k = 64$ atau $d_k = 128$, simpangan baku $\sigma = \sqrt{d_k} = 8$ hingga $11.3$. Hal ini menyebabkan skor-skor dot product memiliki magnitudo yang sangat besar (misal $+35$ atau $-40$).

Ketika angka-angka ekstrem ini diumpankan ke fungsi Softmax:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Output Softmax akan menjadi distribusi yang sangat runcing (*extremely peaky / one-hot distribution*), di mana satu nilai bernilai $1.0$ dan sisanya mendekati $0.0$. Pada daerah saturasi ini, turunan gradien Softmax mendekati nol:

$$\frac{\partial \text{softmax}(z_i)}{\partial z_j} = \text{softmax}(z_i)(\delta_{ij} - \text{softmax}(z_j)) \approx 0$$

Gradien menghilang (*vanishing gradient*), dan optimasi model melalui *backpropagation* akan macet total. Dengan membagi skor dengan $\sqrt{d_k}$, variansi skor ternormalisasi kembali ke $1.0$:

$$\operatorname{Var}\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{\operatorname{Var}(\mathbf{q} \cdot \mathbf{k})}{d_k} = \frac{d_k}{d_k} = 1.0$$

Gradien tetap mengalir lancar dalam rentang fungsi Softmax yang responsif.

---

## 5. Stabilitas Numerik Softmax: Shift-Max Trick

Dalam implementasi perangkat keras komputer menggunakan representasi bilangan pecahan *floating-point* (IEEE 754), fungsi eksponensial $e^x$ bertumbuh sangat cepat. Untuk tipe data 32-bit *single-precision* (`float32`), batas atas representasi bilangan adalah $\approx 3.4 \times 10^{38}$. Nilai $x > 88.7$ akan langsung memicu luapan aritmetika (*overflow* / `inf`). Untuk 64-bit (`float64`), nilai $x > 709.78$ akan memicu luapan serupa.

Ketika luapan terjadi:

$$\text{softmax}([1000, 1005, 995]) = \left[\frac{e^{1000}}{\sum e^z}, \frac{e^{1005}}{\sum e^z}, \frac{e^{995}}{\sum e^z}\right] = \left[\frac{\infty}{\infty}, \frac{\infty}{\infty}, \frac{\infty}{\infty}\right] = [\text{NaN}, \text{NaN}, \text{NaN}]$$

Seluruh bobot atensi hancur menjadi `NaN` (*Not a Number*).

### Bukti Identitas Shift-Max Trick
Stabilitas numerik dijamin dengan mengurangkan setiap elemen input dengan nilai maksimum dari vektor tersebut sebelum eksponensiasi:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^n e^{z_j - \max(\mathbf{z})}}$$

**Pembuktian Kesetaraan Matematis**:

$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_j (e^{z_j} \cdot e^{-c})} = \frac{e^{-c} \cdot e^{z_i}}{e^{-c} \cdot \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Dengan memilih $c = \max(\mathbf{z})$, nilai eksponen terbesar yang pernah dihitung adalah $e^{\max(\mathbf{z}) - \max(\mathbf{z})} = e^0 = 1.0$. Luapan (*overflow*) secara matematis mustahil terjadi, dan penyebut dijamin minimal bernilai $\ge 1.0$, mencegah terjadinya pembagian dengan nol (*underflow*).

---

## 6. Self-Attention vs Cross-Attention

Dalam arsitektur Transformer, mekanisme atensi digunakan dalam dua konfigurasi fundamental:

```
+---------------------------------------------------------------------------------------------------------+
|                                SELF-ATTENTION VS CROSS-ATTENTION                                        |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   A. SELF-ATTENTION (Atensi-Diri)                                                                       |
|      * Sumber Q, K, V berasal dari sekuens yang SAMA.                                                   |
|      * Panjang sekuens simetris: T_q = T_k = T.                                                         |
|      * Digunakan pada: Transformer Encoder & Decoder Self-Attention.                                    |
|                                                                                                         |
|         X ──┬──► [ W_Q ] ──► Q                                                                          |
|             ├──► [ W_K ] ──► K   ──► Matriks Skor Berdimensi (T x T)                                    |
|             └──► [ W_V ] ──► V                                                                          |
|                                                                                                         |
|   B. CROSS-ATTENTION (Atensi-Silang)                                                                    |
|      * Query (Q) berasal dari DECODER (target).                                                         |
|      * Key (K) dan Value (V) berasal dari MEMORI ENCODER (sumber).                                       |
|      * Panjang sekuens bisa ASIMETRIS: T_q = T_dec, T_k = T_v = T_enc.                                  |
|      * Digunakan pada: Lapisan kedua Transformer Decoder untuk memetakan bahasa asal ke bahasa tujuan.  |
|                                                                                                         |
|         X_target (Decoder)  ──► [ W_Q ] ──► Q (T_dec x d_k)                                             |
|                                                                                                         |
|         X_memory (Encoder)  ──┬──► [ W_K ] ──► K (T_enc x d_k) ──► Matriks Skor (T_dec x T_enc)         |
|                               └──► [ W_V ] ──► V (T_enc x d_v)                                          |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 7. Mekanisme Masking: Padding & Causal Masks

Mekanisme atensi menghubungkan seluruh token tanpa batasan struktural. Namun, ada situasi di mana interaksi tertentu harus dicegah secara eksplisit. Hal ini diwujudkan dengan menambahkan matriks masker $\mathbf{M}$ ke skor sebelum fungsi Softmax:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

di mana:
- $\mathbf{M}_{ij} = 0$, jika koneksi antara token $i$ dan $j$ diperbolehkan.
- $\mathbf{M}_{ij} = -\infty$ (atau bilangan negatif raksasa seperti $-10^9$), jika koneksi dilarang.

Karena $\lim_{x \to -\infty} e^x = 0$, posisi yang dimasker akan memperoleh bobot atensi tepat $0.0000$ setelah melewati Softmax.

```
+---------------------------------------------------------------------------------------------------------+
|                                    JENIS-JENIS MASKER ATENSI                                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. PADDING MASK (Penyaringan Token Bantalan)                                                          |
|      Dalam satu batch, kalimat memiliki panjang berbeda. Kalimat pendek diberi token <pad>.             |
|      Token <pad> tidak memiliki arti semantik dan dilarang menyerap atau mendistribusikan atensi.       |
|                                                                                                         |
|      K:  "Saya"  "suka"  "AI"   <pad>                                                                   |
|      Q:                                                                                                 |
|      "Saya"  [  1      1      1     0  ]   <-- Kolom <pad> dipaksa bernilai 0                           |
|      "suka"  [  1      1      1     0  ]                                                                |
|      "AI"    [  1      1      1     0  ]                                                                |
|      <pad>   [  0      0      0     0  ]                                                                |
|                                                                                                         |
|   2. CAUSAL / LOOK-AHEAD MASK (Pemodelan Bahasa Autoregresif)                                           |
|      Pada Decoder, saat memprediksi kata pada langkah t, model dilarang melihat kata masa depan t+1.    |
|      Diwujudkan melalui matriks segitiga bawah (Lower Triangular Matrix):                               |
|                                                                                                         |
|      K:   "Hari"  "ini"   "cerah"  "sekali"                                                             |
|      Q:                                                                                                 |
|      "Hari"    [  1      0       0        0   ]  <-- Hanya boleh melihat "Hari"                         |
|      "ini"     [  1      1       0        0   ]  <-- Boleh melihat "Hari", "ini"                        |
|      "cerah"   [  1      1       1        0   ]  <-- Boleh melihat "Hari", "ini", "cerah"               |
|      "sekali"  [  1      1       1        1   ]  <-- Boleh melihat seluruh kalimat                      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 8. Multi-Head Attention (MHA): Teori Proyeksi Subspace Paralel

Daripada hanya menghitung satu fungsi atensi tunggal dengan dimensi $d_{\text{model}}$, Vaswani et al. menemukan bahwa jauh lebih menguntungkan untuk memproyeksikan Query, Key, dan Value secara linier sebanyak $h$ kali menggunakan proyeksi terpelajari yang berbeda ke dimensi $d_k, d_k, d_v$.

$$\text{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O$$

di mana:

$$\text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V)$$

```
+---------------------------------------------------------------------------------------------------------+
|                                ARSITEKTUR MULTI-HEAD ATTENTION (MHA)                                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                                  [ Multi-Head Output ]                                                  |
|                                            ▲                                                            |
|                                            │                                                            |
|                                    [ Linear (W_O) ]                                                     |
|                                            ▲                                                            |
|                                            │                                                            |
|                             [ Concat (head_1 ... head_h) ]                                              |
|                                            ▲                                                            |
|                                            │                                                            |
|                   ┌────────────────────────┼────────────────────────┐                                   |
|                   │                        │                        │                                   |
|             [ Head 1 ]                [ Head 2 ]               [ Head h ]                               |
|             Scaled Dot-Product        Scaled Dot-Product       Scaled Dot-Product                       |
|                   ▲                        ▲                        ▲                                   |
|            ┌──────┼──────┐          ┌──────┼──────┐          ┌──────┼──────┐                            |
|            Q1     K1     V1         Q2     K2     V2         Qh     Kh     Vh                           |
|            │      │      │          │      │      │          │      │      │                            |
|          [W_1Q] [W_1K] [W_1V]     [W_2Q] [W_2K] [W_2V]     [W_hQ] [W_hK] [W_hV]                         |
|            ▲      ▲      ▲          ▲      ▲      ▲          ▲      ▲      ▲                            |
|            └──────┼──────┴──────────┴──────┼──────┴──────────┴──────┼──────┘                            |
|                   │                        │                        │                                   |
|                   Q                        K                        V                                   |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Mengapa Multi-Head Mengungguli Single-Head?
1. **Diferensiasi Subspace Representasi**: Satu kepala atensi mungkin terfokus pada keselarasan sintaktis (misalnya menghubungkan kata kerja dengan objek penderita), kepala kedua fokus pada hubungan anafora/kata ganti (misalnya menghubungkan kata *"ia"* dengan nama *"Budi"* pada awal paragraf), sedangkan kepala ketiga fokus pada relasi semantik lokal.
2. **Pencegahan Dominasi Rata-Rata**: Pada single-head berdimensi 512, informasi dari berbagai aspek linguistik akan dirata-ratakan menjadi satu skor tunggal, mengaburkan detail relasi yang halus.
3. **Efisiensi Komputasi**: Karena dimensi setiap kepala adalah $d_k = d_{\text{model}} / h$, total biaya komputasi untuk $h$ kepala identik dengan biaya komputasi single-head berskala penuh:
   $$h \times O(T^2 \cdot d_k) = O(T^2 \cdot (h \cdot d_k)) = O(T^2 \cdot d_{\text{model}})$$

---

## 9. Pembedahan PyTorch `nn.MultiheadAttention`

Dalam pustaka produksi PyTorch, MHA dikemas dalam kelas teroptimasi `torch.nn.MultiheadAttention`:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(
    embed_dim=512,       # d_model
    num_heads=8,         # h
    dropout=0.1,         # Rasio dropout atensi
    batch_first=True     # Bentuk tensor (Batch, SeqLen, Dim)
)
```

### Efisiensi Memori: `in_proj_weight`
Jika kita memeriksa parameter internal dari `nn.MultiheadAttention`, PyTorch tidak membuat tiga variabel matriks terpisah untuk $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$. Sebaliknya, PyTorch menggabungkan ketiganya ke dalam satu matriks besar tunggal bernama `in_proj_weight` berukuran $(3 \cdot d_{\text{model}} \times d_{\text{model}})$ dan `in_proj_bias` berukuran $(3 \cdot d_{\text{model}})$.

Penggabungan ini memungkinkan kartu grafis melakukan satu operasi perkalian matriks besar (*fused GEMM kernel*), yang jauh lebih cepat daripada tiga operasi perkalian terpisah.

### Rumus Analitis Jumlah Parameter MHA:
$$\text{Parameter In-Projection} = 3 \cdot (d_{\text{model}} \cdot d_{\text{model}} + d_{\text{model}})$$
$$\text{Parameter Out-Projection} = d_{\text{model}} \cdot d_{\text{model}} + d_{\text{model}}$$
$$\text{Total Parameter} = 4 \cdot d_{\text{model}}^2 + 4 \cdot d_{\text{model}}$$

*Contoh Perhitungan*:
Untuk $d_{\text{model}} = 16$:
$$\text{Total} = 4 \cdot (16)^2 + 4 \cdot 16 = 4 \cdot 256 + 64 = 1.024 + 64 = 1.088 \text{ parameter}$$
Nilai ini persis cocok secara analitis dengan inspeksi `sum(p.numel() for p in mha.parameters())` di PyTorch.

---

## 10. Positional Encoding Sinusoidal (Vaswani et al., 2017)

Salah satu kelemahan intrinsik dari operasi Self-Attention murni adalah sifatnya yang **invarian permutasi** (*permutation invariant*).

### Bukti Permutation Invariance Self-Attention
Misalkan $\mathbf{P}$ adalah matriks permutasi (matriks baris yang mengacak urutan token). Jika masukan dibalik menjadi $\mathbf{X}' = \mathbf{P} \mathbf{X}$, maka skor atensi baru adalah:

$$\mathbf{S}' = (\mathbf{P} \mathbf{Q}) (\mathbf{P} \mathbf{K})^T = \mathbf{P} \mathbf{Q} \mathbf{K}^T \mathbf{P}^T = \mathbf{P} \mathbf{S} \mathbf{P}^T$$

Matriks bobot atensi baru adalah $\mathbf{A}' = \mathbf{P} \mathbf{A} \mathbf{P}^T$, dan output akhir adalah:

$$\mathbf{O}' = \mathbf{A}' (\mathbf{P} \mathbf{V}) = (\mathbf{P} \mathbf{A} \mathbf{P}^T)(\mathbf{P} \mathbf{V}) = \mathbf{P} \mathbf{A} \mathbf{I} \mathbf{V} = \mathbf{P} (\mathbf{A} \mathbf{V}) = \mathbf{P} \mathbf{O}$$

Artinya: **Mengacak urutan masukan hanya akan mengacak urutan luaran secara identik**. Model tidak memiliki mekanisme bawaan untuk membedakan antara *"anjing menggigit manusia"* dan *"manusia menggigit anjing"*.

### Formulasi Positional Encoding Sinusoidal
Untuk menyuntikkan informasi urutan tanpa menggunakan rekurensi, Vaswani et al. merumuskan **Sinusoidal Positional Encoding** berbasis kombinasi fungsi sinus dan kosinus dengan frekuensi geometris:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

di mana:
- $pos \in [0, T-1]$ adalah indeks posisi kata dalam kalimat.
- $i \in [0, d_{\text{model}}/2 - 1]$ adalah indeks dimensi fitur.

Vektor $PE$ ini langsung dijumlahkan dengan vektor token embedding:

$$\mathbf{X}_{\text{input}} = \text{Embedding}(\mathbf{X}) + PE$$

```
+---------------------------------------------------------------------------------------------------------+
|                                VISUALISASI FREKUENSI POSITIONAL ENCODING                                |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Dimensi Rendah (i=0) : Frekuensi Tinggi (Panjang gelombang pendek: 2*pi)                             |
|                          Gelombang berosilasi sangat cepat antar posisi berurutan.                      |
|                          -> Menyandikan relasi jarak sangat dekat (posisi n vs n+1).                    |
|                                                                                                         |
|   Dimensi Tinggi (i=d) : Frekuensi Sangat Rendah (Panjang gelombang raksasa: 10000 * 2*pi)             |
|                          Gelombang berubah sangat lambat melintasi puluhan ribu token.                  |
|                          -> Menyandikan posisi absolut skala besar dalam dokumen.                       |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Bukti Identitas Pergeseran Linier
Mengapa memilih fungsi sinus dan kosinus? Karena untuk setiap pergeseran posisi tetap $k$, $PE_{pos+k}$ dapat dinyatakan sebagai **transformasi linier** dari $PE_{pos}$.
Berdasarkan rumus penjumlahan trigonometri:

$$\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)$$

$$\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)$$

Dengan memisahkan $\alpha = pos \cdot \omega_i$ dan $\beta = k \cdot \omega_i$, vektor posisi $pos+k$ dapat diperoleh melalui rotasi matriks 2D linier dari vektor pada posisi $pos$:

$$\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(k \omega_i) & \sin(k \omega_i) \\ -\sin(k \omega_i) & \cos(k \omega_i) \end{pmatrix} \begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}$$

Hal ini memudahkan model berbasis atensi untuk belajar memperhatikan posisi relatif secara konsisten.

---

## 11. Blok Pembangun Transformer Encoder

Satu blok utuh Transformer Encoder terdiri dari dua sub-lapisan bertingkat yang diapit oleh koneksi residu (*residual connection*) dan normalisasi lapisan (*Layer Normalization*):

```
+---------------------------------------------------------------------------------------------------------+
|                                    BLOK TRANSFORMER ENCODER LENGKAP                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                                         Output Blok Encoder                                             |
|                                                  ▲                                                      |
|                                                  │                                                      |
|                                         [ Add & LayerNorm ] ◄──────────┐                                |
|                                                  ▲                     │ (Skip Connection 2)            |
|                                                  │                     │                                |
|                                      [ Feed-Forward Network ]          │                                |
|                                                  ▲                     │                                |
|                                                  ├─────────────────────┘                                |
|                                                  │                                                      |
|                                         [ Add & LayerNorm ] ◄──────────┐                                |
|                                                  ▲                     │ (Skip Connection 1)            |
|                                                  │                     │                                |
|                                     [ Multi-Head Self-Attention ]      │                                |
|                                                  ▲                     │                                |
|                                                  ├─────────────────────┘                                |
|                                                  │                                                      |
|                                          Input Token/Blok                                               |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### A. Residual Skip Connection (He et al., 2016)
Setiap sub-lapisan dirumuskan sebagai:

$$\mathbf{x}_{\text{sub}} = \mathbf{x} + \text{Sublayer}(\mathbf{x})$$

Koneksi pintas ini menyediakan jalan bebas hambatan bagi gradien selama *backpropagation*:

$$\frac{\partial \mathbf{x}_{\text{sub}}}{\partial \mathbf{x}} = \mathbf{I} + \frac{\partial \text{Sublayer}(\mathbf{x})}{\partial \mathbf{x}}$$

Meskipun gradien pada sub-lapisan menyusut, suku identitas $\mathbf{I}$ menjamin aliran gradien tidak akan pernah mati, memungkinkan pelatihan model hingga puluhan bahkan ratusan lapisan.

### B. Layer Normalization (Ba, Kiros, & Hinton, 2016)
Berbeda dengan Batch Normalization yang menormalkan statistik di seluruh sampel dalam satu batch, **Layer Normalization** menormalkan fitur secara independen untuk setiap token di sepanjang dimensi fitur:

$$\mu = \frac{1}{d} \sum_{k=1}^d x_k, \quad \sigma^2 = \frac{1}{d} \sum_{k=1}^d (x_k - \mu)^2$$

$$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

di mana $\boldsymbol{\gamma}$ dan $\boldsymbol{\beta}$ adalah parameter skala dan pergeseran yang dapat dipelajari.

### C. Position-Wise Feed-Forward Network (FFN)
FFN terdiri dari dua transformasi linier dengan aktivasi non-linier (ReLU atau GELU) di antaranya, diaplikasikan secara terpisah dan identik pada setiap posisi token:

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

Umumnya dimensi tersembunyi dibuat 4 kali lebih besar: $d_{\text{ff}} = 4 \cdot d_{\text{model}}$.

---

## 12. Blok Pembangun Transformer Decoder

Transformer Decoder memiliki struktur yang lebih kompleks dengan **tiga sub-lapisan**:

```
+---------------------------------------------------------------------------------------------------------+
|                                    BLOK TRANSFORMER DECODER LENGKAP                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                                         Output Blok Decoder                                             |
|                                                  ▲                                                      |
|                                                  │                                                      |
|                                         [ Add & LayerNorm ] ◄──────────┐                                |
|                                                  ▲                     │ (Skip Connection 3)            |
|                                                  │                     │                                |
|                                      [ Feed-Forward Network ]          │                                |
|                                                  ▲                     │                                |
|                                                  ├─────────────────────┘                                |
|                                                  │                                                      |
|                                         [ Add & LayerNorm ] ◄──────────┐                                |
|                                                  ▲                     │ (Skip Connection 2)            |
|                                                  │                     │                                |
|    Memori K, V dari Encoder ────────► [ Multi-Head Cross-Attention ]    │                                |
|                                                  ▲                     │                                |
|                                                  ├─────────────────────┘                                |
|                                                  │                                                      |
|                                         [ Add & LayerNorm ] ◄──────────┐                                |
|                                                  ▲                     │ (Skip Connection 1)            |
|                                                  │                     │                                |
|                                     [ Masked Multi-Head Attention ]    │ (Causal Masking)               |
|                                                  ▲                     │                                |
|                                                  ├─────────────────────┘                                |
|                                                  │                                                      |
|                                         Input Target Decoder                                            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

1. **Masked Multi-Head Self-Attention**: Menerapkan *Causal Masking* segitiga bawah sehingga prediksi posisi $i$ hanya boleh bergantung pada token sebelumnya $< i$.
2. **Multi-Head Cross-Attention**: Menghubungkan Query Decoder dengan Key dan Value yang dihasilkan oleh tumpukan Encoder teratas.
3. **Position-Wise Feed-Forward Network**: Melakukan pemetaan fitur non-linier identik dengan Encoder.

---

## 13. Arsitektur Seq2Seq Transformer Utuh

Ketika tumpukan $N$ lapisan Encoder digabungkan dengan $N$ lapisan Decoder, terbentuk arsitektur Sequence-to-Sequence (Seq2Seq) yang utuh:

```
+---------------------------------------------------------------------------------------------------------+
|                                  ARSITEKTUR LENGKAP SEQ2SEQ TRANSFORMER                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|     MASUKAN SUMBER (Bahasa Asal)                         KELUARAN TARGET (Bahasa Tujuan)                 |
|     "Machine learning is powerful"                       "Pembelajaran mesin sangat hebat"              |
|                   │                                                      │                              |
|                   ▼                                                      ▼                              |
|          [ Token Embedding ]                                    [ Token Embedding ]                     |
|                   │                                                      │                              |
|                   ▼                                                      ▼                              |
|        [ + Positional Encoding ]                              [ + Positional Encoding ]                 |
|                   │                                                      │                              |
|                   ▼                                                      ▼                              |
|       ┌───────────────────────┐                              ┌───────────────────────┐                  |
|       │  ENCODER BLOCK (x N)  │                              │  DECODER BLOCK (x N)  │                  |
|       │  - Self-Attention     │ ── K, V Representasi ──────► │  - Masked Self-Attn   │                  |
|       │  - Add & LayerNorm    │                              │  - Cross-Attention    │                  |
|       │  - Feed-Forward Net   │                              │  - Add & LayerNorm    │                  |
|       │  - Add & LayerNorm    │                              │  - Feed-Forward Net   │                  |
|       └───────────────────────┘                              └───────────────────────┘                  |
|                   │                                                      │                              |
|                   └──────────────────────────────────────────────────────┤                              |
|                                                                          ▼                              |
|                                                                 [ Linear Projection ]                   |
|                                                                          │                              |
|                                                                          ▼                              |
|                                                                 [ Softmax Probabilities ]               |
|                                                                          │                              |
|                                                                          ▼                              |
|                                                                  Prediksi Token Kata                    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 14. Taksonomi Tiga Keluarga Besar Arsitektur Transformer

Dalam perkembangannya, arsitektur dasar Transformer bercabang menjadi tiga keluarga besar yang mendominasi seluruh ekosistem AI modern:

| Karakteristik | 1. Encoder-Only | 2. Decoder-Only | 3. Encoder-Decoder |
|---|---|---|---|
| **Pola Atensi** | Dua arah penuh (*Fully Bidirectional*) | Searah kausal (*Left-to-Right Causal*) | Dua arah pada Encoder, Searah pada Decoder |
| **Model Representatif** | **BERT**, RoBERTa, DeBERTa | **GPT-1/2/3/4**, LLaMA, Mistral | **T5**, BART, Asli Vaswani (2017) |
| **Tujuan Pra-Pelatihan** | Masked Language Modeling (MLM) | Next-Token Prediction (Causal LM) | Span Corruption / Denoising |
| **Kasus Penggunaan Utama** | Klasifikasi teks, NER, ekstraksi fitur | Generasi teks bebas, *chat*, penalaran *prompting* | Terjemahan bahasa, peringkasan teks |

---

## 15. Analisis Kompleksitas Komputasi & Memori: $O(T^2)$ vs $O(T)$

Keberhasilan Transformer ditopang oleh paralelisasi masif, tetapi kelemahan utamanya adalah kompleksitas komputasi dan memori kuadratik terhadap panjang sekuens $T$:

| Aspek Komputasi | Recurrent Neural Network (RNN/LSTM) | Self-Attention (Transformer) |
|---|:---:|:---:|
| **Kompleksitas Komputasi per Layer** | $O(T \cdot d^2)$ | $O(T^2 \cdot d + T \cdot d^2)$ |
| **Operasi Minimum Sekuensial (Paralel)** | $O(T)$ (Serial bertahap) | **$O(1)$ (Simultan Paralel)** |
| **Jarak Jalur Maksimum Antar-Token** | $O(T)$ (Teredam peluruhan waktu) | **$O(1)$ (Koneksi Langsung Instan)** |
| **Kebutuhan Memori Matriks Atensi** | $O(d)$ | **$O(T^2)$ (Matriks Skor Kuadratik)** |

### Bottleneck Kuadratik Sekuens Panjang:
Matriks atensi berdimensi $(T \times T)$. Jika panjang sekuens dinaikkan $10\times$ (misal dari $1.000$ ke $10.000$ token), alokasi memori melonjak **$100\times$ lipat**!
Untuk mengatasi kendala ini, komunitas riset melahirkan inovasi perangkat keras dan algoritma mutakhir:
1. **FlashAttention (Dao et al., 2022)**: Menghitung Softmax secara bertahap menggunakan *tiling* di SRAM GPU berkecepatan tinggi tanpa pernah menyimpan matriks $T \times T$ secara utuh di memori HBM, memangkas memori menjadi linier $O(T)$.
2. **Linear Attention & State Space Models (Mamba / S4)**: Merumuskan mekanisme atensi berbasis kernel atau perambatan status kontinu yang sepenuhnya memulihkan kompleksitas waktu $O(T)$ untuk sekuens jutaan token.

---

## 16. Ringkasan & Diskusi Reflektif

### Intisari Bab:
1. **Attention Is All You Need**: Menghilangkan rekurensi dan konvolusi, menggantikannya dengan Self-Attention paralel $O(1)$ yang menghubungkan seluruh pasangan kata secara instan.
2. **Skalasi $\sqrt{d_k}$ & Stabilitas Softmax**: Mencegah lonjakan variansi hasil kali titik yang memicu kepunahan gradien, diperkuat oleh *shift-max trick* yang kebal terhadap *floating-point overflow*.
3. **Multi-Head Attention**: Memungkinkan model membagi representasi ke dalam beberapa subspace independen untuk menangkap beragam relasi sintaksis dan semantik secara bersamaan.
4. **Positional Encoding Sinusoidal**: Mengatasi sifat invarian permutasi dari Self-Attention melalui kombinasi fungsi trigonometri berfrekuensi geometris.
5. **Blok Encoder-Decoder**: Mengintegrasikan MHA, Residual Skip Connection, Layer Normalization, dan Feed-Forward Network untuk membentuk fondasi model bahasa skala besar.

### Pertanyaan Reflektif:
1. Mengapa penambahan Positional Encoding dilakukan melalui operasi penjumlahan linier ($x + PE$) dan bukan penggabungan tensor (*concatenation*)?
2. Bagaimana mekanisme Cross-Attention pada Decoder memastikan bahwa urutan kata bahasa sasaran tetap selaras dengan makna kalimat bahasa sumber?
3. Mengapa hampir seluruh model bahasa generasi teks mutakhir saat ini (seperti LLaMA, GPT-4, dan Claude) menggunakan arsitektur *Decoder-Only* alih-alih arsitektur *Encoder-Decoder* asli?

---

## 17. Latihan Mandiri, Mini Project, Rubrik Penilaian & Referensi

### Soal Latihan Mandiri

#### Tingkat 1: Fondasi Matematika & Konseptual
1. Buktikan secara analitis bahwa jika $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ berdistribusi i.i.d. dengan rata-rata 0 dan variansi 1, maka $\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$.
2. Hitung total parameter lapisan Multi-Head Attention dengan $d_{\text{model}} = 256$ dan $h = 4$ kepala atensi.

#### Tingkat 2: Analisis Alur Kerja & Debugging
1. Jelaskan mengapa fungsi Softmax naif tanpa *shift-max trick* menghasilkan nilai `nan` ketika memproses skor atensi bernilai besar ($> 710$), dan tuliskan implementasi perbaikannya!
2. Jelaskan bahaya *label leakage* jika *Causal Mask* tidak dipasang pada lapisan Self-Attention Decoder saat melatih model autoregresif!

#### Tingkat 3: Rekayasa Komputasional
1. Analisis mengapa kompleksitas memori $O(T^2)$ menjadi penghalang utama dalam pemrosesan dokumen hukum atau buku panjang, dan jelaskan bagaimana algoritma *FlashAttention* memitigasi kendala tersebut!

---

### Mini Project Chapter 28
**Judul**: *"Implementing Attention and a Small Transformer from Scratch"*

**Panduan Lengkap**: Pelajari petunjuk langkah-demi-langkah, instruksi implementasi, dan kriteria penilaian pada berkas [practice.md](practice.md).

#### Rubrik Penilaian Proyek (100 Poin):
- **Theoretical Understanding & Scaled Dot-Product Attention** (20 Poin): Penjelasan matematika Q-K-V lugas, pembuktian penskalaan $\sqrt{d_k}$, stabilitas numerik *shift-max*, dan masker komposit.
- **Multi-Head Attention & Positional Encoding Implementation** (25 Poin): Implementasi MHA NumPy murni modular, pemisahan kepala paralel tepat, fungsi sinusoidal PE benar.
- **Integrated Transformer Blocks & Seq2Seq Architecture** (25 Poin): Perakitan blok Encoder dan Decoder lengkap dengan residual connection, LayerNorm, dan FFN; simulasi forward pass Seq2Seq berjalan mulus pada CPU.
- **Rigorous Numerical Verification & Profiling** (15 Poin): Audit deviasi numerik terhadap PyTorch `nn.MultiheadAttention` menghasilkan deviasi $< 1.0 \times 10^{-6}$, disertai profil latensi CPU.
- **Technical Report, Responsible AI & Academic References** (15 Poin): Laporan komprehensif, analisis arsitektur mendalam, refleksi etika pemodelan bahasa, dan 5 referensi akademik primer.

---

### Referensi Akademik Primer

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** *Attention Is All You Need.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 5998–6008. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. **Bahdanau, D., Cho, K., & Bengio, Y. (2015).** *Neural Machine Translation by Jointly Learning to Align and Translate.* International Conference on Learning Representations (ICLR 2015). [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
3. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
4. **Jurafsky, D., & Martin, J. H. (2024).** *Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition* (3rd ed. draft). Stanford University. [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
5. **ACM / IEEE-CS Joint Task Force on Computing Curricula. (2023).** *Computer Science Curricula 2023 (CS2023): Artificial Intelligence and Machine Learning Curricular Guidelines.* ACM and IEEE Computer Society Press. [https://csed.acm.org/](https://csed.acm.org/)

