# Bab 24: Convolutional Neural Networks (CNN)

## Prinsip Korelasi Silang 2D, Konvolusi Spasial, Hierarki Representasi Visual, dan Pemrosesan Citra Modern dengan PyTorch

---

## 1. Dari Vektor Datar ke Citra 2D: Keterbatasan MLP pada Pengolahan Citra

Pada Bab 21 hingga 23, kita telah mempelajari dan mengimplementasikan arsitektur **Multilayer Perceptron (MLP)** atau jaringan saraf terhubung penuh (*Fully Connected Neural Network*). Pada data bertipe tabular atau vektor fitur 1D, MLP terbukti bekerja sangat andal. Namun, ketika berhadapan dengan data berstruktur kisi spasial dua dimensi (*2D spatial grid*) seperti citra digital, arsitektur MLP menghadapi tiga kelemahan mendasar:

```
+---------------------------------------------------------------------------------------------------------+
|                                KETERBATASAN MLP PADA DATA CITRA DIGITAL 2D                              |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. KEHILANGAN STRUKTUR SPASIAL (SPATIAL TOPOLOGY DESTRUCTION)                                         |
|      Citra 28x28 diratakan (flatten) menjadi vektor 784 dimensi. Piksel yang bertetangga secara        |
|      vertikal (misal baris 1 kolom 1 dan baris 2 kolom 1) menjadi terpisah sejauh 28 indeks,           |
|      menghancurkan korelasi lokal alami antarpiksel.                                                    |
|                                                                                                         |
|   2. LEDAKAN JUMLAH PARAMETER (PARAMETER EXPLOSION)                                                     |
|      Untuk citra RGB resolusi tinggi (misal 224x224x3 = 150.528 piksel), lapisan tersembunyi pertama     |
|      dengan 1.024 neuron membutuhkan: 150.528 x 1.024 ≈ 154,1 juta bobot! Ini memicu overfitting       |
|      parah dan melampaui kapasitas memori komputasi.                                                    |
|                                                                                                         |
|   3. KETIDAKMAMPUAN INVARIANSI TRANSLASI (LACK OF TRANSLATION INVARIANCE)                               |
|      Jika sebuah pola visual (misal mata kucing atau kancing kemeja) bergeser beberapa piksel ke kanan, |
|      MLP menganggapnya sebagai kombinasi input yang sepenuhnya baru dan gagal mengenalinya kecuali     |
|      telah dilatih pada seluruh kemungkinan posisi piksel tersebut.                                     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

Untuk mengatasi keterbatasan ini, **Convolutional Neural Networks (CNN)** dirancang dengan meniru mekanisme pemrosesan sistem visual biologis mamalia (seperti yang ditemukan oleh Hubel & Wiesel pada korteks visual kucing). CNN mempertahankan topologi 2D citra melalui operasi konvolusi spasial, memanfaatkan korelasi lokal, dan menerapkan pembagian bobot (*weight sharing*).

---

## 2. Anatomi Operasi Konvolusi: Cross-Correlation 2D, Kernel, dan Feature Map

Dalam konteks kecerdasan buatan dan pustaka deep learning (termasuk PyTorch), operasi yang disebut "konvolusi" secara matematis merupakan **korelasi silang dua dimensi** (*2D cross-correlation*), di mana kernel filter digeser di atas matriks masukan tanpa dibalik (*flip*) 180 derajat.

Diberikan matriks masukan dua dimensi $X \in \mathbb{R}^{H_{in} \times W_{in}}$ dan kernel filter $K \in \mathbb{R}^{K_h \times K_w}$, nilai aktivasi keluaran pada koordinat spasial $(i, j)$ dihitung melalui penjumlahan perkalian elemen (*element-wise dot product*):

$$
(X \star K)[i, j] = \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} X[i + m, j + n] \cdot K[m, n]
$$

Jika lapisan memiliki bobot bias skalar $b$, maka nilai bias ditambahkan pada setiap sel hasil:

$$
Y[i, j] = (X \star K)[i, j] + b
$$

```
+---------------------------------------------------------------------------------------------------------+
|                                ILUSTRASI MEKANISME KORELASI SILANG 2D                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|     Matriks Citra Masukan X (5x5)                 Kernel Filter K (3x3)        Peta Fitur / Feature Map |
|     +----+----+----+----+----+                    +----+----+----+             +----+----+----+         |
|     |  1 |  2 |  3 |  4 |  5 |                    | -1 | -1 | -1 |             | 30 | 30 | 30 |         |
|     +----+----+----+----+----+                    +----+----+----+             +----+----+----+         |
|     |  6 |  7 |  8 |  9 | 10 |                    |  0 |  0 |  0 |      ===>   | 30 | 30 | 30 |         |
|     +----+----+----+----+----+                    +----+----+----+             +----+----+----+         |
|     | 11 | 12 | 13 | 14 | 15 |                    |  1 |  1 |  1 |             | 30 | 30 | 30 |         |
|     +----+----+----+----+----+                    +----+----+----+             +----+----+----+         |
|     | 16 | 17 | 18 | 19 | 20 |               (Detektor Garis Horizontal)           Dimensi (3x3)        |
|     +----+----+----+----+----+                                                                          |
|     | 21 | 22 | 23 | 24 | 25 |                                                                          |
|     +----+----+----+----+----+                                                                          |
|                                                                                                         |
|   Perhitungan Sel Kiri-Atas Y[0, 0]:                                                                    |
|   Y[0, 0] = (1*-1 + 2*-1 + 3*-1) + (6*0 + 7*0 + 8*0) + (11*1 + 12*1 + 13*1)                            |
|           = (-1 - 2 - 3) + (0 + 0 + 0) + (11 + 12 + 13)                                                 |
|           = -6 + 36 = 30                                                                                |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Konvolusi pada Tensor Multi-Channel (3D Input)
Pada citra nyata dengan $C_{in}$ kanal (misal RGB dengan 3 kanal warna atau feature map perantara dengan 16 kanal), setiap kernel konvolusi memiliki kedalaman yang sama dengan $C_{in}$.

Jika masukan adalah $X \in \mathbb{R}^{C_{in} \times H \times W}$ dan kita ingin menghasilkan $C_{out}$ fitur keluaran:
1. Kita menggunakan $C_{out}$ buah filter berdimensi $C_{in} \times K_h \times K_w$.
2. Setiap filter $k \in \{0, \dots, C_{out}-1\}$ mengalikan seluruh kanal masukan secara simultan dan menjumlahkannya:

$$
Y_k[i, j] = b_k + \sum_{c=0}^{C_{in} - 1} \sum_{m=0}^{K_h - 1} \sum_{n=0}^{K_w - 1} X[c, i + m, j + n] \cdot K[k, c, m, n]
$$

---

## 3. Padding, Stride, dan Dilated Convolution

Operasi konvolusi standar menyebabkan dua fenomena geometris:
1. **Penyusutan Dimensi Spasial**: Setiap kali konvolusi diterapkan, batas tepi citra mengecil sebesar $K - 1$ piksel.
2. **Hilangnya Informasi Sudut**: Piksel di sudut citra hanya tersentuh oleh kernel satu kali, sedangkan piksel di bagian tengah diproses berulang kali.

Untuk mengendalikan geometri spasial dan mempertahankan informasi tepi, kita menggunakan teknik **Padding** dan **Stride**.

```
+---------------------------------------------------------------------------------------------------------+
|                                PADDING DAN STRIDE PADA OPERASI KONVOLUSI                                |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [ VALID PADDING (P = 0) ]                   [ SAME PADDING (P = 1, K = 3) ]                           |
|   Input: 4x4, Kernel: 3x3                     Input: 4x4, Kernel: 3x3                                   |
|   Output mengecil menjadi 2x2                 Tambahkan padding 0 di sekeliling matriks                |
|                                               Output tetap berdimensi 4x4!                              |
|                                                                                                         |
|   x  x  x  x                                  0  0  0  0  0  0                                          |
|   x  x  x  x                                  0 [x  x  x  x] 0                                          |
|   x  x  x  x                                  0 [x  x  x  x] 0                                          |
|   x  x  x  x                                  0 [x  x  x  x] 0                                          |
|                                               0 [x  x  x  x] 0                                          |
|                                               0  0  0  0  0  0                                          |
|                                                                                                         |
|   [ STRIDE (S = 2) ]                          [ DILATED CONVOLUTION (D = 2, K = 3) ]                    |
|   Langkah pergeseran kernel = 2 piksel        Kernel disisipi lubang spasial sebesar D - 1              |
|   Mereduksi dimensi spasial menjadi 50%       Memperluas Receptive Field tanpa menambah parameter!      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Rumus Matematis Dimensi Spasial Output
Diberikan ukuran spasial masukan $H_{in}$, ukuran kernel $K$, ukuran padding $P$, langkah pergeseran (*stride*) $S$, dan faktor dilasi $D$, dimensi spasial keluaran $H_{out}$ dirumuskan secara analitis:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - D(K - 1) - 1}{S} + 1 \right\rfloor
$$

Untuk konvolusi standar tanpa dilasi ($D=1$):

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} + 1 \right\rfloor
$$

### Rumus Penghitungan Parameter Bobot (Weight & Bias)
Lapisan konvolusi `nn.Conv2d(in_channels, out_channels, kernel_size)` memiliki total parameter terbobot yang sepenuhnya independen dari resolusi input:

$$
\text{Params}_{\text{Conv}} = (K_h \times K_w \times C_{in} + 1) \times C_{out}
$$

di mana $+1$ merepresentasikan satu bobot bias skalar untuk setiap kanal keluaran $C_{out}$.

---

## 4. Lapisan Pooling: Max Pooling, Invariansi Translasi, dan Downsampling

Lapisan **Pooling** bertugas mereduksi dimensi spasial (*downsampling*) dari feature map, mengurangi beban komputasi lapisan berikutnya, dan memberikan derajat **invariansi translasi lokal** (*local translation invariance*).

```
+---------------------------------------------------------------------------------------------------------+
|                                    MEKANISME MAX POOLING (2x2, STRIDE 2)                                |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|        Feature Map Input (4x4)                           Output Max Pooling (2x2)                       |
|        +----+----+----+----+                             +----+----+                                    |
|        | 12 | 20 | 30 |  0 |                             | 20 | 37 |                                    |
|        +----+----+----+----+     Max Pooling 2x2         +----+----+                                    |
|        |  8 | 15 |  2 | 37 |   ─────────────────>        | 88 | 19 |                                    |
|        +----+----+----+----+       Stride 2              +----+----+                                    |
|        | 45 | 88 |  9 |  1 |                                                                            |
|        +----+----+----+----+     Karakteristik Kunci:                                                   |
|        | 67 | 34 | 14 | 19 |     - Parameter Belajar: TEPAT 0 (Non-parametric operator)                 |
|        +----+----+----+----+     - Mereduksi memori spasial sebesar 75%                                 |
|                                  - Mempertahankan fitur paling menonjol (aktivasi tertinggi)            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Dua Varian Utama Pooling:
1. **Max Pooling (`nn.MaxPool2d`)**: Mengambil nilai maksimum dalam jendela $K \times K$. Berfungsi mempertahankan fitur dengan aktivasi paling kuat (seperti tepi kontras tinggi) serta mengabaikan variasi posisi minor.
2. **Average Pooling (`nn.AvgPool2d` / `nn.AdaptiveAvgPool2d`)**: Menghitung rata-rata seluruh nilai dalam jendela. Sering digunakan pada akhir arsitektur modern (*Global Average Pooling*) untuk menggantikan lapisan perataan (*flatten*) besar sebelum klasifikasi akhir.

---

## 5. Dua Prinsip Desain CNN: Local Receptive Field dan Weight Sharing

Mengapa CNN begitu superior dibandingkan MLP dalam memproses pola citra? Keunggulan ini bertumpu pada dua prinsip desain fundamental:

```
+---------------------------------------------------------------------------------------------------------+
|                               DUA PILAR FUNDAMENTAL KEUNGGULAN CNN                                      |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. KONEKTIVITAS LOKAL (LOCAL RECEPTIVE FIELD)                                                         |
|      Setiap neuron pada lapisan tersembunyi CNN hanya terhubung ke wilayah piksel spasial kecil         |
|      (misal 3x3 piksel) pada lapisan sebelumnya, bukan ke seluruh piksel citra.                         |
|      Ini mencerminkan realitas visual: korelasi piksel lokal jauh lebih informatif daripada jarak jauh. |
|                                                                                                         |
|   2. PEMBAGIAN BOBOT (WEIGHT SHARING)                                                                   |
|      Kernel filter yang sama (misal 3x3) disapu ke seluruh area koordinat citra masukan.                 |
|      Artinya, jika suatu filter belajar mendeteksi tepi melengkung, filter tersebut dapat mengenali     |
|      tepi melengkung di sudut kiri atas, tengah, maupun kanan bawah tanpa perlu bobot tambahan!         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Komparasi Kuantitatif Efisiensi Parameter
Mari bandingkan lapisan pertama pengolahan citra Fashion-MNIST ($28 \times 28$ grayscale, 1 kanal):
- **Lapisan Pertama CNN** (`nn.Conv2d(1, 16, kernel_size=3)`):
  $$\text{Params} = (3 \times 3 \times 1 + 1) \times 16 = 10 \times 16 = \mathbf{160} \text{ parameter}$$
- **Lapisan Pertama MLP** (`nn.Linear(784, 128)`):
  $$\text{Params} = 784 \times 128 + 128 = \mathbf{100.480} \text{ parameter}$$

CNN menghemat **99.84%** bobot pada lapisan pertama sembari tetap mempertahankan representasi topologi spasial 2D secara utuh!

---

## 6. Normalisasi Batch pada CNN (Spatial Batch Normalization)

Pada Bab 22 kita telah memahami bahwa pergeseran distribusi aktivasi (*internal covariate shift*) dapat memperlambat pelatihan jaringan saraf tiruan. Untuk jaringan konvolusional, Sergey Ioffe dan Christian Szegedy (2015) memperkenalkan **Spatial Batch Normalization** (`nn.BatchNorm2d`).

```
+---------------------------------------------------------------------------------------------------------+
|                                   SPATIAL BATCH NORMALIZATION 2D                                        |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Untuk tensor aktivasi mini-batch berdimensi (Batch B, Channels C, Height H, Width W):                 |
|   Statistik rata-rata μ_c dan variansi σ_c^2 dihitung untuk SETIAP KANAL c secara terpisah               |
|   dengan merata-ratakan seluruh dimensi B, H, dan W (total B x H x W elemen per kanal):                 |
|                                                                                                         |
|                1                                                                                        |
|      μ_c = ───────────  ∑    ∑    ∑   X[b, c, h, w]                                                     |
|            B * H * W    b    h    w                                                                     |
|                                                                                                         |
|   Aktivasi dinormalisasi dan diskalakan melalui parameter terpelajari gamma_c dan beta_c:              |
|                                                                                                         |
|                    X[b, c, h, w] - μ_c                                                                  |
|      Y[b, c, h, w] = γ_c ───────────────────── + β_c                                                    |
|                           sqrt(σ_c^2 + ε)                                                               |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

Keuntungan utama `nn.BatchNorm2d`:
1. Memungkinkan penggunaan laju pembelajaran (*learning rate*) yang lebih agresif.
2. Mengurangi sensitivitas terhadap inisialisasi bobot awal.
3. Memberikan efek regularisasi ringan yang meredam kebutuhan dropout pada lapisan konvolusi.

---

## 7. Hierarki Representasi Visual: Low-Level ke High-Level Features

Salah satu penemuan paling elegan dalam arsitektur deep learning visual adalah munculnya **hierarki fitur representasional** (*feature hierarchy*) secara mandiri selama proses pembelajaran:

```
+---------------------------------------------------------------------------------------------------------+
|                                HIERARKI REPRESENTASI FITUR PADA CNN                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Citra Input ──────> [ Layer Konvolusi 1 ] ──────> [ Layer Konvolusi 2 ] ──────> [ Lapisan Dense / FC ] |
|   (Piksel Mentah)     (Fitur Tingkat Rendah)        (Fitur Tingkat Menengah)      (Fitur Semantik)       |
|                                                                                                         |
|   - Intensitas piksel - Detektor garis/tepi        - Pola tekstur lingkaran      - Kategori objek:      |
|     warna/grayscale   - Orientasi sudut horizontal  - Pola kisi berulang            * T-shirt / Kemeja   |
|   - Gradien kontras   - Orientasi sudut vertikal   - Siluet kerah dan lengan        * Sepatu Boot        |
|                       - Tepi diagonal kontras      - Garis lipatan kain             * Tas Jinjing        |
|                                                                                                         |
|   Ukuran Spasial:      28x28 (Resolusi Tinggi)      14x14 -> 7x7 (Tereduksi)       Vektor 64 -> 10       |
|   Kedalaman Kanal:     1 Kanal                      16 -> 32 Kanal                Kelas Output          |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 8. Pipeline Data Citra: Torchvision Transforms dan Normalisasi Empiris

Dalam memproses citra untuk CNN menggunakan PyTorch, modul `torchvision.transforms` menyediakan alur kerja prapemrosesan data terstandar:

```python
from torchvision import transforms

# Pipeline Transformasi Citra
transform = transforms.Compose([
    transforms.ToTensor(), # Mengonversi PIL Image [0, 255] menjadi Tensor [0.0, 1.0] shape (C, H, W)
    transforms.Normalize(mean=(0.2860,), std=(0.3530,)) # Normalisasi Z-score empiris Fashion-MNIST
])
```

### Mengapa Normalisasi Citra Sangat Penting?
1. Mengubah intensitas piksel $[0.0, 1.0]$ menjadi distribusi nol-terpusat dengan rentang sekitar $[-1.0, 1.0]$.
2. Mencegah gradien menjauh dari titik operasi optimal fungsi aktivasi non-linier seperti ReLU.
3. Mempercepat laju konvergensi optimasi gradien pada seluruh lapisan jaringan.

---

## 9. Perancangan Arsitektur Model: FashionCNN vs FlattenMLPBaseline

Untuk membedah keunggulan CNN secara adil, kita merancang dua arsitektur yang memiliki jumlah total parameter yang seimbang (~105k vs ~109k parameter):

```
+---------------------------------------------------------------------------------------------------------+
|                                KOMPARASI SPESIFIKASI ARSITEKTUR KEDUA MODEL                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   MODEL A: FashionCNN (Total Parameter: 105.962)                                                        |
|   ├── Input: (Batch, 1, 28, 28)                                                                         |
|   ├── Blok Fitur 1: Conv2d(1 -> 16, K=3, P=1) -> BatchNorm2d(16) -> ReLU -> MaxPool2d(2, 2)            |
|   │   Output Spasial: (Batch, 16, 14, 14) | Parameter Layer 1: 160 bobot                                |
|   ├── Blok Fitur 2: Conv2d(16 -> 32, K=3, P=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2, 2)           |
|   │   Output Spasial: (Batch, 32, 7, 7)   | Parameter Layer 2: 4.640 bobot                              |
|   └── Klasifikator: Flatten -> Dropout(0.25) -> Linear(1568 -> 64) -> ReLU -> Linear(64 -> 10)         |
|       Output Akhir: (Batch, 10) Logits Mentah                                                           |
|                                                                                                         |
|   MODEL B: FlattenMLPBaseline (Total Parameter: 109.386)                                                |
|   ├── Input: (Batch, 1, 28, 28)                                                                         |
|   ├── Flatten: (Batch, 784)                                                                             |
|   ├── Linear(784 -> 128) -> ReLU -> Dropout(0.25) (Parameter Layer 1: 100.480 bobot!)                   |
|   ├── Linear(128 -> 64) -> ReLU                                                                         |
|   └── Linear(64 -> 10)                                                                                  |
|       Output Akhir: (Batch, 10) Logits Mentah                                                           |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 10. Protokol Pelatihan, Validasi, dan Hasil Benchmark Empiris

Kedua model dilatih pada dataset resmi **Fashion-MNIST** (Zalando Research, 60.000 citra latih dan 10.000 citra uji) di bawah protokol pengujian CPU yang sepenuhnya identik:
- Optimizer: Adam (`lr = 0.001`)
- Loss Function: `nn.CrossEntropyLoss()`
- Epochs: 5 epoch
- Batch Size: 64 sampel
- Hardware: Standard CPU

```
+---------------------------------------------------------------------------------------------------------+
|                                   HASIL BENCHMARK EMPIRIS FASHION-MNIST                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Metrik Pengujian             FlattenMLPBaseline       FashionCNN             Selisih / Keunggulan     |
|   ───────────────────────────────────────────────────────────────────────────────────────────────────   |
|   Test Accuracy                84.15%                   86.30%                 +2.15%                   |
|   Test Loss                    0.4410                   0.3664                 -0.0746 (Lebih stabil)   |
|   Total Parameter              109.386                  105.962                -3.424 (CNN lebih hemat) |
|                                                                                                         |
|   Akurasi Kelas 'Shirt'        54.31%                   75.63%                 +21.32% (MASIF!)         |
|   Akurasi Kelas 'Pullover'     76.10%                   83.40%                 +7.30%                   |
|   Akurasi Kelas 'Ankle boot'   94.60%                   96.10%                 +1.50%                   |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

> [!NOTE]
> Perhatikan lonjakan akurasi masif sebesar **+21.32%** pada kategori pakaian *Shirt*. Pakaian jenis kemeja memiliki detail tekstur halus seperti kerah, kancing depan vertikal, dan lipatan kain. Model MLP yang meratakan citra gagal menangkap pola spasial tersebut, sedangkan filter konvolusi CNN dengan mudah mengekstraksi representasi kerah dan tekstur kain.

---

## 11. Analisis Kesalahan (Error Analysis) & Inspeksi Feature Maps

Evaluasi ilmiah tidak berhenti pada angka akurasi agregat. Kita wajib membedah di mana model melakukan kesalahan dan bagaimana model "melihat" citra.

### 1. Diagnosis Pasangan Kelas Tertukar (Top Confused Pairs)
Berdasarkan matriks kontingensi kekeliruan prediksi pada test set:
1. **T-shirt/top vs Shirt** (226 kasus tertukar): Kedua jenis pakaian memiliki siluet potongan badan atas yang sangat mirip pada resolusi rendah $28 \times 28$.
2. **Pullover vs Coat** (129 kasus tertukar): Keduanya memiliki bentuk lengan panjang dan penutup leher yang hampir identik.
3. **Shirt vs Coat** (118 kasus tertukar): Garis bukaan kancing depan pada kemeja sering disalahartikan sebagai garis resleting/kancing mantel luar.

Sebaliknya, model hampir tidak pernah menukar pakaian atas (*Shirt*) dengan alas kaki (*Sneaker* atau *Ankle boot*), dengan tingkat kesalahan mendekati 0.

### 2. Inspeksi Visual Feature Maps Lapisan Pertama
Dengan mengekstraksi luaran lapisan konvolusi pertama `FashionCNN.features[0:3]`, kita dapat mengamati bagaimana 16 filter merespons pola citra:
- **Filter Kanal 0–3**: Berfungsi sebagai detektor tepi kontras horizontal (merespons garis bahu baju dan sol sepatu).
- **Filter Kanal 4–7**: Berfungsi sebagai detektor tepi vertikal (merespons garis lengan baju dan tepi celana).
- **Filter Kanal 8–15**: Merespons kepadatan tekstur material kain dan siluet tepi luar objek.

---

## 12. Latihan Mandiri 3-Tingkat Lengkap dengan Solusi

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. **Perhitungan Output Spasial Konvolusi**: Suatu citra satelit berukuran $(C_{in}, H_{in}, W_{in}) = (3, 128, 128)$ diproses melalui lapisan konvolusi dengan kernel berukuran $K = 7 \times 7$, padding $P = 3$, stride $S = 2$, dan jumlah filter $C_{out} = 64$. Berapakah dimensi spasial keluaran $(H_{out}, W_{out})$ dan berapakah total parameter bobot lapisan tersebut?
   - *Jawaban*:
     - Dimensi spasial:
       $$H_{out} = \left\lfloor \frac{128 + 2(3) - 7}{2} + 1 \right\rfloor = \left\lfloor \frac{128 + 6 - 7}{2} + 1 \right\rfloor = \left\lfloor \frac{127}{2} + 1 \right\rfloor = 63 + 1 = 64$$
       Jadi, tensor keluaran berdimensi $(64, 64, 64)$.
     - Total parameter bobot:
       $$\text{Params} = (K_h \times K_w \times C_{in} + 1) \times C_{out} = (7 \times 7 \times 3 + 1) \times 64 = (147 + 1) \times 64 = 148 \times 64 = \mathbf{9.472} \text{ parameter}$$

2. **Dilema Ukuran Kernel**: Mengapa arsitektur modern (seperti VGG dan ResNet) lebih memilih menumpuk dua atau tiga lapisan konvolusi kecil berukuran $3 \times 3$ daripada menggunakan satu lapisan konvolusi besar berukuran $7 \times 7$?
   - *Jawaban*:
     - **Receptive Field Setara**: Menumpuk dua konvolusi $3 \times 3$ menghasilkan receptive field efektif sebesar $5 \times 5$, dan tiga lapis konvolusi $3 \times 3$ menghasilkan receptive field setara $7 \times 7$.
     - **Efisiensi Parameter Drastis**: Tiga kernel $3 \times 3$ membutuhkan $3 \times (3^2 \times C^2) = 27 C^2$ bobot, sedangkan satu kernel $7 \times 7$ membutuhkan $7^2 \times C^2 = 49 C^2$ bobot (penghematan parameter sebesar $45\%$).
     - **Non-Linearitas Tambahan**: Tiga lapisan memiliki tiga fungsi aktivasi non-linier berturut-turut (misal ReLU), memberikan kapasitas representasi fungsi yang jauh lebih ekspresif.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Kegagalan Konvergensi Citra**: Seorang insinyur melatih model CNN pada citra mikroskopik dan mendapati bahwa loss pelatihan tidak kunjung turun setelah 20 epoch. Kode pemuatan datanya adalah sebagai berikut:
   ```python
   # Kode mahasiswa
   img = cv2.imread(filepath) # Membaca citra format BGR uint8 [0, 255]
   tensor_x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) # Shape (C, H, W)
   # Langsung diumpankan ke model tanpa normalisasi!
   ```
   Jelaskan akar penyebab masalah dan solusi perbaikannya!
   - *Jawaban*:
     - Akar Masalah: Nilai piksel berada pada rentang $[0, 255]$. Ketika diumpankan ke lapisan konvolusi berbobot inisialisasi normal, hasil kombinasi linier $\sum X \cdot W$ akan meledak ke angka ratusan (*exploding activations*). Ketika melewati fungsi aktivasi dan lapisan berikutnya, gradien meledak (*exploding gradient*) atau optimizer terjebak dalam saturasi numerik.
     - Solusi: Skalakan nilai piksel ke interval $[0.0, 1.0]$ dengan membaginya dengan $255.0$ atau gunakan `transforms.ToTensor()`, lalu lanjutkan dengan normalisasi Z-score standar:
       ```python
       tensor_x = (tensor_x / 255.0 - mean) / std
       ```

2. **Pengaruh Lapisan Pooling pada Posisi Objek**: Jika suatu sistem pengenalan wajah menggunakan Max Pooling berturut-turut hingga dimensi spasial menjadi $1 \times 1$, jelaskan mengapa sistem tersebut kehilangan kemampuan menentukan posisi koordinat spasial mata dan hidung!
   - *Jawaban*:
     - Max Pooling secara sengaja membuang koordinat spasial absolut demi mengejar invariansi translasi. Jika pooling dilakukan terlalu agresif hingga $1 \times 1$, model hanya mengetahui *apakah* mata atau hidung ada di dalam gambar (*feature presence*), tetapi kehilangan total informasi *di mana* fitur tersebut berada secara relatif satu sama lain (*spatial arrangement*).

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Perhitungan Receptive Field Efektif (ERF)**: Diberikan arsitektur CNN 4 lapis berikut:
   - Layer 1: Conv2d ($K_1=3, S_1=1, P_1=1$)
   - Layer 2: Conv2d ($K_2=3, S_2=1, P_2=1$)
   - Layer 3: MaxPool2d ($K_3=2, S_3=2$)
   - Layer 4: Conv2d ($K_4=3, S_4=1, P_4=1$)
   Hitung Receptive Field (RF) pada lapisan keluaran ke-4 terhadap citra input asli!
   - *Jawaban*:
     - Rumus rekursif: $RF_l = RF_{l-1} + (K_l - 1) \times J_{l-1}$ dengan $J_l = J_{l-1} \times S_l$ ($RF_0 = 1, J_0 = 1$).
     - Layer 1: $RF_1 = 1 + (3 - 1) \times 1 = 3$. Jump: $J_1 = 1 \times 1 = 1$.
     - Layer 2: $RF_2 = 3 + (3 - 1) \times 1 = 5$. Jump: $J_2 = 1 \times 1 = 1$.
     - Layer 3 (Pool): $RF_3 = 5 + (2 - 1) \times 1 = 6$. Jump: $J_3 = 1 \times 2 = 2$.
     - Layer 4: $RF_4 = 6 + (3 - 1) \times 2 = 6 + 4 = \mathbf{10}$.
     - Jadi, satu nilai keluaran pada lapisan ke-4 memiliki Receptive Field sebesar $10 \times 10$ piksel pada citra input asli.

---

## 13. Panduan Mini Project, Rubrik Penilaian, dan Referensi Terverifikasi

### Mini Project Chapter 24: "Benchmarking Convolutional Neural Networks on Visual Patterns"
Sebagai tugas akhir Bab 24, Anda diminta merancang dan mengeksekusi studi komparatif terpadu antara arsitektur **Convolutional Neural Networks (CNN)** dan arsitektur **Multilayer Perceptron (MLP)** pada dataset visi komputer **Fashion-MNIST** menggunakan PyTorch.

Panduan teknis langkah-demi-langkah, instruksi eksekusi kode, dan checklist penyerahan tersedia secara terperinci pada dokumen [practice.md](file:///D:/Workstation/Repo%20My%20Corses/Kecerdasan%20Buatan/content/level-3-deep-learning/chapter-24/practice.md).

### Rubrik Penilaian Proyek (Total: 100 Poin)

| Komponen Evaluasi | Bobot | Kriteria Kinerja Maksimal |
|---|:---:|---|
| **1. Data, Split & Transforms** | 20 Poin | Pipeline data torchvision bersih, transformasi `ToTensor()` dan normalisasi empiris ($mean=0.2860, std=0.3530$) diterapkan tepat, partisi 3-way (Train 50k, Val 10k, Test 10k) terisolasi tanpa tumpang tindih data leakage, dan mekanisme fallback sintetis tersedia. |
| **2. Architecture & Shape Correctness** | 20 Poin | Desain `FashionCNN` modular berbasis `nn.Module`, kalkulasi dimensi spasial tensor ($H_{out}, W_{out}$) akurat di setiap layer, perataan (*flatten*) ke lapisan linear presisi, dan luaran berupa logits mentah 10-kelas. |
| **3. Experiments & Baseline Comparison** | 25 Poin | Pelatihan komparatif CNN vs FlattenMLP tereksekusi mulus pada CPU, kondisi eksperimen dikontrol ketat (jumlah epoch, optimizer, learning rate, loss function identik), dan hasil benchmark tercatat rapi dalam format JSON. |
| **4. Evaluation & Error Analysis** | 20 Poin | Evaluasi performa uji menyeluruh, pelaporan akurasi per kelas yang membuktikan keunggulan CNN pada pola spasial kompleks (*Shirt*), identifikasi pasangan kelas tertukar (*top confused pairs*), dan visualisasi/ekstraksi aktivasi *feature maps* lapisan pertama. |
| **5. Reproducibility & Communication** | 15 Poin | Repositori kode bersih tanpa duplikasi, dokumentasi keterbatasan resolusi spasial $28 \times 28$, ketiadaan ketergantungan GPU khusus, dan penyajian kesimpulan ilmiah yang lugas dan berbasis data empiris. |

---

## 14. Daftar Referensi Terverifikasi

1. **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P.** (1998). *Gradient-based learning applied to document recognition*. Proceedings of the IEEE, 86(11), 2278-2324. [https://ieeexplore.ieee.org/document/726791](https://ieeexplore.ieee.org/document/726791)
2. **Xiao, H., Rasul, K., & Vollgraf, R.** (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv preprint arXiv:1708.07747. [https://arxiv.org/abs/1708.07747](https://arxiv.org/abs/1708.07747)
3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning: Chapter 9 Convolutional Networks*. MIT Press. [https://www.deeplearningbook.org](https://www.deeplearningbook.org)
4. **Krizhevsky, A., Sutskever, I., & Hinton, G. E.** (2012). *ImageNet classification with deep convolutional neural networks*. Advances in Neural Information Processing Systems (NeurIPS 2012), 25, 1097-1105. [https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
5. **IEEE Computer Society & ACM.** (2023). *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. Joint Task Force on Computing Curricula. [https://csed.acm.org](https://csed.acm.org)

