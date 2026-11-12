# Bab 21: Neural Network Fundamentals

## Arsitektur Jaringan Saraf Tiruan, Fungsi Aktivasi, dan Mekanisme Propagasi Maju

---

## 1. Hakikat dan Filosofi Deep Learning

Selamat datang di **Level 3: Deep Learning & Generative AI**. Jika pada Level 2 (Machine Learning) kita berfokus pada algoritma terawasi dan tak terawasi klasik yang sangat bergantung pada rekayasa fitur manual (*handcrafted feature engineering*), maka Level 3 membuka paradigma baru: **Representation Learning** (Pembelajaran Representasi).

Dalam pembelajaran mesin klasik, seorang insinyur data harus mengekstraksi parameter statistik, rasio fisik, atau transformasi domain secara eksplisit sebelum melatih algoritma seperti Regresi Logistik atau Support Vector Machines. Sebaliknya, **Deep Learning** memanfaatkan arsitektur jaringan saraf tiruan lapis banyak (*Multilayer Artificial Neural Networks*) untuk mempelajari representasi fitur hierarkis secara otomatis langsung dari data mentah (*raw data*) — mulai dari fitur tingkat rendah (tepi dan sudut pada piksel gambar), fitur tingkat menengah (tekstur dan bentuk), hingga fitur semantik tingkat tinggi (objek utuh).

```
+---------------------------------------------------------------------------------------------------------+
|                                  PARADIGMA PEMBELAJARAN MESIN VS DEEP LEARNING                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [PENDEKATAN KLASIK ML]                                                                                 |
|  Data Mentah ───> [Rekayasa Fitur Manual] ───> Fitur Rekayasa ───> [Model ML Klasik] ───> Luaran       |
|                   (Membutuhkan domain expert)                      (SVM / Random Forest)                |
|                                                                                                         |
|  [PENDEKATAN DEEP LEARNING]                                                                             |
|  Data Mentah ───> [ Layer 1: Tepi ] ───> [ Layer 2: Bentuk ] ───> [ Layer 3: Objek ] ───> Luaran       |
|                   └─────────────────── REPRESENTATION LEARNING ──────────────────┘                      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

```
+---------------------------------------------------------------------------------------------------------+
|                                HUBUNGAN INKLUSIF AI, MACHINE LEARNING & DEEP LEARNING                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   ===================================================================================                   |
|   | ARTIFICIAL INTELLIGENCE (AI)                                                    |                   |
|   | Mencakup seluruh sistem komputasi cerdas (logika simbolik, sistem pakar, dll.)  |                   |
|   |   ===========================================================================   |                   |
|   |   | MACHINE LEARNING (ML)                                                   |   |                   |
|   |   | Pembelajaran pola berbasis data statistik tanpa aturan terprogram       |   |                   |
|   |   |   ===================================================================   |   |                   |
|   |   |   | DEEP LEARNING (DL)                                              |   |   |                   |
|   |   |   | Jaringan saraf tiruan lapis banyak (Multilayer Neural Networks) |   |   |                   |
|   |   |   ===================================================================   |   |                   |
|   |   ===========================================================================   |   |                   |
|   ===================================================================================                   |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Model Neuron Buatan: Dari Biologi ke Matematika

Gagasan awal jaringan saraf tiruan terinspirasi dari struktur sel saraf biologis otak manusia. Pada tahun 1943, Warren McCulloch dan Walter Pitts merumuskan model komputasi pertama dari neuron biologis. Namun, penting untuk dicatat secara ilmiah: **Artificial Neural Networks HANYA terinspirasi secara abstrak dan TIDAK merepresentasikan simulasi biologis otak yang sesungguhnya**.

```
+---------------------------------------------------------------------------------------------------------+
|                               KOMPARASI STRUKTUR BIOLOGIS VS MODEL MATEMATIS                           |
+---------------------------------------------------------------------------------------------------------+
|  Komponen Biologis     Peran Fisiologis                 Model Matematika Neuron Buatan                  |
|  -----------------------------------------------------------------------------------------------------  |
|  Dendrit               Menerima impuls dari neuron lain  Vektor fitur masukan (x_1, x_2, ..., x_n)      |
|  Sinapsis              Kekuatan sambungan antar sel     Parameter bobot sinaptik (w_1, w_2, ..., w_n)   |
|  Soma (Badan Sel)      Akumulasi potensial membran      Kombinasi linier terbobot (Weighted Sum z)      |
|  Akson                 Transmisi potensial aksi         Fungsi aktivasi non-linier a = f(z)             |
+---------------------------------------------------------------------------------------------------------+
```

```
+---------------------------------------------------------------------------------------------------------+
|                                ANATOMI KOMPUTASI NEURON BUATAN TUNGGAL                                  |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|    Masukan (x_i)        Bobot (w_i)           Kombinasi Linier           Aktivasi          Luaran (a)   |
|                                                                                                         |
|       x_1  ──────────> ( w_1 ) ────────┐                                                                |
|                                        │                                                                |
|       x_2  ──────────> ( w_2 ) ────────┼───> [ z = w^T x + b ] ───> [ a = f(z) ] ───> a                |
|                                        │                                                                |
|       x_n  ──────────> ( w_n ) ────────┘                                                                |
|                                        │                                                                |
|       Bias ──────────> (  b  ) ────────┘                                                                |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Formulasi Aljabar Neuron Matematis
Operasi matematika neuron buatan terdiri dari dua tahap berurutan:

1. **Transformasi Linier Terbobot (*Weighted Sum*)**:
   $$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
   Di mana:
   - $\mathbf{x} = [x_1, x_2, \dots, x_n]^T \in \mathbb{R}^n$ adalah vektor fitur masukan.
   - $\mathbf{w} = [w_1, w_2, \dots, w_n]^T \in \mathbb{R}^n$ adalah vektor bobot sinaptik.
   - $b \in \mathbb{R}$ adalah skalar bias.

2. **Pemetaan Non-Linier (*Activation Function*)**:
   $$a = f(z) = f(\mathbf{w}^T \mathbf{x} + b)$$
   Di mana $f(\cdot)$ adalah fungsi aktivasi yang memetakan nilai kombinasi linier ke domain luaran yang diharapkan.

### Peran Fundamental Parameter Bias ($b$)
Parameter bias bertindak sebagai konstanta intersep pada persamaan garis atau bidang batas keputusan (*hyperplane*). Tanpa bias ($b = 0$), batas keputusan $\mathbf{w}^T \mathbf{x} = 0$ selalu terpaksa melewati titik asal (*origin* $\mathbf{0}$). Bias memberikan derajat kebebasan translasional yang memungkinkan hyperplane bergeser secara fleksibel di ruang fitur.

---

## 3. Rosenblatt Perceptron dan Batas Keterpisahan Linier

Pada tahun 1958, Frank Rosenblatt memperkenalkan **Perceptron**, arsitektur neuron pembelajaran terawasi pertama yang mampu mempelajari bobot dari data. Perceptron menggunakan fungsi tangga Heaviside (*Unit Step Function*) sebagai fungsi aktivasinya:

$$f(z) = \begin{cases} 1 & \text{jika } z \ge 0 \\ 0 & \text{jika } z < 0 \end{cases}$$

Batas keputusan Perceptron ditentukan oleh persamaan hyperplane:
$$\mathbf{w}^T \mathbf{x} + b = 0 \iff w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = 0$$

```
+---------------------------------------------------------------------------------------------------------+
|                                    BATAS KEPUTUSAN LINEAR HYPERPLANE 2D                                |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       x_2 ^                                                                                             |
|           |         Wilayah Positif (z >= 0 -> a = 1)                                                   |
|           |            (Kelas 1)   *       *                                                            |
|           |                   *        *                                                                |
|           |                      *   *                                                                  |
|           |  Garis Batas: w1*x1 + w2*x2 + b = 0                                                         |
|           |──────────────────────────────────────────                                                   |
|           |         Wilayah Negatif (z < 0 -> a = 0)                                                    |
|           |            (Kelas 0)   o       o                                                            |
|           |                   o        o                                                                |
|           +──────────────────────────────────────────> x_1                                              |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Implementasi Gerbang Logika Biner Linier
Sebuah Perceptron tunggal mampu merepresentasikan fungsi logika dasar yang bersifat **linearly separable** (dapat dipisahkan secara linier):

```
+---------------------------------------------------------------------------------------------------------+
|                               TABEL KEBENARAN & PARAMETER GERBANG LOGIKA                                |
+---------------------------------------------------------------------------------------------------------+
|  x1   x2  │  AND (w=[1,1], b=-1.5)  │  OR (w=[1,1], b=-0.5)  │  NAND (w=[-1,-1], b=1.5)                 |
|  ─────────┼─────────────────────────┼────────────────────────┼──────────────────────────                |
|   0    0  │  z = -1.5 -> y = 0      │  z = -0.5 -> y = 0     │  z = +1.5 -> y = 1                       |
|   0    1  │  z = -0.5 -> y = 0      │  z = +0.5 -> y = 1     │  z = +0.5 -> y = 1                       |
|   1    0  │  z = -0.5 -> y = 0      │  z = +0.5 -> y = 1     │  z = +0.5 -> y = 1                       |
|   1    1  │  z = +0.5 -> y = 1      │  z = +1.5 -> y = 1     │  z = -0.5 -> y = 0                       |
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Masalah XOR dan Batasan Teoretis Minsky-Papert

Pada tahun 1969, Marvin Minsky dan Seymour Papert menerbitkan buku *Perceptrons* yang membuktikan kelemahan fatal Perceptron lapis tunggal: ketidakmampuannya memodelkan fungsi logika **XOR (Exclusive OR)**.

```
+---------------------------------------------------------------------------------------------------------+
|                                       MASALAH XOR (NON-LINEAR SEPARABILITY)                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       x_2 ^                                                                                             |
|           |                                                                                             |
|         1 |    (0, 1) [Kelas 1]  *            o  (1, 1) [Kelas 0]                                       |
|           |                                                                                             |
|           |                                                                                             |
|           |                                                                                             |
|         0 |    (0, 0) [Kelas 0]  o            *  (1, 0) [Kelas 1]                                       |
|           +─────────────────────────────────────────────> x_1                                           |
|                  0                           1                                                          |
|                                                                                                         |
|  TIDAK ADA SATU GARIS LURUS PUN YANG DAPAT MEMISAHKAN TITIK (*) DARI TITIK (o)!                         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Bukti Matematis Kontradiksi Aljabar XOR
Misalkan terdapat Perceptron dengan bobot $w_1, w_2$ dan bias $b$ yang dapat memecahkan XOR. Maka harus dipenuhi 4 pertidaksamaan simultan:
1. Untuk $\mathbf{x} = (0, 0) \implies y = 0$:
   $$w_1(0) + w_2(0) + b < 0 \implies b < 0$$
2. Untuk $\mathbf{x} = (0, 1) \implies y = 1$:
   $$w_1(0) + w_2(1) + b \ge 0 \implies w_2 + b \ge 0$$
3. Untuk $\mathbf{x} = (1, 0) \implies y = 1$:
   $$w_1(1) + w_2(0) + b \ge 0 \implies w_1 + b \ge 0$$
4. Untuk $\mathbf{x} = (1, 1) \implies y = 0$:
   $$w_1(1) + w_2(1) + b < 0 \implies w_1 + w_2 + b < 0$$

Jika kita menjumlahkan pertidaksamaan (2) dan pertidaksamaan (3):
$$(w_2 + b) + (w_1 + b) \ge 0 \implies w_1 + w_2 + 2b \ge 0 \implies w_1 + w_2 + b \ge -b$$
Dari pertidaksamaan (1), kita tahu bahwa $b < 0$, sehingga $-b > 0$. Oleh karena itu:
$$w_1 + w_2 + b > 0$$
Namun, pertidaksamaan (4) secara eksplisit menuntut bahwa:
$$w_1 + w_2 + b < 0$$
Pernyataan $w_1 + w_2 + b > 0$ dan $w_1 + w_2 + b < 0$ berkontradiksi secara mutlak! **Q.E.D.** Terbukti secara formal bahwa Perceptron tunggal tidak mungkin memecahkan gerbang XOR.

### Solusi Jaringan Lapis Banyak (*Multilayer Perceptron*)
Masalah XOR diselesaikan dengan menyusun jaringan multi-lapisan. Fungsi XOR dapat didekomposisi menjadi kombinasi gerbang linier dasar:
$$\text{XOR}(x_1, x_2) = (x_1 \lor x_2) \land \neg(x_1 \land x_2) = (x_1 \lor x_2) \land (x_1 \text{ NAND } x_2)$$

```
+---------------------------------------------------------------------------------------------------------+
|                                ARSITEKTUR 2-LAYER SEBAGAI PEMECAH MASALAH XOR                           |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       Input Layer              Hidden Layer (Feature Space Baru)             Output Layer               |
|                                                                                                         |
|         ( x_1 ) ────────┬──────────────> [ Neuron 1: OR ] ─────────┐                                    |
|                         │                (h_1 = x_1 OR x_2)        │                                    |
|                         │                                          ├───> [ Neuron 3: AND ] ──> XOR      |
|                         │                                          │     (y = h_1 AND h_2)              |
|         ( x_2 ) ────────┴──────────────> [ Neuron 2: NAND ] ───────┘                                    |
|                                          (h_2 = x_1 NAND x_2)                                           |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

Lapisan tersembunyi (*hidden layer*) mentransformasikan ruang masukan non-linier $\mathbf{x} \in \{0,1\}^2$ menjadi ruang representasi baru $\mathbf{h} \in \{0,1\}^2$ di mana titik data menjadi *linearly separable* bagi neuron luaran.

---

## 5. Fungsi Aktivasi Non-Linier: Karakteristik dan Turunan

Jika kita hanya menyusun banyak lapisan tanpa fungsi aktivasi non-linier, jaringan saraf sedalam apa pun akan mengalami keruntuhan linier (*linear collapse*):

$$\mathbf{a}^{[2]} = \mathbf{W}^{[2]} (\mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]} = (\mathbf{W}^{[2]} \mathbf{W}^{[1]}) \mathbf{x} + (\mathbf{W}^{[2]} \mathbf{b}^{[1]} + \mathbf{b}^{[2]}) = \mathbf{W}_{\text{net}} \mathbf{x} + \mathbf{b}_{\text{net}}$$

Jaringan 100 lapisan linier secara matematis setara dengan regresi linier 1 lapisan. Oleh karena itu, fungsi aktivasi non-linier adalah komponen wajib yang menyuntikkan kelengkungan fungsional.

```
+---------------------------------------------------------------------------------------------------------+
|                                   TAKSONOMI FUNGSI AKTIVASI POPULER                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|    SIGMOID: f(z) = 1 / (1 + e^-z)           TANH: f(z) = (e^z - e^-z) / (e^z + e^-z)                    |
|          1 |          *----*                      1 |               *---*                               |
|            |        *                             |             *                                       |
|        0.5 |    ---*---                           0 | -------*-------                                   |
|            |      *                               |        *                                            |
|          0 | *---*                               -1 | *---*                                             |
|            +-------------------->                   +-------------------->                              |
|             -4   0   4                               -3   0   3                                         |
|                                                                                                         |
|    RELU: f(z) = max(0, z)                   LEAKY RELU: f(z) = max(alpha*z, z)                          |
|          4 |             /                        4 |             /                                     |
|          2 |           /                          2 |           /                                       |
|          0 | ________/                            0 | ________/  (slope kecil alpha saat z < 0)         |
|            +-------------------->                   +-------------------->                              |
|             -3   0   3                               -3   0   3                                         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 1. Sigmoid (Fungsi Logistik)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Domain & Range**: $\mathbb{R} \to (0, 1)$.
- **Turunan Pertama**:
  $$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$
- **Analisis Kritis**: Nilai turunan maksimum berada di $z=0$ dengan nilai $\sigma'(0) = 0.25$. Saat $|z| > 4$, turunan mendekati $0$. Masalah ini menyebabkan fenomena **Vanishing Gradient** pada jaringan dalam.

### 2. Tangen Hiperbolik (Tanh)
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$
- **Domain & Range**: $\mathbb{R} \to (-1, 1)$.
- **Turunan Pertama**:
  $$\frac{d\tanh}{dz} = 1 - \tanh^2(z)$$
- **Karakteristik**: Bersifat *zero-centered* (luaran rata-rata mendekati 0), yang mempercepat konvergensi gradien dibandingkan sigmoid.

### 3. Rectified Linear Unit (ReLU)
$$f(z) = \max(0, z)$$
- **Domain & Range**: $\mathbb{R} \to [0, \infty)$.
- **Turunan Pertama**:
  $$f'(z) = \begin{cases} 1 & \text{jika } z > 0 \\ 0 & \text{jika } z < 0 \end{cases}$$
- **Kelebihan**: Komputasi sangat efisien (hanya operasi ambang threshold) dan tidak mengalami saturasi pada wilayah positif ($z > 0$).
- **Kelemahan (Dying ReLU)**: Jika neuron menerima pembaruan gradien yang mendorong nilai $z$ selalu negatif, gradiennya menjadi nol permanen ($f'(z) = 0$). Neuron tersebut 'mati' dan tidak pernah memperbarui bobotnya lagi.

### 4. Leaky ReLU
$$f(z) = \max(\alpha z, z) = \begin{cases} z & \text{jika } z > 0 \\ \alpha z & \text{jika } z \le 0 \end{cases}$$
- Biasanya konstanta kemiringan $\alpha = 0.01$.
- **Solusi Dying ReLU**: Mempertahankan gradien kecil $\alpha$ pada wilayah negatif, memungkinkan neuron yang tidak aktif untuk pulih selama proses training.

### 5. Softmax (Klasifikasi Multikelas)
Untuk klasifikasi dengan $K$ kelas yang saling eksklusif, Softmax memetakan vektor logits $\mathbf{z} \in \mathbb{R}^K$ menjadi distribusi probabilitas yang valid:
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad \text{untuk } i = 1, \dots, K$$

#### Trik Stabilitas Numerik (*Subtract-Max Trick*)
Dalam floating-point standar IEEE 754, komputasi eksponensial $e^{z_i}$ pada $z_i > 709$ akan menghasilkan nilai `inf` (*Overflow*). Untuk mencegahnya, kita mengurangkan nilai maksimum dari seluruh komponen vektor logits:
$$\text{Softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{K} e^{z_j - \max(\mathbf{z})}}$$
Secara matematis nilainya identik karena faktor $e^{-\max(\mathbf{z})}$ saling meniadakan di pembilang dan penyebut, tetapi menjamin bahwa pangkat eksponensial terbesar bernilai $e^0 = 1$, mengeliminasi risiko overflow 100%.

---

## 6. Arsitektur Multilayer Perceptron (MLP)

Multilayer Perceptron (MLP) adalah jaringan saraf feedforward yang terdiri dari lapisan masukan (*input layer*), satu atau lebih lapisan tersembunyi (*hidden layers*), dan satu lapisan luaran (*output layer*).

```
+---------------------------------------------------------------------------------------------------------+
|                                  ARSITEKTUR LENGKAP FEEDFORWARD MLP                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       INPUT LAYER                   HIDDEN LAYER 1                HIDDEN LAYER 2          OUTPUT LAYER  |
|         (l = 0)                        (l = 1)                       (l = 2)                (l = 3)     |
|                                                                                                         |
|          (x_1) ──────────────────────> ( h_1^[1] ) ────────────────> ( h_1^[2] ) ─────────> ( y_hat_1 )  |
|                                   X                             X                    X                  |
|          (x_2) ──────────────────────> ( h_2^[1] ) ────────────────> ( h_2^[2] ) ─────────> ( y_hat_2 )  |
|                                   X                             X                                       |
|          (x_3) ──────────────────────> ( h_3^[1] ) ────────────────> ( h_3^[2] )                        |
|                                                                                                         |
|                     W^[1], b^[1]                  W^[2], b^[2]                  W^[3], b^[3]            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Konvensi Dimensi Tensor & Bentuk Matriks
Untuk batch berukuran $N$ sampel:
- **Matriks Masukan $\mathbf{X}$**: Berukuran $(N, D_{\text{in}})$.
- **Matriks Bobot $\mathbf{W}^{[l]}$**: Menghubungkan layer $l-1$ ke layer $l$, berukuran $(n_{l-1}, n_l)$.
- **Vektor Bias $\mathbf{b}^{[l]}$**: Berukuran $(1, n_l)$, ditambahkan ke setiap sampel melalui *broadcasting*.
- **Matriks Kombinasi Linier $\mathbf{Z}^{[l]}$**: Berukuran $(N, n_l)$.
- **Matriks Aktivasi $\mathbf{A}^{[l]}$**: Berukuran $(N, n_l)$.

### Teorema Aproksimasi Universal (Universal Approximation Theorem)
Dinyatakan oleh Cybenko (1989) dan Hornik (1991):
> *Sebuah jaringan saraf tiruan feedforward dengan satu hidden layer yang memuat sejumlah berhingga neuron dan fungsi aktivasi non-linier kontinu dapat mengaproksimasi fungsi kontinu apa pun pada himpunan kompak $\mathbb{R}^n$ dengan derajat akurasi arbitrer $\epsilon > 0$.*

**Keterbatasan Praktis Teorema**: Teorema ini adalah bukti eksistensi (*existence proof*), bukan bukti konstruktif. Teorema ini menyatakan bahwa bobot parameter yang tepat itu ada, tetapi tidak menjamin:
1. Bahwa jumlah neuron yang dibutuhkan masuk akal (mungkin memerlukan jutaan neuron pada satu layer).
2. Bahwa algoritma optimasi berbasis gradien (*Gradient Descent*) dapat menemukan konfigurasi bobot tersebut.
3. Bahwa jaringan yang dilatih akan memiliki kemampuan generalisasi yang baik pada data uji.

---

## 7. Mekanisme Propagasi Maju (Forward Propagation)

Propagasi maju adalah proses deterministik untuk mengevaluasi luaran model dengan melewatkan tensor data dari lapisan masukan hingga ke lapisan luaran.

```
+---------------------------------------------------------------------------------------------------------+
|                                ALUR KOMPUTASI REKURSIF PROPAGASI MAJU                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       Inisialisasi: A^[0] = X (Matriks masukan N x D_in)                                                |
|                                                                                                         |
|       Untuk setiap layer l = 1, 2, ..., L:                                                              |
|         1. Transformasi Linier   : Z^[l] = A^[l-1] W^[l] + b^[l]                                        |
|         2. Aktivasi Non-Linier   : A^[l] = f^[l]( Z^[l] )                                               |
|         3. Simpan dalam Cache    : simpan A^[l-1], W^[l], b^[l], Z^[l] untuk backward pass              |
|                                                                                                         |
|       Terminasi: Prediksi Model y_hat = A^[L]                                                           |
|       Evaluasi Loss: Loss = L( y_true, y_hat )                                                          |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 8. Worked Numerical Example: Perhitungan Manual Lengkap (2 -> 3 -> 1)

Mari kita buktikan perhitungan propagasi maju langkah-demi-langkah dengan angka-angka konkret pada arsitektur jaringan kecil:
- **Masukan**: 2 fitur $\mathbf{x} = [1.5, 2.0]$.
- **Hidden Layer**: 3 neuron dengan aktivasi $\text{ReLU}$.
- **Output Layer**: 1 neuron dengan aktivasi $\text{Sigmoid}$.

```
+---------------------------------------------------------------------------------------------------------+
|                                DIAGRAM SKALAR WORKED EXAMPLE (2 -> 3 -> 1)                              |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|    x_1 = 1.5 ───┬──(w11=0.5)──> [ Neuron h1: z1=1.65 -> a1=1.65 ] ───(w1=0.6)───┐                     |
|                 │                                                                │                     |
|                 ├──(w12=-0.2)─> [ Neuron h2: z2=0.80 -> a2=0.80 ] ───(w2=-0.8)──┼──> [ Output ]        |
|                 │                                                                │     z2 = 0.370      |
|                 └──(w13=0.8)──> [ Neuron h3: z3=0.40 -> a3=0.40 ] ───(w3=0.3)───┘     a2 = 0.5915     |
|                                                                                  │     (y_hat)         |
|    x_2 = 2.0 ───┼──(w21=0.4)─────────────────────────────────────────────────────┤                     |
|                 ├──(w22=0.7)─────────────────────────────────────────────────────┤                     |
|                 └──(w23=-0.5)────────────────────────────────────────────────────┘                     |
|                                                                                                         |
|    Bias b^[1] = [0.1, -0.3, 0.2]                                Bias b^[2] = [-0.1]                    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Parameter Eksplisit:
$$\mathbf{W}^{[1]} = \begin{bmatrix} 0.5 & -0.2 & 0.8 \\ 0.4 & 0.7 & -0.5 \end{bmatrix}, \quad \mathbf{b}^{[1]} = \begin{bmatrix} 0.1 & -0.3 & 0.2 \end{bmatrix}$$
$$\mathbf{W}^{[2]} = \begin{bmatrix} 0.6 \\ -0.8 \\ 0.3 \end{bmatrix}, \quad b^{[2]} = -0.1$$

### Langkah 1: Kombinasi Linier Hidden Layer ($\mathbf{Z}^{[1]}$)
$$\mathbf{Z}^{[1]} = \mathbf{x} \mathbf{W}^{[1]} + \mathbf{b}^{[1]}$$
- **Neuron 1**: $z_1^{[1]} = x_1 w_{11}^{[1]} + x_2 w_{21}^{[1]} + b_1^{[1]} = 1.5(0.5) + 2.0(0.4) + 0.1 = 0.75 + 0.80 + 0.10 = \mathbf{1.65}$
- **Neuron 2**: $z_2^{[1]} = x_1 w_{12}^{[1]} + x_2 w_{22}^{[1]} + b_2^{[1]} = 1.5(-0.2) + 2.0(0.7) - 0.3 = -0.30 + 1.40 - 0.30 = \mathbf{0.80}$
- **Neuron 3**: $z_3^{[1]} = x_1 w_{13}^{[1]} + x_2 w_{23}^{[1]} + b_3^{[1]} = 1.5(0.8) + 2.0(-0.5) + 0.2 = 1.20 - 1.00 + 0.20 = \mathbf{0.40}$
$$\mathbf{Z}^{[1]} = \begin{bmatrix} 1.65 & 0.80 & 0.40 \end{bmatrix}$$

### Langkah 2: Aktivasi Non-Linier Hidden Layer ($\mathbf{A}^{[1]} = \text{ReLU}(\mathbf{Z}^{[1]})$)
Karena seluruh nilai $z_i^{[1]} > 0$, maka $\max(0, z) = z$:
$$\mathbf{A}^{[1]} = \begin{bmatrix} \max(0, 1.65) & \max(0, 0.80) & \max(0, 0.40) \end{bmatrix} = \begin{bmatrix} \mathbf{1.65} & \mathbf{0.80} & \mathbf{0.40} \end{bmatrix}$$

### Langkah 3: Kombinasi Linier Output Layer ($Z^{[2]}$)
$$Z^{[2]} = \mathbf{A}^{[1]} \mathbf{W}^{[2]} + b^{[2]}$$
$$Z^{[2]} = (1.65 \times 0.6) + (0.80 \times (-0.8)) + (0.40 \times 0.3) + (-0.1)$$
$$Z^{[2]} = 0.990 - 0.640 + 0.120 - 0.100 = \mathbf{0.370}$$

### Langkah 4: Aktivasi Sigmoid Output Layer ($A^{[2]}$)
$$A^{[2]} = \sigma(Z^{[2]}) = \frac{1}{1 + e^{-0.370}} = \frac{1}{1 + 0.690726} = \frac{1}{1.690726} \approx \mathbf{0.5915}$$

Prediksi probabilitas model adalah $\hat{y} = 0.5915$ (59.15% peluang kelas positif). Perhitungan matematis manual ini terverifikasi identik 100% dengan luaran skrip komputasi `forward_propagation.py`.

---

## 9. Pemilihan Lapisan Luaran dan Fungsi Rugi (Loss Functions)

Pasangan antara fungsi aktivasi pada lapisan luaran dan fungsi rugi (*loss function*) harus dipilih secara konsisten berdasarkan prinsip pemodelan probabilitas (*Maximum Likelihood Estimation*):

```
+---------------------------------------------------------------------------------------------------------+
|                                MATRIKS PASANGAN AKTIVASI OUTPUT & LOSS FUNCTION                         |
+---------------------------------------------------------------------------------------------------------+
|  Tipe Masalah                Aktivasi Output      Fungsi Rugi (Loss Function)   Formula Standar         |
|  -----------------------------------------------------------------------------------------------------  |
|  Regresi Kontinu             Linear (Identitas)   Mean Squared Error (MSE)       (1/N) * sum (y - y_hat)^2|
|  Klasifikasi Biner           Sigmoid              Binary Cross-Entropy (BCE)    - (1/N) * sum [y*log(p) ]|
|  Klasifikasi Multikelas      Softmax              Categorical Cross-Entropy     - (1/N) * sum sum y*log(p|
+---------------------------------------------------------------------------------------------------------+
```

```
+---------------------------------------------------------------------------------------------------------+
|                                    PERILAKU LOGARITMIK PENALTI CROSS-ENTROPY                            |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Loss (-log p)                                                                                         |
|       ^                                                                                                 |
|   inf | *  (Jika target y = 1, namun model memprediksi p -> 0, loss meledak tak hingga!)                 |
|       |  *                                                                                              |
|   3.0 |   *                                                                                             |
|   2.0 |     *                                                                                           |
|   1.0 |       *                                                                                         |
|   0.0 |─────────*──────────────────> Probabilitas Prediksi (p)                                          |
|       0.0      0.5       1.0                                                                            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Stabilitas Numerik Fungsi Rugi
Dalam implementasi perangkat lunak, evaluasi langsung $\ln(0)$ menghasilkan $-\infty$ atau `NaN`. Standar rekayasa kecerdasan buatan mewajibkan pemotongan probabilitas (*epsilon clipping*):
$$\hat{y}_{\text{clipped}} = \text{clip}(\hat{y}, \epsilon, 1 - \epsilon), \quad \epsilon = 10^{-15}$$
Trik sederhana ini menjamin komputasi loss stabil dan bebas *crash*.

---

## 10. Batasan Jaringan Saraf Tiruan & Tata Kelola AI Bertanggung Jawab

Meskipun memiliki kapasitas representasi yang revolusioner, jaringan saraf tiruan memiliki keterbatasan nyata yang wajib dipahami oleh setiap insinyur kecerdasan buatan:

1. **Kebutuhan Data Skala Besar (*Data Hunger*)**: Jaringan saraf dengan jutaan parameter sangat rentan mengalami *overfitting* parah jika dilatih pada dataset berukuran kecil ($N < 1,000$). Dalam kasus data kecil atau tabel terstruktur, algoritma klasik seperti Random Forest atau XGBoost hampir selalu mengungguli deep learning.
2. **Kotak Hitam (*Black-Box Problem*)**: Penalaran internal pada jutaan bobot sinaptik sangat sulit ditafsirkan secara langsung oleh manusia, menimbulkan risiko tinggi pada domain krusial seperti diagnosis medis atau pengadilan pidana.
3. **Biaya Komputasi & Jejak Karbon**: Pelatihan arsitektur modern menuntut akselerator perangkat keras (GPU/TPU) dan konsumsi energi listrik yang sangat masif.

---

## 11. Latihan Mandiri 3 Tingkat (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. Jelaskan mengapa fungsi aktivasi tangga Heaviside pada Perceptron Rosenblatt tidak dapat digunakan untuk melatih jaringan multi-lapisan dengan algoritma berbasis gradien (*Gradient Descent*)!
   - *Jawaban*: Turunan fungsi tangga Heaviside adalah 0 di semua titik (kecuali di $z=0$ di mana ia tidak terdefinisi). Karena gradien bernilai nol di mana-mana, sinyal galat tidak dapat dialirkan balik (*zero gradient*) untuk memperbarui bobot parameter.

2. Sebutkan kelebihan utama fungsi aktivasi Tanh dibandingkan Sigmoid ketika digunakan pada lapisan tersembunyi (*hidden layers*)!
   - *Jawaban*: Tanh bersifat *zero-centered* (luaran berpusat di nol, rentang $-1$ hingga $+1$). Hal ini mencegah terjadinya *zigzagging gradient dynamics* selama pembaruan bobot, sehingga konvergensi pelatihan berlangsung lebih cepat dibanding Sigmoid.

### Tingkat 2: Analisis Matematis (Menengah)
1. Diberikan jaringan saraf dengan 4 fitur masukan, Hidden Layer 1 berisi 16 neuron, Hidden Layer 2 berisi 8 neuron, dan Output Layer berisi 3 neuron. Hitunglah:
   - Dimensi matriks bobot $\mathbf{W}^{[1]}, \mathbf{W}^{[2]}, \mathbf{W}^{[3]}$ dan vektor bias $\mathbf{b}^{[1]}, \mathbf{b}^{[2]}, \mathbf{b}^{[3]}$.
   - Total seluruh parameter yang dapat dilatih (*trainable parameters*)!
   - *Jawaban*:
     - $\mathbf{W}^{[1]}: (4, 16)$, $\mathbf{b}^{[1]}: (1, 16) \implies (4 \times 16) + 16 = 64 + 16 = 80$.
     - $\mathbf{W}^{[2]}: (16, 8)$, $\mathbf{b}^{[2]}: (1, 8) \implies (16 \times 8) + 8 = 128 + 8 = 136$.
     - $\mathbf{W}^{[3]}: (8, 3)$, $\mathbf{b}^{[3]}: (1, 3) \implies (8 \times 3) + 3 = 24 + 3 = 27$.
     - Total Parameter = $80 + 136 + 27 = \mathbf{243}$ parameter.

2. Turunkan secara aljabar turunan fungsi aktivasi Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ dan buktikan bahwa $\sigma'(z) = \sigma(z)(1 - \sigma(z))$!
   - *Jawaban*:
     $$\sigma'(z) = \frac{d}{dz}(1 + e^{-z})^{-1} = -1(1 + e^{-z})^{-2}(-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$
     Tambahkan dan kurangkan 1 pada pembilang:
     $$\sigma'(z) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = \sigma(z) \left( 1 - \frac{1}{1 + e^{-z}} \right) = \sigma(z)(1 - \sigma(z)). \quad \text{Q.E.D.}$$

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. Tuliskan kode fungsi Python `forward_pass(X, parameters, activations)` yang mampu memproses batch matriks masukan secara rekursif untuk kedalaman $L$ lapisan arbitrer menggunakan NumPy murni!
   - *Jawaban*: Lihat implementasi referensi lengkap pada berkas `content/level-3-deep-learning/chapter-21/code/numpy_neural_network_forward.py`.

---

## 12. Mini Project Chapter 21

### Judul Proyek:
**"Building a Neural Network Forward Pass from Scratch Using NumPy"**

### Panduan Pengerjaan:
1. Bangun kelas `NeuralNetworkForward` yang menerima argumen: `layer_dims` (list dimensi lapisan), `activations` (list fungsi aktivasi: `relu`, `sigmoid`, `tanh`, `softmax`), dan metode inisialisasi bobot (`he` atau `xavier`).
2. Implementasikan metode `forward(X)` yang mengembalikan aktivasi luaran akhir dan dictionary `cache` penyimpan seluruh matriks perantara $\mathbf{Z}^{[l]}$ dan $\mathbf{A}^{[l]}$.
3. Implementasikan fungsi rugi terstabilisasi: `compute_loss(y_true, y_pred, loss_type='bce')`.
4. Uji kelas Anda pada dataset sintetis multikelas 3 kelas (misal: klaster 2D) dan buktikan bahwa dimensi tensor serta total parameter konsisten.

### Rubrik Penilaian Proyek (100 Poin):
- **1. Pemahaman Arsitektur & Dimensi (20 Poin)**: Pemetaan bentuk matriks $\mathbf{W}$ dan $\mathbf{b}$ benar pada berbagai ukuran batch $N$.
- **2. Ketepatan Matematis (25 Poin)**: Perhitungan kombinasi linier, fungsi aktivasi, dan formula loss diverifikasi secara manual.
- **3. Implementasi Kode NumPy (25 Poin)**: Kode bersih, modular, berbasis OOP, dan bebas loop lambat pada sampel batch.
- **4. Pengujian & Stabilitas Numerik (20 Poin)**: Membuktikan ketahanan Softmax dan Cross-Entropy terhadap bahaya overflow/underflow.
- **5. Dokumentasi & Reproducibility (10 Poin)**: Kode memiliki docstring lengkap dan hasil deterministik (`seed=42`).

---

## 13. Referensi Primer Terverifikasi

1. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. ISBN: 978-0262035613. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
2. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer. ISBN: 978-0387310732. [https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/)
3. **Nielsen, M. A.** (2015). *Neural Networks and Deep Learning*. Determination Press. [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)
4. **Cybenko, G.** (1989). *Approximation by superpositions of a sigmoidal function*. Mathematics of Control, Signals, and Systems, 2(4), 303-314. [https://doi.org/10.1007/BF02551274](https://doi.org/10.1007/BF02551274)
5. **Rosenblatt, F.** (1958). *The perceptron: A probabilistic model for information storage and organization in the brain*. Psychological Review, 65(6), 386-408. [https://doi.org/10.1037/h0042519](https://doi.org/10.1037/h0042519)

