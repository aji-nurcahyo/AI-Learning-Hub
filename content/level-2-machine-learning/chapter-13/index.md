# Chapter 13: Supervised Learning — Regression Algorithms

Selamat datang di **Chapter 13: Supervised Learning — Regression Algorithms**. Bab ini merupakan pilar pertama dalam pemodelan pembelajaran terbimbing (*Supervised Learning*). Di bab ini, kita akan mempelajari algoritma regresi mulai dari fondasi matematis *Ordinary Least Squares* (OLS), regresi berganda multivariat, ekspansi polinomial untuk menangkap hubungan non-linear, hingga teknik regularisasi modern (*Ridge, Lasso, ElasticNet*) serta evaluasi diagnostik residual yang ketat.

---

## Daftar Isi
1. [Pengantar Konsep Regresi dalam Supervised Learning](#1-pengantar-konsep-regresi-dalam-supervised-learning)
2. [Perbedaan Mendasar: Regresi vs Klasifikasi](#2-perbedaan-mendasar-regresi-vs-klasifikasi)
3. [Karakteristik Variabel Target Kontinu](#3-karakteristik-variabel-target-kontinu)
4. [Simple Linear Regression (Regresi Linier Sederhana)](#4-simple-linear-regression-regresi-linier-sederhana)
5. [Multiple Linear Regression (Regresi Linier Berganda)](#5-multiple-linear-regression-regresi-linier-berganda)
6. [Persamaan Matematis Model Linier](#6-persamaan-matematis-model-linier)
7. [Interpretasi Koefisien dan Intersep](#7-interpretasi-koefisien-dan-intersep)
8. [Nilai Prediksi dan Ruang Hipotesis](#8-nilai-prediksi-dan-ruang-hipotesis)
9. [Definisi dan Karakteristik Galat Residual](#9-definisi-dan-karakteristik-galat-residual)
10. [Metode Estimasi Ordinary Least Squares (OLS)](#10-metode-estimasi-ordinary-least-squares-ols)
11. [Sum of Squared Errors (SSE / RSS)](#11-sum-of-squared-errors-sse--rss)
12. [Mean Squared Error (MSE) sebagai Fungsi Biaya](#12-mean-squared-error-mse-sebagai-fungsi-biaya)
13. [Tinjauan Optimasi: Solusi Analitis vs Gradient Descent](#13-tinjauan-optimasi-solusi-analitis-vs-gradient-descent)
14. [Asumsi-Asumsi Klasik Model Regresi Linier (Gauss-Markov)](#14-asumsi-asumsi-klasik-model-regresi-linier-gauss-markov)
15. [Ekspansi Fitur Polinomial (Polynomial Features)](#15-ekspansi-fitur-polinomial-polynomial-features)
16. [Polynomial Regression: Linearitas Parameter vs Non-Linearitas Fitur](#16-polynomial-regression-linearitas-parameter-vs-non-linearitas-fitur)
17. [Dilema Underfitting dan Overfitting pada Regresi](#17-dilema-underfitting-dan-overfitting-pada-regresi)
18. [Dekomposisi Bias–Variance Trade-off](#18-dekomposisi-biasvariance-trade-off)
19. [Prinsip dan Filosofi Regularisasi](#19-prinsip-dan-filosofi-regularisasi)
20. [Ridge Regression (Penalti Regularisasi L2)](#20-ridge-regression-penalti-regularisasi-l2)
21. [Lasso Regression (Penalti Regularisasi L1 & Sparsity)](#21-lasso-regression-penalti-regularisasi-l1--sparsity)
22. [ElasticNet Regression: Menyeimbangkan L1 dan L2](#22-elasticnet-regression-menyeimbangkan-l1-dan-l2)
23. [Keharusan Penskalaan Fitur Sebelum Regularisasi](#23-keharusan-penskalaan-fitur-sebelum-regularisasi)
24. [Mean Absolute Error (MAE)](#24-mean-absolute-error-mae)
25. [Mean Squared Error (MSE)](#25-mean-squared-error-mse)
26. [Root Mean Squared Error (RMSE)](#26-root-mean-squared-error-rmse)
27. [Koefisien Determinasi (R²) dan Fenomena R² Negatif](#27-koefisien-determinasi-r²-dan-fenomena-r²-negatif)
28. [Analisis Diagnostik Residual](#28-analisis-diagnostik-residual)
29. [Protokol Evaluasi Train, Validation, dan Test](#29-protokol-evaluasi-train-validation-dan-test)
30. [Cross-Validation Khusus Tugas Regresi](#30-cross-validation-khusus-tugas-regresi)
31. [Kerangka Perbandingan Model (Model Comparison)](#31-kerangka-perbandingan-model-model-comparison)
32. [Keterbatasan Model dan Jebakan Interpretasi Kausalitas](#32-keterbatasan-model-dan-jebakan-interpretasi-kausalitas)
33. [Aplikasi Nyata Regresi di Industri](#33-aplikasi-nyata-regresi-di-industri)
34. [Ringkasan Bab](#34-ringkasan-bab)
35. [Latihan Pemahaman Mandiri](#35-latihan-pemahaman-mandiri)
36. [Mini Project: Regression-Based Energy Consumption Prediction](#36-mini-project-regression-based-energy-consumption-prediction)
37. [Referensi Utama](#37-referensi-utama)

---

## 1. Pengantar Konsep Regresi dalam Supervised Learning

Dalam paradigma pembelajaran terbimbing (*Supervised Learning*), tugas pemodelan diarahkan untuk mempelajari fungsi pemetaan $f: \mathcal{X} \to \mathcal{Y}$ yang menghubungkan ruang vektor fitur masukan $X \in \mathbb{R}^D$ dengan ruang target keluaran $y$.

Ketika variabel target $y$ merupakan besaran numerik kontinu ($y \in \mathbb{R}$), tugas pemodelan tersebut diklasifikasikan sebagai **Regresi** (*Regression*). Istilah regresi pertama kali diperkenalkan oleh Sir Francis Galton pada abad ke-19 melalui fenomena *"regression toward mediocrity"* (anak dari orang tua bertubuh sangat tinggi cenderung memiliki tinggi badan yang mendekati rata-rata populasi). Dalam sains data modern, regresi merupakan fondasi utama peramalan kuantitatif, estimasi parameter fisik, dan pemodelan hubungan antar-variabel terukur.

---

## 2. Perbedaan Mendasar: Regresi vs Klasifikasi

Meskipun keduanya berada di bawah payung *Supervised Learning*, regresi dan klasifikasi memiliki perbedaan fundamental pada ruang target dan fungsi kerugian (*loss function*):

```text
+-----------------------------------+-----------------------------------+
|             REGRESI               |            KLASIFIKASI            |
+-----------------------------------+-----------------------------------+
| Target: Bilangan kontinu (R)      | Target: Label kategori diskrit    |
| Contoh: 342.8 kW, 27.5 °C, $45.2  | Contoh: Normal vs Rusak (0/1)     |
| Pertanyaan: "Berapa nilainya?"    | Pertanyaan: "Masuk kelompok apa?" |
| Fungsi Loss: Kuadratis / L1 (MSE) | Fungsi Loss: Cross-Entropy / Log  |
| Output: Garis / Bidang Hipotesis  | Output: Batas Keputusan (Boundary)|
+-----------------------------------+-----------------------------------+
```

---

## 3. Karakteristik Variabel Target Kontinu

Variabel kontinu memiliki sifat kardinalitas tak terhingga dalam suatu interval tertutup atau terbuka ($y \in [a, b]$ atau $y \in \mathbb{R}$). 
Implikasi utamanya:
1. Peluang model menebak nilai target tepat hingga presisi desimal tak terhingga secara identik adalah nol ($P(\hat{y} = y) = 0$).
2. Oleh karena itu, evaluasi regresi tidak pernah menggunakan metrik biner "benar/salah", melainkan mengukur besarnya jarak atau simpangan galat ($|y - \hat{y}|$) antara nilai taksiran dan kenyataan.

---

## 4. Simple Linear Regression (Regresi Linier Sederhana)

Regresi Linier Sederhana memodelkan hubungan garis lurus antara **satu fitur masukan tunggal** ($x$) dengan target ($y$):

$$\hat{y} = w x + b$$

Di mana:
- $w$ (*slope coefficient* / koefisien kemiringan): Perubahan rata-rata pada target $y$ untuk setiap kenaikan satu satuan fitur $x$.
- $b$ (*intercept* / titik potong sumbu-Y): Nilai harapan target ketika fitur $x = 0$.

```text
Target y ^
         |                 * (y_i)
         |                /| 
         |               / |  <-- Residual e_i = y_i - y_hat_i
         |              /  v 
         |             *------- Garis Prediksi y_hat = w*x + b
         |            /
         |    *      /
         |   b -----/
         +---------------------------------> Fitur x
```

---

## 5. Multiple Linear Regression (Regresi Linier Berganda)

Di dunia nyata, fenomena fisik jarang sekali dipicu oleh satu faktor tunggal. Konsumsi daya listrik pada turbin industri dipengaruhi secara simultan oleh suhu lingkungan, beban operasional, putaran mesin, tekanan fluida, dan keausan mekanik.

Regresi Linier Berganda memperluas model ke ruang multi-fitur ($D$ prediktor):

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_D x_D + b$$

Secara visual, jika $D = 2$, garis linier berubah menjadi bidang datar dua dimensi (*hyperplane*). Jika $D > 2$, fungsi prediksi membentuk bidang hipotesis berdimensi tinggi yang memotong ruang data.

---

## 6. Persamaan Matematis Model Linier

Dalam notasi aljabar linier vektor terpadu, dengan menyertakan suku bias $b$ ke dalam vektor bobot $\mathbf{w} = [b, w_1, w_2, \dots, w_D]^T$ dan menambahkan kolom 1 pada baris fitur $\mathbf{x} = [1, x_1, x_2, \dots, x_D]^T$:

$$\hat{y} = \mathbf{w}^T \mathbf{x} = \sum_{j=0}^D w_j x_j \quad (\text{dengan } x_0 = 1, w_0 = b)$$

Untuk keseluruhan matriks dataset $X \in \mathbb{R}^{N \times (D+1)}$:

$$\hat{\mathbf{y}} = X \mathbf{w}$$

---

## 7. Interpretasi Koefisien dan Intersep

Koefisien parsial $w_j$ mengindikasikan **efek marginal** dari fitur $x_j$ terhadap target, dengan asumsi seluruh fitur lainnya konstan (*ceteris paribus*):

$$w_j = \frac{\partial \hat{y}}{\partial x_j}$$

- Jika $w_{\text{load}} = 2.4640$ kW/%, artinya setiap kenaikan beban kerja sebesar $1\%$ diasosiasikan dengan kenaikan konsumsi daya sebesar $2.4640$ kW.
- Jika fitur diskalakan menggunakan `StandardScaler` (z-score), besaran $|w_j|$ merefleksikan kepentingan relatif fitur tersebut terhadap target.

---

## 8. Nilai Prediksi dan Ruang Hipotesis

Ruang hipotesis $\mathcal{H}$ dari regresi linier dibatasi secara ketat oleh kombinasi linear variabel masukan. Model tidak mampu melengkung secara mandiri kecuali ruang fitur ditransformasi secara eksplisit. Keterbatasan struktural ini memberikan keuntungan berupa ketahanan terhadap *overfitting* pada dataset kecil, namun membawa risiko *underfitting* jika fenomena yang dimodelkan memiliki sifat non-linearitas intrinsik yang kuat.

---

## 9. Definisi dan Karakteristik Galat Residual

**Residual** ($e_i$) adalah selisih antara nilai pengamatan aktual ($y_i$) dengan nilai yang ditaksir oleh model ($\hat{y}_i$):

$$e_i = y_i - \hat{y}_i$$

Residual merupakan komponen variansi data yang **gagal dijelaskan** oleh model. Dalam model yang dirancang dengan benar, residual harus merefleksikan derau acak murni (*white noise*) yang bebas dari pola terstruktur.

---

## 10. Metode Estimasi Ordinary Least Squares (OLS)

Prinsip **Ordinary Least Squares (OLS)**, yang dirumuskan oleh Carl Friedrich Gauss dan Adrien-Marie Legendre, mencari parameter $\mathbf{w}$ yang meminimalkan jumlah kuadrat dari seluruh galat residual:

$$\min_{\mathbf{w}} \sum_{i=1}^N (y_i - \hat{y}_i)^2 = \min_{\mathbf{w}} \|\mathbf{y} - X\mathbf{w}\|_2^2$$

---

## 11. Sum of Squared Errors (SSE / RSS)

Jumlah kuadrat residual (*Residual Sum of Squares*, RSS atau SSE):

$$\text{RSS}(\mathbf{w}) = \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2 = (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w})$$

Mengapa galat dikuadratkan, bukan dimutlakkan?
1. Fungsi kuadrat bersifat mulus dan dapat diturunkan secara analitis di setiap titik ($\mathcal{C}^\infty$).
2. Memberikan penalti yang progresif lebih berat pada galat besar, memaksa garis regresi mendekati pencilan.

---

## 12. Mean Squared Error (MSE) sebagai Fungsi Biaya

Untuk menstandarkan fungsi biaya terhadap jumlah sampel data $N$, digunakan **Mean Squared Error (MSE)**:

$$J(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

Dalam buku teks teoretis, faktor $\frac{1}{2N}$ sering digunakan untuk menyederhanakan perhitungan turunan:

$$J_{\text{teori}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - X\mathbf{w}\|_2^2$$

---

## 13. Tinjauan Optimasi: Solusi Analitis vs Gradient Descent

Terdapat dua pendekatan komputasi untuk menemukan nilai bobot optimal $\mathbf{w}^*$:

### 1. Solusi Bentuk Tertutup (Normal Equation)
Dengan menyamakan turunan gradien terhadap nol ($\nabla_{\mathbf{w}} J = \mathbf{0}$):

$$\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y}$$

- **Kelebihan**: Solusi eksak langsung ditemukan tanpa perlu tuning *learning rate* atau iterasi.
- **Kekurangan**: Menuntut inversi matriks berdimensi $(D \times D)$ dengan kompleksitas komputasi $\mathcal{O}(D^3)$. Jika fitur berjumlah jutaan ($D > 100.000$), pendekatan ini kehabisan memori RAM.

### 2. Gradient Descent Iteratif
Memperbarui bobot secara berulang berlawanan dengan arah gradien:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} J(\mathbf{w})$$

Di mana $\nabla_{\mathbf{w}} J = -\frac{2}{N} X^T (\mathbf{y} - X\mathbf{w})$. Skalabel untuk data berukuran masif.

---

## 14. Asumsi-Asumsi Klasik Model Regresi Linier (Gauss-Markov)

Teorema Gauss-Markov membuktikan bahwa estimator OLS bersifat **BLUE** (*Best Linear Unbiased Estimator*) jika empat asumsi dasar berikut terpenuhi:
1. **Linearitas dalam Parameter**: Hubungan antara fitur dan target bersifat linier terhadap koefisien bobot $\mathbf{w}$.
2. **Nilai Harapan Galat Nol**: $E[\mathbf{e} \mid X] = 0$. Tidak ada bias sistematis terarah pada prediksi.
3. **Homoskedastisitas & Tanpa Autokorelasi**: Varians residual bersifat konstan di seluruh skala prediksi ($\text{Var}(e_i) = \sigma^2$) dan galat antar-sampel saling bebas ($\text{Cov}(e_i, e_j) = 0$ untuk $i \neq j$).
4. **Ketiadaan Multikolinearitas Sempurna**: Kolom-kolom pada matriks $X$ saling bebas linier sehingga matriks $X^T X$ memiliki invers (*full rank*).

---

## 15. Ekspansi Fitur Polinomial (Polynomial Features)

Jika hubungan fisik antara fitur $x$ dan target $y$ melengkung (misal: konsumsi energi melonjak secara kuadratik saat beban mencapai kapasitas maksimal), kita dapat menciptakan fitur baru berupa pangkat dan perkalian silang fitur:

$$\phi(x) = [x, x^2, x^3, \dots, x^d]$$

Pada Scikit-learn, transformasi ini diotomasi oleh `PolynomialFeatures(degree=d)`.

---

## 16. Polynomial Regression: Linearitas Parameter vs Non-Linearitas Fitur

Satu konsep yang sering disalahpahami oleh pembelajar pemula adalah:
> *"Apakah Polynomial Regression termasuk model non-linear?"*

Secara visual terhadap ruang input asli $x$, garis prediksi melengkung secara non-linear. Namun **secara matematis dan statistik, Polynomial Regression tetaplah Model Linier**.

Mengapa? Karena fungsi prediksi tetap merupakan kombinasi linier terhadap **parameter bobot $\mathbf{w}$**:

$$\hat{y} = w_1 (x) + w_2 (x^2) + w_3 (x^3) + b = w_1 z_1 + w_2 z_2 + w_3 z_3 + b$$

Dengan melakukan substitusi $z_1 = x, z_2 = x^2, z_3 = x^3$, persamaan tersebut kembali menjadi Multiple Linear Regression biasa. Oleh karena itu, seluruh teori OLS dan solusi normal equation tetap berlaku seutuhnya.

---

## 17. Dilema Underfitting dan Overfitting pada Regresi

```text
[ DERAJAT 1: UNDERFITTING ]     [ DERAJAT 2: GOOD FIT ]       [ DERAJAT 15: OVERFITTING ]
        y ^                            y ^                           y ^
          |      *                       |      *                      |   /\ * /\
          |     /                        |    /   \                    |  /  \ /  \
          |    /  *                      |   /  *  \                   | *   v     *
          |   /                          |  /       \                  |
          +------------> x               +------------> x              +------------> x
      Bias Tinggi, Varians Rendah    Bias Rendah, Varians Rendah    Bias Rendah, Varians Tinggi
```

- **Underfitting (Bias Tinggi)**: Derajat polinomial terlalu rendah ($d=1$). Model terlalu kaku dan gagal menangkap kelengkungan data sebenarnya.
- **Overfitting (Varians Tinggi)**: Derajat polinomial terlalu tinggi ($d=15$). Model menghafal derau acak pada data latih, membentuk kurva osilasi liar yang menghasilkan galat katastropik pada data uji baru.

---

## 18. Dekomposisi Bias–Variance Trade-off

Ekspektasi galat kuadratik pada data uji baru dapat diuraikan secara analitis menjadi tiga komponen independen:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\text{Bias}[\hat{f}(x)]^2}_{\text{Kekakuan Asumsi}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{Sensitivitas Data}} + \underbrace{\sigma^2}_{\text{Derau Tak Terhindarkan}}$$

Tujuan rekayasa regresi bukanlah meminimalkan bias semata, melainkan menemukan titik kesetimbangan optimal (*sweet spot*) di mana total galat uji mencapai nilai terendah.

---

## 19. Prinsip dan Filosofi Regularisasi

Ketika model regresi memiliki terlalu banyak fitur atau derajat polinomial tinggi, koefisien bobot $\mathbf{w}$ cenderung meledak menjadi angka-angka yang sangat besar untuk saling membatalkan derau data.

**Regularisasi** adalah teknik membatasi kapasitas model dengan menyuntikkan suku penalti kompleksitas ke dalam fungsi biaya:

$$\mathcal{L}_{\text{reg}}(\mathbf{w}) = \text{Loss}(\mathbf{y}, X\mathbf{w}) + \lambda \cdot \Omega(\mathbf{w})$$

Penalti $\Omega(\mathbf{w})$ memaksa algoritma optimasi mencari kompromi antara mencocokkan data latih dengan mempertahankan nilai bobot tetap kecil dan teratur.

---

## 20. Ridge Regression (Penalti Regularisasi L2)

**Ridge Regression** (Tikhonov Regularization) menggunakan penalti kuadratik norma $L_2$:

$$J_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - X\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2 = \frac{1}{2N} \sum_{i=1}^N (y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \alpha \sum_{j=1}^D w_j^2$$

### Karakteristik Kunci Ridge
1. **Coefficient Shrinkage**: Bobot koefisien disusutkan mendekati nol secara halus (*smooth shrinkage*), tetapi **tidak pernah menjadi tepat nol**.
2. **Solusi Analitis Stabil**: Menambahkan $\alpha I$ ke matriks normal equation: $\mathbf{w}^* = (X^T X + \alpha I)^{-1} X^T \mathbf{y}$, menjamin invers matriks selalu ada bahkan jika fitur mengalami multikolinearitas sempurna.

---

## 21. Lasso Regression (Penalti Regularisasi L1 & Sparsity)

**Lasso** (*Least Absolute Shrinkage and Selection Operator*) menggunakan penalti nilai mutlak norma $L_1$:

$$J_{\text{Lasso}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - X\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1 = \frac{1}{2N} \sum_{i=1}^N (y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \alpha \sum_{j=1}^D |w_j|$$

```text
GEOMETRI PENALTI REGULARISASI:
        Lasso (L1): Belah Ketupat             Ridge (L2): Lingkaran Mulus
                 w2 ^                                   w2 ^
                    |                                      |      Kontur OLS
                 /\ |                                    .-+-.   (   *   )
                /  \|   Kontur OLS                      /  |  \ (         )
               /    +-- (   *   )                      |   +---+-----------
              /    /|  (         )                     \  /   /
             /    / |                                   '-+-'
            +----+--+------> w1                           +--------------> w1
            
Titik potong terjadi tepat di SUDUT (w2 = 0)   Titik potong terjadi di sembarang BUSUR
    --> MENGHASILKAN MODEL RENGGANG!               --> KOEFISIEN TIDAK PERNAH NOL!
```

Karena kontur penalti $L_1$ memiliki sudut tajam pada sumbu-sumbu koordinat, optimasi sering kali memotong kontur tepat pada sumbu, **memaksa koefisien fitur yang kurang penting menjadi tepat $0.0000$**. Ini menjadikan Lasso sebagai metode seleksi fitur otomatis (*embedded feature selection*).

---

## 22. ElasticNet Regression: Menyeimbangkan L1 dan L2

Lasso memiliki dua kelemahan praktis:
1. Jika terdapat sekelompok fitur yang berkorelasi tinggi, Lasso cenderung memilih satu fitur secara acak dan mengabaikan sisanya.
2. Jika jumlah fitur lebih besar dari jumlah sampel ($D > N$), Lasso hanya mampu memilih maksimal $N$ fitur.

**ElasticNet** memadukan penalti $L_1$ dan $L_2$ secara konveks:

$$J_{\text{ElasticNet}}(\mathbf{w}) = \frac{1}{2N} \|\mathbf{y} - X\mathbf{w}\|_2^2 + \alpha \cdot \rho \|\mathbf{w}\|_1 + \frac{\alpha (1 - \rho)}{2} \|\mathbf{w}\|_2^2$$

Di mana $\rho$ (`l1_ratio` pada Scikit-learn) mengontrol perimbangan: $\rho = 1$ murni Lasso, $\rho = 0$ murni Ridge.

---

## 23. Keharusan Penskalaan Fitur Sebelum Regularisasi

Perhatikan suku penalti $\alpha \sum w_j^2$. Penalti ini tidak memperhitungkan satuan fisik variabel.
- Jika fitur $x_1$ diukur dalam meter (rentang $1 - 10$) dan $x_2$ diukur dalam milimeter (rentang $1000 - 10000$), maka koefisien $w_1$ secara alami bernilai ribuan kali lebih besar daripada $w_2$.
- Akibatnya, regularisasi akan memberikan penalti yang sangat tidak adil pada $w_1$, merusak hubungan fisik yang mendasarinya.
- **Aturan Baku**: Selalu gunakan `StandardScaler` (fit strictly pada data latih di dalam Pipeline) sebelum menerapkan Ridge, Lasso, atau ElasticNet.

---

## 24. Mean Absolute Error (MAE)

Rata-rata selisih nilai mutlak antara target aktual dan estimasi model:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

- **Keunggulan**: Memiliki interpretasi linier yang sangat intuitif dalam satuan asli target (kW).
- **Ketahanan**: Kebal terhadap pencilan ekstrem (*robust to outliers*) karena tidak menguadratkan galat.

---

## 25. Mean Squared Error (MSE)

Rata-rata kuadrat selisih antara nilai aktual dan prediksi:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

- **Keunggulan**: Mulus secara diferensial untuk optimasi matematis.
- **Kelemahan**: Satuannya berkuadrat ($\text{kW}^2$), menyulitkan interpretasi intuitif bagi pemangku kepentingan non-teknis.

---

## 26. Root Mean Squared Error (RMSE)

Akar kuadrat dari Mean Squared Error:

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

- Mengembalikan skala galat kuadratik ke satuan asli data ($kW$).
- Nilai $\text{RMSE} \ge \text{MAE}$. Semakin besar jarak selisih antara RMSE dan MAE, semakin banyak variasi galat besar atau pencilan pada model.

---

## 27. Koefisien Determinasi (R²) dan Fenomena R² Negatif

Koefisien determinasi mengukur proporsi variansi total variabel target yang berhasil dijelaskan oleh fitur-fitur model:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

Di mana $\text{SS}_{\text{tot}}$ adalah variansi total data terhadap rata-rata $\bar{y}$.

### Peringatan Kritis: Mitos Nilai R²
1. **$R^2$ Tinggi Tidak Otomatis Berarti Model Baik**: Model polinomial derajat 15 yang mengalami overfitting ekstrem pada data latih dapat memiliki $R^2 = 0.9999$, namun gagal total pada data uji.
2. **$R^2$ pada Data Uji Bisa Bernilai Negatif ($R^2 < 0$)**:
   - Jika model dievaluasi pada data uji baru (*out-of-sample*) dan prediksinya sangat melenceng, nilai $\text{SS}_{\text{res}}$ akan melampaui $\text{SS}_{\text{tot}}$.
   - Nilai $R^2 = -14.20$ berarti model tersebut berkinerja **jauh lebih buruk daripada sekadar menebak rata-rata historis** $\bar{y}$!

---

## 28. Analisis Diagnostik Residual

Sebelum model regresi dinyatakan siap produksi, residual wajib diuji melalui tiga pemeriksaan diagnostik:

```text
[ PLOT RESIDUAL VS FITTED VALUES: HOMOSKEDASTISITAS ]

      Residual e ^
                 |     *       *    *       *
                 |  *      *            *       *  <-- Sebaran Acak Seragam
                 +---------------------------------> Prediksi y_hat (IDEAL!)
                 |    *     *     *         *
                 |         *            *

      Residual e ^
                 |              *    *   *   *
                 |        *   *   *           *    <-- Pola Terompet (Heteroskedastik!)
                 +---------------------------------> Varians Membesar di Beban Tinggi
                 |   *    *
                 | *
```

1. **Uji Nilai Harapan Nol**: Rata-rata residual pada data uji harus mendekati nol ($\bar{e} \approx 0$).
2. **Uji Homoskedastisitas**: Varians sebaran titik residual tidak boleh membentuk corong/terompet seiring bertambahnya nilai prediksi $\hat{y}$.
3. **Uji Normalitas Galat**: Histogram residual harus simetris (kemencengan *skewness* $\approx 0$) tanpa ekor pencilan ekstrem.

---

## 29. Protokol Evaluasi Train, Validation, dan Test

Pencegahan kebocoran data (*data leakage*) mutlak ditegakkan dalam regresi:
1. **Pemisahan Partisi Dilakukan Pertama Kali**: Jangan pernah menghitung parameter skala (`StandardScaler.fit()`) sebelum pemisahan data selesai.
2. **Tuning Hiperparameter $\alpha$**: Nilai penalti Ridge atau Lasso dipilih berdasarkan performa pada *Validation Set* atau melalui *K-Fold Cross-Validation* data latih.
3. **Kunci Data Uji (Test Set)**: Partisi uji hanya disentuh satu kali pada akhir evaluasi untuk mengukur kapabilitas generalisasi sejati.

---

## 30. Cross-Validation Khusus Tugas Regresi

Berbeda dengan klasifikasi yang memerlukan stratifikasi label diskrit, validasi silang pada regresi menggunakan **Standard K-Fold Cross-Validation** dengan pengacakan acak (`shuffle=True, random_state=42`):

```text
5-Fold CV Regresi:
Fold 1: [ Test ] [ Train ] [ Train ] [ Train ] [ Train ] -> RMSE = 6.8 kW
Fold 2: [ Train ] [ Test ] [ Train ] [ Train ] [ Train ] -> RMSE = 5.4 kW
Fold 3: [ Train ] [ Train ] [ Test ] [ Train ] [ Train ] -> RMSE = 6.1 kW
Fold 4: [ Train ] [ Train ] [ Train ] [ Test ] [ Train ] -> RMSE = 5.9 kW
Fold 5: [ Train ] [ Train ] [ Train ] [ Train ] [ Test ] -> RMSE = 6.6 kW

Estimasi Performa Generalisasi = 6.17 (+/- 0.90) kW
```

---

## 31. Kerangka Perbandingan Model (Model Comparison)

Membandingkan berbagai keluarga algoritma regresi pada dataset telemetri konsumsi daya industri:

| Arsitektur Model | CV RMSE (kW) | Test RMSE (kW) | Test R² | Keunggulan & Batasan |
|---|---|---|---|---|
| **Dummy Mean Baseline** | $82.68 \pm 6.07$ | $76.65$ | $-0.0054$ | Patokan kinerja terendah tanpa sinyal fitur. |
| **Linear Regression (OLS)** | $6.17 \pm 0.90$ | $5.37$ | $0.9951$ | Sangat cepat, interpretabel, namun mengabaikan kelengkungan termal. |
| **Polynomial Reg (Deg 2)** | $5.47 \pm 0.61$ | $4.12$ | $0.9971$ | **Model Terbaik**: Menangkap efek kuadratik pemanasan beban. |
| **Ridge Regression (L2)** | $7.36 \pm 1.00$ | $6.74$ | $0.9922$ | Kebal multikolinearitas, menyusutkan koefisien secara stabil. |
| **Lasso Regression (L1)** | $6.22 \pm 1.10$ | $5.51$ | $0.9948$ | Menghasilkan model renggang; membuang fitur getaran berlebih. |
| **ElasticNet (L1+L2)** | $10.52 \pm 1.37$ | $9.72$ | $0.9838$ | Penalti ganda; membutuhkan tuning $\alpha$ dan $\rho$ lebih lanjut. |

---

## 32. Keterbatasan Model dan Jebakan Interpretasi Kausalitas

Dua jebakan kritis yang wajib dihindari oleh praktisi kecerdasan buatan:
1. **Korelasi Bukan Kausalitas (*Correlation is not Causation*)**:
   Koefisien positif pada `vibration_severity_mms` ($w = +5.28$ kW per mm/s) tidak berarti bahwa kita dapat menghemat listrik pabrik dengan sengaja meredam sensor getaran secara paksa. Getaran tinggi dan lonjakan daya adalah gejala bersama dari mesin yang bekerja di bawah beban ekstrem.
2. **Bahaya Ekstrapolasi di Luar Jangkauan Data**:
   Model regresi yang dilatih pada rentang suhu $16^\circ\text{C} - 44^\circ\text{C}$ tidak dapat dipercaya jika digunakan untuk memprediksi konsumsi daya di kutub utara pada suhu $-30^\circ\text{C}$.

---

## 33. Aplikasi Nyata Regresi di Industri

1. **Smart Grid & Manajemen Beban Energi**: Memprediksi kebutuhan daya listrik puncak pabrik 24 jam ke depan untuk mencegah denda kelebihan beban dari penyedia listrik.
2. **Estimasi Umur Pakai Sisa (*Remaining Useful Life / RUL*)**: Memperkirakan berapa ratus jam operasional yang tersisa pada bantalan turbin sebelum mengalami kegagalan fatal.
3. **Optimasi Proses Manufaktur Kimia**: Menyesuaikan rasio campuran reaktan secara presisi untuk memaksimalkan hasil rendemen (*yield*) pada bejana reaktor.

---

## 34. Ringkasan Bab

- Regresi memetakan fitur masukan ke variabel target numerik kontinu.
- OLS meminimalkan kuadrat selisih residual, menghasilkan estimator tak bias dengan varians minimum jika asumsi Gauss-Markov terpenuhi.
- Polynomial Regression memodelkan relasi non-linear input namun tetap mempertahankan linearitas terhadap parameter bobot.
- Regularisasi mengorbankan sedikit bias demi memangkas varians secara masif: Ridge menyusutkan bobot secara halus, Lasso menciptakan sparsity terintegrasi, dan ElasticNet menggabungkan keduanya.
- Penskalaan fitur menggunakan `StandardScaler` mutlak diwajibkan sebelum regularisasi.
- Metrik evaluasi MAE, MSE, dan RMSE mengukur besaran galat dalam satuan fisik, sedangkan $R^2$ mengukur proporsi variansi yang dapat bernilai negatif pada pengujian *out-of-sample*.

---

## 35. Latihan Pemahaman Mandiri

### Kategori A: Pemahaman Konseptual (Tingkat Dasar)
1. Diberikan model regresi linier sederhana $\hat{y} = 2.5 x + 10$. Jika sebuah sampel memiliki nilai $x = 8$ dan nilai aktual $y = 35$, hitung nilai prediksi $\hat{y}$ dan nilai residual $e$!
2. Mengapa kita tidak boleh menyimpulkan bahwa fitur dengan koefisien regresi terbesar adalah fitur yang paling penting jika data belum distandarisasi?
3. Jelaskan mengapa nilai $R^2$ pada data uji dapat bernilai negatif! Apa artinya jika model Anda menghasilkan $R^2 = -0.50$?

### Kategori B: Aplikasi Metodologis (Tingkat Menengah)
4. Sebuah tim data science mengamati bahwa model regresi linier derajat 5 mereka memiliki Train RMSE sebesar $2.1$ kW, namun Test RMSE sebesar $28.4$ kW. Jelaskan fenomena apa yang sedang terjadi dan sebutkan dua solusi teknis untuk memperbaikinya!
5. Jelaskan perbedaan mendasar antara sifat penyusutan koefisien pada Ridge Regression dan Lasso Regression berdasarkan bentuk geometri penaltinya!

### Kategori C: Analisis Tingkat Lanjut (Tingkat Lanjut)
6. Pada pengujian residual suatu model regresi, ditemukan bahwa sebaran residual membentuk pola corong (varians membesar seiring naiknya beban operasional). Asumsi Gauss-Markov mana yang dilanggar, apa konsekuensinya terhadap estimasi interval kepercayaan, dan teknik transformasi apa yang dapat diterapkan untuk mengatasinya?

---

## 36. Mini Project: Regression-Based Energy Consumption Prediction

### Deskripsi Tugas
Kembangkan sistem pemodelan regresi terintegrasi (*End-to-End Regression Pipeline*) untuk memprediksi konsumsi daya listrik peralatan industri berbasis dataset telemetri `equipment_power_demand.csv`.

### Persyaratan Teknis
1. **Eksplorasi & Pembagian Data**: Lakukan split data 80% Train dan 20% Test secara bersih. Dokumentasikan statistik deskriptif awal target.
2. **Baseline Model**: Bangun `DummyRegressor(strategy='mean')` sebagai tolok ukur kinerja dasar.
3. **Implementasi Multi-Model**: Bangun minimal dua arsitektur regresi yang berbeda (misal: Multiple Linear Regression vs Polynomial Regression Derajat 2 ter-regularisasi Ridge).
4. **Enkapsulasi Pipeline Bebas Leakage**: Seluruh penskalaan fitur (`StandardScaler`) dan ekspansi polinomial wajib terbungkus di dalam `sklearn.pipeline.Pipeline`.
5. **Evaluasi Validasi Silang**: Lakukan 5-Fold Cross-Validation pada data latih untuk memilih hiperparameter $\alpha$ terbaik.
6. **Diagnostik Residual**: Buat analisis residual pada data uji akhir mencakup nilai mean residual, homoskedastisitas, dan normalitas sebaran galat.

### Rubrik Penilaian Mini Project (100 Poin)

| Komponen Penilaian | Bobot | Kriteria Evaluasi Capaian |
|---|---|---|
| **1. Problem & Data Understanding** | 15 Poin | Mampu merumuskan masalah regresi kontinu, mengidentifikasi satuan fitur/target, dan memetakan batas operasional fisik data. |
| **2. Model Implementation** | 25 Poin | Mengimplementasikan minimal dua model regresi berbasis Scikit-learn Pipeline; penskalaan fitur terisolasi sempurna pada data latih. |
| **3. Validation & Metrics Correctness** | 25 Poin | Melakukan 5-Fold CV tanpa kebocoran; menyajikan metrik MAE, MSE, RMSE, dan $R^2$ secara akurat pada data uji. |
| **4. Residual & Error Analysis** | 20 Poin | Melakukan pengujian asumsi residual secara empiris (mean galat mendekati nol, uji homoskedastisitas tertil, dan analisis keterbatasan model). |
| **5. Documentation & References** | 15 Poin | Menyajikan kode yang dapat direproduksi (*fully reproducible*), tabel hasil komparatif yang rapi, dan mencantumkan referensi akademik primer. |

---

## 37. Referensi Utama

1. **Géron, A. (2022).** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd Edition. O'Reilly Media. [https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
2. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd Edition. Springer. [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/)
3. **James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021).** *An Introduction to Statistical Learning with Applications in R/Python*, 2nd Edition. Springer. [https://www.statlearning.com/](https://www.statlearning.com/)
4. **Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021).** *Introduction to Linear Regression Analysis*, 6th Edition. John Wiley & Sons. [https://www.wiley.com/en-us/Introduction+to+Linear+Regression+Analysis%2C+6th+Edition-p-9781119578727](https://www.wiley.com/en-us/Introduction+to+Linear+Regression+Analysis%2C+6th+Edition-p-9781119578727)
5. **ACM / IEEE-CS / AAAI. (2024).** *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. ACM Digital Library. [https://csed.acm.org/](https://csed.acm.org/)

