# Chapter 14: Supervised Learning — Classification Algorithms

Selamat datang di **Chapter 14: Supervised Learning — Classification Algorithms**. Bab ini merupakan pilar kedua dalam pembelajaran terbimbing (*Supervised Learning*). Jika pada bab sebelumnya kita memprediksi besaran numerik kontinu, pada bab ini kita fokus pada tugas memprediksi label kategori diskrit. Kita akan mempelajari fondasi matematika dan komputasi regresi logistik (*Logistic Regression*), pengklasifikasi probabilistik Naive Bayes (*Gaussian & Multinomial*), kalibrasi ambang batas keputusan (*decision threshold*), matriks konfusi, metrik evaluasi komprehensif (*Precision, Recall, F1, ROC-AUC, PR-AUC*), serta penanganan data tidak seimbang (*class imbalance*).

---

## Daftar Isi
1. [Pengantar Konsep Klasifikasi dalam Supervised Learning](#1-pengantar-konsep-klasifikasi-dalam-supervised-learning)
2. [Perbandingan Paradigma: Klasifikasi vs Regresi](#2-perbandingan-paradigma-klasifikasi-vs-regresi)
3. [Klasifikasi Biner (Binary Classification)](#3-klasifikasi-biner-binary-classification)
4. [Klasifikasi Multikelas (Multiclass Classification)](#4-klasifikasi-multikelas-multiclass-classification)
5. [Struktur Fitur dan Label pada Tugas Klasifikasi](#5-struktur-fitur-dan-label-pada-tugas-klasifikasi)
6. [Konsep Batas Keputusan (Decision Boundary)](#6-konsep-batas-keputusan-decision-boundary)
7. [Algoritma Regresi Logistik (Logistic Regression)](#7-algoritma-regresi-logistik-logistic-regression)
8. [Fungsi Aktivasi Sigmoid dan Properti Matematisnya](#8-fungsi-aktivasi-sigmoid-dan-properti-matematisnya)
9. [Formulasi Log-Odds (Logit) dan Rasio Odds](#9-formulasi-log-odds-logit-dan-rasio-odds)
10. [Ambang Batas Keputusan (Decision Threshold)](#10-ambang-batas-keputusan-decision-threshold)
11. [Fungsi Biaya Logistic Loss (Binary Cross-Entropy)](#11-fungsi-biaya-logistic-loss-binary-cross-entropy)
12. [Regresi Logistik Multikelas: Softmax vs One-vs-Rest](#12-regresi-logistik-multikelas-softmax-vs-one-vs-rest)
13. [Fondasi Teorema Bayes dalam Pembelajaran Mesin](#13-fondasi-teorema-bayes-dalam-pembelajaran-mesin)
14. [Asumsi Independensi Bersyarat (Conditional Independence)](#14-asumsi-independensi-bersyarat-conditional-independence)
15. [Gaussian Naive Bayes untuk Fitur Kontinu](#15-gaussian-naive-bayes-untuk-fitur-kontinu)
16. [Multinomial Naive Bayes dan Laplace Smoothing](#16-multinomial-naive-bayes-dan-laplace-smoothing)
17. [Probabilitas Kelas: Prior, Likelihood, Evidence, dan Posterior](#17-probabilitas-kelas-prior-likelihood-evidence-dan-posterior)
18. [Perbedaan Mendasar: Prediksi Kelas vs Estimasi Probabilitas](#18-perbedaan-mendasar-prediksi-kelas-vs-estimasi-probabilitas)
19. [Matriks Konfusi (Confusion Matrix) dan Struktur Kuadran](#19-matriks-konfusi-confusion-matrix-dan-struktur-kuadran)
20. [Akurasi (Accuracy) dan Paradoks Akurasi](#20-akurasi-accuracy-dan-paradoks-akurasi)
21. [Presisi (Precision) dan Pengendalian False Positive](#21-presisi-precision-dan-pengendalian-false-positive)
22. [Recall (Sensitivitas / TPR) dan Pengendalian False Negative](#22-recall-sensitivitas--tpr-dan-pengendalian-false-negative)
23. [Spesifisitas (Specificity / TNR) dan False Positive Rate (FPR)](#23-spesifisitas-specificity--tnr-dan-false-positive-rate-fpr)
24. [F1-Score: Rata-rata Harmonik Presisi dan Recall](#24-f1-score-rata-rata-harmonik-presisi-dan-recall)
25. [Ketidakseimbangan Kelas (Class Imbalance)](#25-ketidakseimbangan-kelas-class-imbalance)
26. [Metodologi Averaging Multikelas: Macro, Micro, dan Weighted](#26-metodologi-averaging-multikelas-macro-micro-dan-weighted)
27. [Kurva Receiver Operating Characteristic (ROC)](#27-kurva-receiver-operating-characteristic-roc)
28. [Area Under the ROC Curve (ROC-AUC)](#28-area-under-the-roc-curve-roc-auc)
29. [Kurva Precision-Recall (PR) dan Average Precision (AP)](#29-kurva-precision-recall-pr-dan-average-precision-ap)
30. [Pemilihan Ambang Keputusan Berbasis Biaya Bisnis](#30-pemilihan-ambang-keputusan-berbasis-biaya-bisnis)
31. [Tinjauan Kalibrasi Probabilitas (Probability Calibration)](#31-tinjauan-kalibrasi-probabilitas-probability-calibration)
32. [Validasi Silang Terstratifikasi (Stratified Cross-Validation)](#32-validasi-silang-terstratifikasi-stratified-cross-validation)
33. [Analisis Galat Klasifikasi (Error Analysis)](#33-analisis-galat-klasifikasi-error-analysis)
34. [Etika dan Klasifikasi yang Bertanggung Jawab](#34-etika-dan-klasifikasi-yang-bertanggung-jawab)
35. [Ringkasan Bab](#35-ringkasan-bab)
36. [Latihan Pemahaman Mandiri](#36-latihan-pemahaman-mandiri)
37. [Mini Project: Classification-Based Equipment Fault Identification](#37-mini-project-classification-based-equipment-fault-identification)
38. [Referensi Utama](#38-referensi-utama)

---

## 1. Pengantar Konsep Klasifikasi dalam Supervised Learning

Klasifikasi (*Classification*) adalah tugas memetakan ruang vektor fitur masukan $X \in \mathbb{R}^D$ ke dalam himpunan label kategori diskrit yang saling lepas (*mutually exclusive categories*):

$$f: \mathcal{X} \to \mathcal{C}, \quad \mathcal{C} = \{c_1, c_2, \dots, c_K\}$$

Di mana $K$ adalah jumlah kelas yang telah ditentukan sebelumnya (*predefined classes*). Berbeda dengan regresi yang mengestimasi besaran numerik tak terhingga, klasifikasi berupaya memisahkan ruang fitur menjadi wilayah-wilayah keputusan (*decision regions*) yang terikat pada masing-masing label.

---

## 2. Perbandingan Paradigma: Klasifikasi vs Regresi

```text
+-----------------------------------+-----------------------------------+
|             REGRESI               |            KLASIFIKASI            |
+-----------------------------------+-----------------------------------+
| Target: Bilangan kontinu (R)      | Target: Label kategori diskrit (C)|
| Ruang Keluaran: Tak berhingga     | Ruang Keluaran: K kelas terhingga |
| Contoh: 342.8 kW konsumsi daya    | Contoh: Normal, Kerusakan Bantalan|
| Fungsi Objektif: Minimalkan MSE   | Fungsi Objektif: Cross-Entropy    |
| Evaluasi: MAE, RMSE, R^2          | Evaluasi: Precision, Recall, F1   |
| Batas Keputusan: Tidak ada        | Batas Keputusan: Hyperplane / NonL|
+-----------------------------------+-----------------------------------+
```

---

## 3. Klasifikasi Biner (Binary Classification)

Klasifikasi biner merupakan bentuk klasifikasi paling mendasar di mana jumlah kelas tepat dua ($K = 2$):

$$y \in \{0, 1\}$$

Secara konvensi standar internasional:
- Kelas $0$ (*Negative Class*): Merepresentasikan kondisi lazim, normal, tidak aktif, atau status dasar sistem.
- Kelas $1$ (*Positive Class*): Merepresentasikan kondisi langka, anomali, kerusakan, atau peristiwa penting yang sedang dicari (misal: alarm kebakaran, penipuan transaksi, atau kerusakan mekanik).

---

## 4. Klasifikasi Multikelas (Multiclass Classification)

Ketika target memiliki lebih dari dua kategori diskrit ($K > 2$):

$$y \in \{0, 1, 2, \dots, K-1\}$$

Contoh industri: Diagnosis kegagalan turbin yang membedakan apakah mesin berada pada status $0$: `NORMAL`, $1$: `BEARING_FAULT`, $2$: `OVERHEATING`, atau $3$: `ELECTRICAL_SURGE`. Setiap sampel hanya boleh memiliki tepat satu label benar. (Jika satu sampel dapat memiliki banyak label sekaligus, masalah tersebut disebut *Multi-label Classification*).

---

## 5. Struktur Fitur dan Label pada Tugas Klasifikasi

Data masukan disusun dalam format matriks tabular:
- Matriks Fitur $X \in \mathbb{R}^{N \times D}$: Berisi $N$ sampel dengan $D$ variabel terukur (misal: suhu $^\circ\text{C}$, getaran RMS, kebisingan dB, fluktuasi voltase).
- Vektor Label $y \in \{0, 1, \dots, K-1\}^N$: Berisi indeks bilangan bulat yang merepresentasikan kategori masing-masing baris observasi.

---

## 6. Konsep Batas Keputusan (Decision Boundary)

**Batas Keputusan** (*Decision Boundary*) adalah garis, kurva, atau bidang hipersurface pada ruang fitur di mana probabilitas model menghasilkan keputusan setara antar-kelas ($P(y=1|x) = P(y=0|x) = 0.5$):

```text
Fitur 2 (Getaran) ^
                  |     *  *  (Kelas 1: Rusak)
                  |   *   *    *
                  |----------/------------------ Batas Keputusan: w^T x + b = 0
                  |         /    o   o
                  |       o   o    o  (Kelas 0: Normal)
                  +---------------------------------> Fitur 1 (Suhu)
```

Pada model linier (seperti Regresi Logistik), batas keputusan selalu berbentuk bidang datar linier ($\mathbf{w}^T \mathbf{x} + b = 0$). Pada model non-linier (seperti Decision Tree atau SVM Kernel), batas keputusan dapat melengkung secara fleksibel.

---

## 7. Algoritma Regresi Logistik (Logistic Regression)

Meskipun menggunakan kata "regresi" dalam namanya karena alasan historis pemodelan statistika, **Logistic Regression adalah algoritma klasifikasi**.

Algoritma ini menghitung kombinasi linier fitur $z = \mathbf{w}^T \mathbf{x} + b$, kemudian memetakan nilai $z \in (-\infty, +\infty)$ ke dalam interval probabilitas yang valid $[0, 1]$ menggunakan fungsi non-linear khusus.

---

## 8. Fungsi Aktivasi Sigmoid dan Properti Matematisnya

Fungsi **Sigmoid** (atau fungsi logistik standar) didefinisikan sebagai:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}$$

```text
Probabilitas P(y=1) ^
                1.0 |                   .-------- (Jenuh Positif)
                    |                 .'
                0.5 |-------+-------.'  <-- Titik Ambang (z = 0)
                    |     .'
                0.0 | ---'               <-- (Jenuh Negatif)
                    +-------+-------+-------> Logit z
                           z=0
```

### Properti Analitis Kunci
1. **Rentang Nilai Terbatas**: $\forall z \in \mathbb{R}, \; 0 < \sigma(z) < 1$.
2. **Titik Tengah Simetris**: $\sigma(0) = 0.5$ dan $\sigma(-z) = 1 - \sigma(z)$.
3. **Turunan Mulus yang Efisien**: $\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$.

---

## 9. Formulasi Log-Odds (Logit) dan Rasio Odds

Dalam statistika, **Odds** adalah rasio antara probabilitas terjadinya suatu peristiwa ($p$) terhadap probabilitas tidak terjadinya peristiwa tersebut ($1 - p$):

$$\text{Odds} = \frac{p}{1 - p} = \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})}$$

Dengan mensubstitusikan $p = \sigma(z)$, kita memperoleh hubungan **Log-Odds** atau fungsi **Logit**:

$$\ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^T \mathbf{x} + b$$

### Interpretasi Odds Ratio $\exp(w_j)$
Setiap kenaikan 1 satuan pada fitur $x_j$ akan melipatgandakan nilai Odds sebesar faktor pengali $\exp(w_j)$:
- Jika $w_{\text{vibration}} = 1.7575$, maka $\exp(w) = 5.7979$. Artinya, setiap kenaikan getaran sebesar 1 mm/s melipatgandakan peluang terjadinya kerusakan mesin sebesar **$5.80\times$**.

---

## 10. Ambang Batas Keputusan (Decision Threshold)

Model regresi logistik menghasilkan estimasi probabilitas kontinu $\hat{p} = P(y=1 \mid \mathbf{x})$. Untuk mengubahnya menjadi keputusan diskrit $\hat{y} \in \{0, 1\}$, diterapkan **Decision Threshold** ($T$):

$$\hat{y} = \begin{cases} 1, & \text{jika } \hat{p} \ge T \\ 0, & \text{jika } \hat{p} < T \end{cases}$$

Nilai default pada pustaka perangkat lunak adalah $T = 0.5$. Namun, **angka 0.5 bukanlah hukum mutlak**. Ambang batas dapat digeser sesuai dengan pertimbangan kerugian bisnis operasional.

---

## 11. Fungsi Biaya Logistic Loss (Binary Cross-Entropy)

Kita tidak dapat menggunakan Mean Squared Error (MSE) pada regresi logistik karena komposisi MSE dengan fungsi Sigmoid akan menghasilkan permukaan biaya yang tidak cembung (*non-convex*) dengan banyak titik minimum lokal.

Fungsi biaya yang tepat adalah **Binary Cross-Entropy Loss** (atau *Log Loss*), yang diturunkan dari prinsip *Maximum Likelihood Estimation* (MLE):

$$J(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

- Jika $y_i = 1$ dan model memprediksi $\hat{p}_i \to 0$, penalti $-\ln(\hat{p}_i) \to +\infty$.
- Jika $y_i = 0$ dan model memprediksi $\hat{p}_i \to 1$, penalti $-\ln(1 - \hat{p}_i) \to +\infty$.

---

## 12. Regresi Logistik Multikelas: Softmax vs One-vs-Rest

Untuk klasifikasi $K > 2$ kelas, terdapat dua strategi arsitektur utama:

### 1. Multinomial Softmax Regression
Menghitung logit linier terpisah untuk setiap kelas: $z_k = \mathbf{w}_k^T \mathbf{x} + b_k$. Probabilitas dinormalisasi menggunakan fungsi **Softmax**:

$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

Penyebut $\sum_{j=1}^K e^{z_j}$ menjamin bahwa penjumlahan seluruh probabilitas kelas tepat bernilai $1.0$.

### 2. One-vs-Rest (OvR / One-vs-All)
Melatih $K$ buah pengklasifikasi biner terpisah: Pengklasifikasi ke-$k$ membedakan antara "Kelas $k$" versus "Seluruh Kelas Selain $k$". Kelas akhir dipilih berdasarkan nilai probabilitas tertinggi ($\arg\max$).

---

## 13. Fondasi Teorema Bayes dalam Pembelajaran Mesin

Teorema Bayes membalik arah penalaran probabilistik dari penyebab ke akibat:

$$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}$$

Dalam konteks klasifikasi:
- $P(y \mid X)$ (**Posterior**): Peluang status mesin adalah $y$ setelah mengamati pembacaan sensor $X$.
- $P(y)$ (**Prior**): Peluang awal status mesin sebelum melihat sensor (prevalensi historis).
- $P(X \mid y)$ (**Likelihood**): Peluang sensor menghasilkan nilai $X$ jika status mesin diketahui $y$.
- $P(X)$ (**Evidence**): Konstanta normalisasi agar total probabilitas posterior bernilai 1.

---

## 14. Asumsi Independensi Bersyarat (Conditional Independence)

Menghitung likelihood gabungan $P(x_1, x_2, \dots, x_D \mid y)$ pada ruang dimensi tinggi membutuhkan miliaran kombinasi data yang mustahil tersedia.

Pengklasifikasi **Naive Bayes** membuat asumsi penyederhanaan yang berani (*naive*): **Seluruh fitur diasumsikan saling bebas bersyarat terhadap label kelas**:

$$P(x_1, x_2, \dots, x_D \mid y) = \prod_{j=1}^D P(x_j \mid y)$$

Keputusan klasifikasi menjadi:

$$\hat{y} = \arg\max_{y} P(y) \prod_{j=1}^D P(x_j \mid y)$$

---

## 15. Gaussian Naive Bayes untuk Fitur Kontinu

Jika fitur masukan berupa besaran riil kontinu (suhu, getaran, tekanan), **Gaussian Naive Bayes** mengasumsikan likelihood setiap fitur mengikuti kurva lonceng distribusi normal:

$$P(x_j \mid y) = \frac{1}{\sqrt{2\pi \sigma_{yj}^2}} \exp\left( -\frac{(x_j - \mu_{yj})^2}{2\sigma_{yj}^2} \right)$$

Model hanya perlu mempelajari dua parameter statistik sederhana per fitur untuk tiap kelas: rata-rata $\mu_{yj}$ dan varians $\sigma_{yj}^2$.

---

## 16. Multinomial Naive Bayes dan Laplace Smoothing

Jika fitur masukan berupa data frekuensi kejadian diskrit (misal: jumlah alarm bahaya atau frekuensi kata teks), model menggunakan distribusi Multinomial:

$$P(x_j \mid y) = \frac{N_{yj} + \alpha}{N_y + \alpha D}$$

### Peran Laplace Smoothing ($\alpha = 1.0$)
Jika suatu kondisi alarm tidak pernah muncul pada data latih kelas tertentu ($N_{yj} = 0$), maka tanpa smoothing nilai $P(x_j \mid y) = 0$. Karena probabilitas dikalikan secara berantai, **satu nilai nol akan menghancurkan seluruh hasil perkalian posterior menjadi nol** (*Zero-Frequency Tragedy*). Parameter $\alpha = 1.0$ menambahkan angka 1 semu ke pembilang, menjamin probabilitas tidak pernah bernilai nol.

---

## 17. Probabilitas Kelas: Prior, Likelihood, Evidence, dan Posterior

```text
           [ HISTORIS ]                              [ PENGAMATAN SENSOR ]
       P(y) = Prior Base Rate                  Likelihood P(x|y) ~ N(mu, sigma^2)
                 \                                        /
                  \                                      /
                   v                                    v
                 +----------------------------------------+
                 |    FORMULA POSTERIOR TEOREMA BAYES     |
                 |      P(y|X) proporsional P(y) * P(X|y) |
                 +----------------------------------------+
                                     |
                                     v
                        Keputusan: argmax_y P(y|X)
```

---

## 18. Perbedaan Mendasar: Prediksi Kelas vs Estimasi Probabilitas

- `model.predict(X)`: Menghasilkan label kategori diskrit langsung ($0$ atau $1$) berdasarkan ambang default $0.5$.
- `model.predict_proba(X)`: Menghasilkan matriks angka probabilitas kontinu $P(y=k \mid X)$.

Dalam rekayasa sistem keselamatan tinggi (*safety-critical systems*), sistem tidak boleh hanya mengandalkan label biner; derajat kepastian probabilitas model harus diinspeksi untuk menentukan apakah alarm butuh intervensi darurat atau cukup dicatat sebagai peringatan dini.

---

## 19. Matriks Konfusi (Confusion Matrix) dan Struktur Kuadran

Matriks konfusi adalah tabel kontingensi $2 \times 2$ yang memetakan kinerja klasifikasi:

```text
                          PREDIKSI MODEL
                     Prediksi 0 (Normal)   Prediksi 1 (Fault)
AKTUAL  Kelas 0 (Normal)    True Negative (TN)    False Positive (FP) [Alarm Palsu]
        Kelas 1 (Fault)     False Negative (FN)   True Positive (TP)  [Deteksi Sukses]
                            [Kerusakan Lolos!]
```

- **True Positive (TP)**: Mesin rusak yang berhasil dideteksi sistem sebagai rusak.
- **False Positive (FP)**: Mesin normal yang salah dituduh rusak (Alarm Palsu / Type I Error).
- **True Negative (TN)**: Mesin normal yang benar diidentifikasi sebagai normal.
- **False Negative (FN)**: Mesin rusak yang gagal dideteksi dan lolos sebagai normal (Bencana Operasional / Type II Error).

---

## 20. Akurasi (Accuracy) dan Paradoks Akurasi

Proporsi total sampel yang diprediksi secara benar:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

### Paradoks Akurasi (Accuracy Paradox)
Jika dari 1.000 mesin terdapat 990 mesin normal dan 10 mesin rusak:
- Pengklasifikasi bodoh yang selalu memprediksi "NORMAL" akan memperoleh akurasi **$99.0\%$**!
- Namun model tersebut memiliki **Recall $= 0.0\%$**, menjadikannya tidak berharga sama sekali. **Dilarang menggunakan akurasi mentah pada dataset yang tidak seimbang**.

---

## 21. Presisi (Precision) dan Pengendalian False Positive

Dari seluruh sampel yang diprediksi positif, berapa persen yang benar-benar positif?

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

Presisi tinggi sangat penting saat biaya memunculkan alarm palsu (*False Positive*) sangat mahal (misal: sistem penangkapan penipuan kartu kredit yang memblokir akun nasabah setia secara keliru).

---

## 22. Recall (Sensitivitas / TPR) dan Pengendalian False Negative

Dari seluruh sampel yang aktual positif di dunia nyata, berapa persen yang berhasil ditangkap oleh model?

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Recall tinggi sangat krusial dalam skenario keselamatan industri dan diagnosis medis, di mana melewatkan satu pasien kanker atau satu kerusakan turbin (*False Negative*) dapat berakibat fatal.

---

## 23. Spesifisitas (Specificity / TNR) dan False Positive Rate (FPR)

- **Specificity (True Negative Rate / TNR)**: Kemampuan model mengidentifikasi sampel negatif secara benar:
  $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
- **False Positive Rate (FPR)**: Proporsi sampel negatif yang salah dituduh positif:
  $$\text{FPR} = 1 - \text{Specificity} = \frac{\text{FP}}{\text{TN} + \text{FP}}$$

---

## 24. F1-Score: Rata-rata Harmonik Presisi dan Recall

Rata-rata harmonik menyeimbangkan presisi dan recall, memberikan penalti berat jika salah satu metrik bernilai sangat rendah:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

Jika model memiliki Precision $1.0$ tetapi Recall $0.0$, rata-rata aritmatika adalah $0.5$, namun rata-rata harmonik (F1) tepat bernilai $0.0$.

---

## 25. Ketidakseimbangan Kelas (Class Imbalance)

Dalam sistem telemetri industri, sebagian besar mesin beroperasi secara normal (prevalensi mayoritas $70\% - 99\%$), sedangkan insiden kegagalan hanya terjadi pada sebagian kecil waktu ($1\% - 30\%$). 

Penanganan metodologis:
1. Menghindari akurasi mentah.
2. Menggunakan `class_weight='balanced'` pada fungsi kerugian.
3. Menyesuaikan ambang probabilitas pada Validation Set.
4. Mengevaluasi performa menggunakan kurva Precision-Recall dan F1-Score.

---

## 26. Metodologi Averaging Multikelas: Macro, Micro, dan Weighted

Ketika mengevaluasi klasifikasi $K > 2$ kelas, metrik per-kelas dirangkum menjadi satu skor tunggal:

1. **Macro Average**: Menghitung skor metrik untuk tiap kelas secara independen lalu meratakannya secara aritmatika sederhana:
   $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
   Memberikan bobot yang **setara** pada kelas minoritas langka (misal Electrical Surge) maupun kelas mayoritas (Normal).
2. **Weighted Average**: Membobot skor setiap kelas berdasarkan jumlah sampel aktual (*support*) kelas tersebut. Sangat dipengaruhi oleh performa kelas mayoritas.
3. **Micro Average**: Menjumlahkan total global $\sum \text{TP}, \sum \text{FP}, \sum \text{FN}$ lintas seluruh kelas sebelum menghitung metrik.

---

## 27. Kurva Receiver Operating Characteristic (ROC)

Kurva ROC menggambarkan performa model biner di **seluruh kemungkinan ambang batas keputusan** ($T \in [0, 1]$):
- Sumbu-Y: **True Positive Rate (Recall)**
- Sumbu-X: **False Positive Rate ($1 - \text{Spesifisitas}$)**

```text
True Positive Rate (TPR) ^
                     1.0 |         .----------------- (Model Unggul)
                         |       .'
                         |     .'
                         |   .'  <-- Garis Diagonal Acak (Random Guess: AUC = 0.5)
                         | .'
                     0.0 +--------------------------> False Positive Rate (FPR)
                         0.0                      1.0
```

Garis diagonal dari $(0,0)$ ke $(1,1)$ merepresentasikan performa tebakan acak murni (koin seimbang). Semakin kurva melengkung mendekati pojok kiri atas $(0, 1)$, semakin sempurna kapabilitas pemisahan model.

---

## 28. Area Under the ROC Curve (ROC-AUC)

ROC-AUC menghitung luas area di bawah kurva ROC ($0 \le \text{AUC} \le 1$):
- $\text{AUC} = 0.5$: Model tidak memiliki daya pembeda (setara lemparan koin acak).
- $\text{AUC} = 1.0$: Model mampu memisahkan kelas positif dan negatif secara sempurna tanpa tumpang tindih.

### Peringatan Kritis Evaluasi ROC-AUC
1. **ROC-AUC Membutuhkan Probabilitas Kontinu**: Menyuplai prediksi biner diskrit (0/1) akan merusak kurva ROC menjadi hanya satu titik patah tunggal.
2. **Interpretasi Probabilistik Wilcoxon-Mann-Whitney**: Nilai ROC-AUC sama persis dengan probabilitas bahwa sampel positif acak akan diberi skor risiko lebih tinggi oleh model dibanding sampel negatif acak:
   $$\text{AUC} = P(\hat{p}_{\text{pos}} > \hat{p}_{\text{neg}})$$

---

## 29. Kurva Precision-Recall (PR) dan Average Precision (AP)

Ketika dataset mengalami ketidakseimbangan kelas ekstrem (misal penipuan $0.1\%$), kurva ROC dapat memberikan ilusi optimisme palsu karena jumlah True Negative (TN) yang masif menekan nilai False Positive Rate (FPR) mendekati nol.

**Kurva Precision-Recall** hanya menyoroti kelas minoritas positif:
- Sumbu-Y: **Precision**
- Sumbu-X: **Recall**
- **Baseline Acak Kurva PR**: Bukan 0.5, melainkan persis sama dengan **prevalensi kelas positif** ($P / N$).
- **Average Precision (AP)**: Luas area di bawah kurva PR, mencerminkan seberapa andal alarm sistem saat berhadapan dengan anomali langka.

---

## 30. Pemilihan Ambang Keputusan Berbasis Biaya Bisnis

Dalam praktiknya, kita tidak boleh membiarkan ambang batas berada pada default $0.5$ tanpa justifikasi operasional:

$$\text{Total Cost} = (\text{FN} \times C_{\text{FN}}) + (\text{FP} \times C_{\text{FP}})$$

Jika biaya kegagalan mesin mendadak ($C_{\text{FN}}$) adalah $\$50.000$ dan biaya inspeksi pencegahan teknisi ($C_{\text{FP}}$) hanya $\$200$:
- Kita wajib **menurunkan ambang batas keputusan ke $T = 0.20$ atau $T = 0.15$**.
- Sistem akan memicu lebih banyak alarm pencegahan (sedikit menaikkan FP), namun mengeliminasi risiko kerusakan katastropik pabrik (memangkas FN mendekati nol).

---

## 31. Tinjauan Kalibrasi Probabilitas (Probability Calibration)

Banyak algoritma menghasilkan skor antara 0 dan 1, namun skor tersebut belum tentu merupakan **probabilitas terkalibrasi** sejati.
- Suatu model dikatakan terkalibrasi sempurna jika: dari seluruh sampel yang diberi estimasi risiko $0.80$, tepat $80\%$ di antaranya benar-benar mengalami kerusakan.
- Naive Bayes cenderung menghasilkan probabilitas yang terlalu percaya diri mendekati 0 atau 1 akibat asumsi independensi. Kalibrasi dapat disempurnakan menggunakan teknik *Platt Scaling* (regresi logistik satu variabel) atau *Isotonic Regression*.

---

## 32. Validasi Silang Terstratifikasi (Stratified Cross-Validation)

Pada tugas klasifikasi, pengacakan acak standar (*standard K-Fold*) berisiko menghasilkan lipatan validasi tanpa satu pun sampel minoritas.

**Stratified K-Fold** membagi sampel sedemikian rupa sehingga **rasio persentase kelas pada setiap lipatan latih dan validasi identik dengan rasio populasi asli**. Ini adalah standar baku industri untuk menjamin estimasi metrik yang tidak terdistorsi.

---

## 33. Analisis Galat Klasifikasi (Error Analysis)

Sebelum merilis model ke lingkungan server, insinyur AI wajib melakukan audit manual terhadap sampel yang salah diklasifikasikan:
1. **Analisis Sampel False Positive**: Mengapa mesin normal dilaporkan rusak? Apakah terdapat gangguan kabel sensor atau lonjakan getaran sesaat akibat forklift melintas?
2. **Analisis Sampel False Negative**: Mengapa kerusakan bantalan lolos? Apakah pola kerusakan berada pada frekuensi akustik di luar jangkauan sensitivitas alat ukur?

---

## 34. Etika dan Klasifikasi yang Bertanggung Jawab

Dalam implementasi sistem klasifikasi yang berdampak pada keselamatan publik atau karier manusia (misal: sistem seleksi karyawan, persetujuan pinjaman, atau diagnosis medis):
- Menghindari diskriminasi kelompok minoritas (menegakkan *Equalized Odds* dan *Demographic Parity* dari Bab 09).
- Menjamin hak atas penjelasan (*Right to Explanation*): operator manusia berhak mengetahui fitur apa yang mendasari keputusan sistem.
- Model harus dirancang sebagai asisten pengambil keputusan (*Human-in-the-Loop*), bukan pengganti akuntabilitas moral manusia.

---

## 35. Ringkasan Bab

- Klasifikasi memetakan fitur masukan kontinu atau diskrit ke dalam label kategori target.
- Regresi Logistik menggunakan fungsi Sigmoid dan penalti Cross-Entropy untuk menghasilkan estimasi probabilitas yang elegan.
- Naive Bayes menerapkan Teorema Bayes dengan asumsi independensi bersyarat; Gaussian NB untuk data kontinu dan Multinomial NB dengan Laplace Smoothing untuk data diskrit.
- Evaluasi klasifikasi bertumpu pada Matriks Konfusi (TP, FP, TN, FN). Metrik Precision, Recall, dan F1-Score memberikan gambaran objektif pada data tidak seimbang.
- Ambang batas keputusan (*decision threshold*) wajib dioptimasi berdasarkan trade-off kerugian bisnis pada Validation Set.
- Kurva ROC mengevaluasi separabilitas global model, sedangkan kurva PR menyoroti daya tangkap pada kelas anomali minoritas.

---

## 36. Latihan Pemahaman Mandiri

### Kategori A: Pemahaman Konseptual (Tingkat Dasar)
1. Diberikan model regresi logistik biner yang menghasilkan linear logit $z = 0.0$. Berapakah nilai probabilitas keluaran Sigmoid $\sigma(z)$ dan apa keputusan klasifikasinya pada ambang default $0.5$?
2. Dalam konteks diagnosis alarm sensor pabrik kimia, manakah yang lebih berbahaya antara False Positive dan False Negative? Jelaskan alasan operasionalnya!
3. Mengapa akurasi sebesar $95\%$ dapat menjadi metrik yang sepenuhnya menyesatkan pada data deteksi transaksi penipuan?

### Kategori B: Aplikasi Metodologis (Tingkat Menengah)
4. Suatu sistem pengenal kerusakan mesin dievaluasi pada 100 sampel uji dan menghasilkan matriks konfusi berikut: $\text{TP} = 20, \text{FP} = 5, \text{FN} = 10, \text{TN} = 65$. Hitung nilai Akurasi, Presisi, Recall, Spesifisitas, dan F1-Score dari sistem tersebut!
5. Jelaskan mekanisme kerja Laplace Smoothing ($\alpha = 1.0$) pada Multinomial Naive Bayes dan sebutkan bahaya matematis apa yang dicegahnya!

### Kategori C: Analisis Tingkat Lanjut (Tingkat Lanjut)
6. Sebuah tim engineering membandingkan model deteksi anomali pada dataset dengan prevalensi positif hanya $0.5\%$. Mereka melaporkan skor ROC-AUC sebesar $0.94$, namun saat diuji di lapangan, sistem membanjiri teknisi dengan ratusan alarm palsu setiap jam. Jelaskan mengapa skor ROC-AUC gagal mencerminkan tingginya alarm palsu tersebut dan metrik apa yang seharusnya mereka pantau!

---

## 37. Mini Project: Classification-Based Equipment Fault Identification

### Deskripsi Tugas
Bangunlah arsitektur pemodelan klasifikasi terintegrasi (*End-to-End Classification Pipeline*) untuk mendeteksi dan mendiagnosis kerusakan peralatan industri berbasis dataset telemetri `equipment_fault_records.csv`.

### Persyaratan Teknis
1. **Analisis Prevalensi & Partisi Data**: Lakukan 3-way split terstratifikasi (Train 60%, Val 20%, Test 20%) dan laporkan proporsi ketidakseimbangan kelas target biner dan multikelas.
2. **Baseline Model**: Bangun `DummyClassifier(strategy='most_frequent')` sebagai patokan kinerja minimum.
3. **Implementasi Multi-Model**: Bangun minimal dua model klasifikasi yang berbeda (misal: Logistic Regression ter-regularisasi vs Gaussian Naive Bayes).
4. **Enkapsulasi Pipeline**: Seluruh proses penskalaan fitur (`StandardScaler`) dan estimasi wajib terbungkus di dalam `sklearn.pipeline.Pipeline`.
5. **Optimasi Ambang Keputusan**: Lakukan *Threshold Tuning* pada Validation Set untuk memaksimalkan F1-Score atau meminimalkan kerugian biaya operasional.
6. **Laporan Evaluasi Menyeluruh**: Sajikan matriks konfusi lengkap, kurva ROC-AUC, kurva Precision-Recall, dan analisis galat kualitatif pada Test Set final.

### Rubrik Penilaian Mini Project (100 Poin)

| Komponen Penilaian | Bobot | Kriteria Evaluasi Capaian |
|---|---|---|
| **1. Problem & Data Understanding** | 15 Poin | Mampu mengidentifikasi struktur data sensor, menganalisis ketidakseimbangan prevalensi kelas, dan merumuskan konsekuensi risiko operasional. |
| **2. Model Implementation** | 25 Poin | Mengimplementasikan minimal dua algoritma klasifikasi berbasis Scikit-learn Pipeline; pra-pemrosesan terisolasi bebas kebocoran. |
| **3. Metrics & Validation Integrity** | 25 Poin | Menggunakan Stratified CV; menyajikan metrik Precision, Recall, F1, ROC-AUC, dan PR-AUC secara tepat dan objektif. |
| **4. Error & Risk Analysis** | 20 Poin | Melakukan audit mendalam terhadap sampel False Positive dan False Negative; mengoptimasi decision threshold berdasarkan analisis biaya bisnis. |
| **5. Documentation & References** | 15 Poin | Menyajikan kode yang dapat direproduksi penuh (*fully reproducible*), tabel komparasi terstruktur, dan mencantumkan referensi akademik primer. |

---

## 38. Referensi Utama

1. **Géron, A. (2022).** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd Edition. O'Reilly Media. [https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
2. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd Edition. Springer. [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/)
3. **Russell, S., & Norvig, P. (2020).** *Artificial Intelligence: A Modern Approach*, 4th Edition. Pearson. [https://aima.cs.berkeley.edu/](https://aima.cs.berkeley.edu/)
4. **Murphy, K. P. (2022).** *Probabilistic Machine Learning: An Introduction*. MIT Press. [https://probml.github.io/pml-book/book1.html](https://probml.github.io/pml-book/book1.html)
5. **ACM / IEEE-CS / AAAI. (2024).** *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. ACM Digital Library. [https://csed.acm.org/](https://csed.acm.org/)

