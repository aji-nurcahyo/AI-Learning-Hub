# Chapter 12: ML Workflow, Dataset & Feature Engineering

Selamat datang di **Chapter 12: ML Workflow, Dataset & Feature Engineering**. Bab ini merupakan fondasi metodologis utama dalam **Level 2: Machine Learning**. Dalam rekayasa kecerdasan buatan modern, keberhasilan suatu sistem cerdas lebih banyak ditentukan oleh kualitas rancangan alur kerja data (*data workflow*), ketatnya pemisahan partisi (*leak-free splitting*), serta kecerdikan dalam merekayasa fitur representatif (*feature engineering*), dibandingkan sekadar kompleksitas algoritma pemodelan yang digunakan.

---

## Daftar Isi
1. [Definisi dan Urgensi ML Workflow End-to-End](#1-definisi-dan-urgensi-ml-workflow-end-to-end)
2. [Tahapan Standar Machine Learning Workflow](#2-tahapan-standar-machine-learning-workflow)
3. [CRISP-DM Framework untuk Proyek AI/ML](#3-crisp-dm-framework-untuk-proyek-aiml)
4. [Perumusan Masalah Bisnis ke Masalah Machine Learning](#4-perumusan-masalah-bisnis-ke-masalah-machine-learning)
5. [Data Ingestion dan Strategi Pengumpulan Data](#5-data-ingestion-dan-strategi-pengumpulan-data)
6. [Eksplorasi Data Awal (EDA) untuk Kesiapan Pemodelan](#6-eksplorasi-data-awal-eda-untuk-kesiapan-pemodelan)
7. [Definisi dan Filosofi Pembagian Dataset](#7-definisi-dan-filosofi-pembagian-dataset)
8. [Peran Krusial Training Set, Validation Set, dan Test Set](#8-peran-krusial-training-set-validation-set-dan-test-set)
9. [Proporsi Pembagian Dataset Standar Industri](#9-proporsi-pembagian-dataset-standar-industri)
10. [Random Splitting: Konsep, Mekanisme, dan Batasan](#10-random-splitting-konsep-mekanisme-dan-batasan)
11. [Stratified Splitting: Menangani Ketidakseimbangan Distribusi](#11-stratified-splitting-menangani-ketidakseimbangan-distribusi)
12. [Group-Based Splitting: Mengisolasi Entitas Terikat](#12-group-based-splitting-mengisolasi-entitas-terikat)
13. [Time-Series Splitting: Eliminasi Look-Ahead Bias](#13-time-series-splitting-eliminasi-look-ahead-bias)
14. [Definisi dan Bahaya Fatal Data Leakage](#14-definisi-dan-bahaya-fatal-data-leakage)
15. [Taksonomi dan Tipe-Tipe Data Leakage](#15-taksonomi-dan-tipe-tipe-data-leakage)
16. [Aturan Emas Pencegahan Leakage: Fit Strictly pada Training Set](#16-aturan-emas-pencegahan-leakage-fit-strictly-pada-training-set)
17. [Pra-pemrosesan Data Numerik: Imputasi Nilai Hilang](#17-pra-pemrosesan-data-numerik-imputasi-nilai-hilang)
18. [Penskalaan Fitur Numerik: Standard, Min-Max, dan Robust Scaler](#18-penskalaan-fitur-numerik-standard-min-max-dan-robust-scaler)
19. [Transformasi Distribusi Numerik: Logaritmik, Box-Cox, dan Yeo-Johnson](#19-transformasi-distribusi-numerik-logaritmik-box-cox-dan-yeo-johnson)
20. [Penanganan Outlier: Winsorizing, Truncation, dan Masking](#20-penanganan-outlier-winsorizing-truncation-dan-masking)
21. [Pra-pemrosesan Data Kategorikal: Nominal vs Ordinal](#21-pra-pemrosesan-data-kategorikal-nominal-vs-ordinal)
22. [One-Hot Encoding, Dummy Trap, dan Penanganan Unseen Categories](#22-one-hot-encoding-dummy-trap-dan-penanganan-unseen-categories)
23. [Ordinal Encoding dan Pemetaan Hierarkis Terstruktur](#23-ordinal-encoding-dan-pemetaan-hierarkis-terstruktur)
24. [Target Encoding (Mean Encoding) dan Mitigasi Overfitting](#24-target-encoding-mean-encoding-dan-mitigasi-overfitting)
25. [Filosofi Rekayasa Fitur (Feature Engineering)](#25-filosofi-rekayasa-fitur-feature-engineering)
26. [Pembuatan Fitur Domain-Spesifik Berbasis Fisika dan Operasional](#26-pembuatan-fitur-domain-spesifik-berbasis-fisika-dan-operasional)
27. [Rekayasa Fitur Temporal dan Siklus Kalender](#27-rekayasa-fitur-temporal-dan-siklus-kalender)
28. [Suku Interaksi (Interaction Terms) dan Non-Linearitas](#28-suku-interaksi-interaction-terms-dan-non-linearitas)
29. [Jebakan Akurasi pada Ketidakseimbangan Kelas (Class Imbalance)](#29-jebakan-akurasi-pada-ketidakseimbangan-kelas-class-imbalance)
30. [Cost-Sensitive Learning dan Penyesuaian Bobot Kelas](#30-cost-sensitive-learning-dan-penyesuaian-bobot-kelas)
31. [Threshold Tuning dan Optimasi Ambang Keputusan Bisnis](#31-threshold-tuning-dan-optimasi-ambang-keputusan-bisnis)
32. [Validasi Silang (Cross-Validation) dan Analisis Varians](#32-validasi-silang-cross-validation-dan-analisis-varians)
33. [Scikit-learn Pipeline: Enkapsulasi Alur Kerja Bebas Kebocoran](#33-scikit-learn-pipeline-enkapsulasi-alur-kerja-bebas-kebocoran)
34. [Scikit-learn ColumnTransformer: Orkestrasi Data Heterogen](#34-scikit-learn-columntransformer-orkestrasi-data-heterogen)
35. [Dokumentasi Eksperimen dan Penetapan Baseline Produksi](#35-dokumentasi-eksperimen-dan-penetapan-baseline-produksi)
36. [Latihan Pemahaman Mandiri](#36-latihan-pemahaman-mandiri)
37. [Mini Project: Reproducible Machine Learning Data Pipeline](#37-mini-project-reproducible-machine-learning-data-pipeline)
38. [Referensi Utama](#38-referensi-utama)

---

## 1. Definisi dan Urgensi ML Workflow End-to-End

Machine Learning bukanlah sekadar memanggil fungsi `.fit()` dan `.predict()` dari pustaka pihak ketiga. Dalam konteks sistem perangkat lunak nyata (*real-world software systems*), pemodelan matematika hanyalah satu komponen kecil di tengah rantai rekayasa sistem yang kompleks (sering dirujuk sebagai fenomena *Hidden Technical Debt in Machine Learning Systems* oleh Sculley et al., Google).

```text
+-------------------------------------------------------------------------------------------------+
|                                 ARSITEKTUR SISTEM ML INDUSTRI                                   |
|                                                                                                 |
|  [Pengumpulan Data] --> [Verifikasi Data] --> [Rekayasa Fitur & Preprocessing]                  |
|                                                              |                                  |
|                                                              v                                  |
|  [Monitoring & Drift] <-- [Serving & Serving Infra] <-- [Model Training & Validation]           |
+-------------------------------------------------------------------------------------------------+
```

Alur kerja Machine Learning (*ML Workflow*) adalah urutan proses sistematis, terstruktur, dan terulang (*reproducible*) yang menjembatani kebutuhan operasional bisnis dengan kapabilitas komputasi prediktif berbasis data. Tanpa alur kerja yang disiplin, proyek kecerdasan buatan rentan terjebak dalam:
1. **Kegagalan Replikasi (*Reproducibility Crisis*)**: Eksperimen yang berhasil di komputer pengembang tidak dapat direplikasi saat diintegrasikan ke lingkungan server.
2. **Pembusukan Model (*Model Degradation*)**: Performa model menurun drastis segera setelah diluncurkan akibat perbedaan distribusi statistik data pelatihan dan data aktual di lapangan.
3. **Kebocoran Informasi (*Data Leakage*)**: Metrik evaluasi tampak sempurna saat riset, namun model menghasilkan prediksi acak saat berhadapan dengan data pengguna sesungguhnya.

---

## 2. Tahapan Standar Machine Learning Workflow

Alur kerja standar Machine Learning modern terdiri atas delapan tahapan berurutan:

```text
[1. Problem Formulation] 
       |
       v
[2. Data Collection & Ingestion] 
       |
       v
[3. Exploratory Data Analysis (EDA)] 
       |
       v
[4. Dataset Splitting & Isolation] 
       |
       v
[5. Preprocessing & Feature Engineering] 
       |
       v
[6. Model Training & Cross-Validation] 
       |
       v
[7. Model Evaluation & Error Analysis] 
       |
       v
[8. Packaging, Deployment & Monitoring]
```

1. **Problem Formulation**: Mendefinisikan sasaran terukur, metrik keberhasilan bisnis, dan memetakan masalah ke tipe tugas ML yang tepat.
2. **Data Collection & Ingestion**: Mengumpulkan data mentah dari berbagai sensor, basis data, atau log transaksi secara terjadwal.
3. **Exploratory Data Analysis (EDA)**: Menyelidiki struktur data, tipe variabel, sebaran distribusi, nilai hilang, korelasi, dan anomali.
4. **Dataset Splitting**: Memisahkan partisi data (*Train*, *Validation*, *Test*) sebelum transformasi matematika apa pun dilakukan.
5. **Preprocessing & Feature Engineering**: Membersihkan data, mengisi nilai kosong, melakukan standardisasi, mengodekan kategori, dan menciptakan fitur prediktif baru.
6. **Model Training & Cross-Validation**: Melatih estimator pada partisi latih dan mengevaluasi kestabilan performa antar lipatan (*folds*).
7. **Model Evaluation & Error Analysis**: Mengukur generalisasi pada *unseen test data* dan mengaudit pola kesalahan klasifikasi/prediksi.
8. **Packaging, Deployment & Monitoring**: Membungkus alur transformasi ke dalam artefak *pipeline* siap guna dan memantau pergeseran data (*data drift*).

---

## 3. CRISP-DM Framework untuk Proyek AI/ML

**CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) adalah standar metodologi industri yang dikembangkan untuk memandu siklus hidup proyek analitik data dan pembelajaran mesin. Karakteristik utama CRISP-DM adalah sifatnya yang iteratif dan non-linear:

```text
                 +-----------------------+
                 |  Business Understanding| <----+
                 +-----------------------+      |
                             |                  |
                             v                  |
                 +-----------------------+      |
                 |   Data Understanding  | <----+
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |    Data Preparation   | <----+
                 +-----------------------+      |
                             |                  |
                             v                  |
                 +-----------------------+      |
                 |       Modeling        | -----+
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |      Evaluation       |
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |       Deployment      |
                 +-----------------------+
```

Pergeseran bolak-balik antara *Data Preparation* dan *Modeling* merupakan keniscayaan: temuan dari *error analysis* pada tahap pemodelan sering kali menuntut rekayasa fitur baru atau perbaikan imputasi pada tahap penyiapan data.

---

## 4. Perumusan Masalah Bisnis ke Masalah Machine Learning

Kunci keberhasilan implementasi AI berakar pada penerjemahan bahasa operasional menjadi perumusan matematis.

| Masalah Bisnis / Operasional | Rumusan Machine Learning | Target Variabel ($y$) | Fitur Input ($X$) |
|---|---|---|---|
| Mencegah kerusakan mendadak pada turbin pembangkit listrik. | Klasifikasi Biner Terbobot (*Cost-Sensitive Binary Classification*). | $y \in \{0, 1\}$ (0: Normal, 1: Kerusakan dalam 24 jam). | Getaran RMS, suhu bantalan, tekanan fluida, riwayat jam operasi. |
| Memperkirakan sisa umur pakai komponen mekanik (*Remaining Useful Life*). | Regresi Terarah (*Time-to-Event Regression*). | $y \in \mathbb{R}^+$ (Jumlah jam operasional tersisa). | Tren laju kenaikan suhu, variansi akustik, siklus beban kerja. |
| Mengidentifikasi perilaku transaksi mencurigakan pada perbankan. | Deteksi Anomali / Klasifikasi Imbalanced Ekstrem. | $y \in \{0, 1\}$ (0: Legal, 1: Penipuan/Fraud). | Deviasi nilai transfer dari rata-rata historis, selisih lokasi IP, waktu transaksi. |

Kegagalan perumusan sering terjadi saat pengembang mengoptimalkan metrik yang tidak selaras dengan kerugian bisnis. Misalnya, memaksimalkan akurasi mentah pada kasus deteksi penipuan justru menghasilkan model yang mengabaikan seluruh transaksi penipuan.

---

## 5. Data Ingestion dan Strategi Pengumpulan Data

Data mentah bersumber dari berbagai kanal dengan frekuensi dan format yang beraneka ragam:
- **Batch Ingestion**: Pemuatan berkala (harian, mingguan) dari *data warehouse* (misal: Apache Spark, Parquet, SQL tables).
- **Streaming Ingestion**: Aliran data berkecepatan tinggi per milidetik dari sensor IoT menggunakan antrean pesan (misal: Apache Kafka, MQTT).

Strategi pengumpulan data wajib menjamin *data provenance* (asal-usul data), skema data yang tervalidasi (*schema enforcement*), serta pencatatan metadata waktu pengambilan (*snapshot timestamp*).

---

## 6. Eksplorasi Data Awal (EDA) untuk Kesiapan Pemodelan

EDA sebelum pemodelan berfokus pada empat pemeriksaan wajib:
1. **Kelengkapan Fitur (*Completeness Check*)**: Memetakan persentase missing values pada tiap kolom. Kolom dengan kehilangan $>50\%$ umumnya membutuhkan evaluasi kritis sebelum dipertahankan.
2. **Kardinalitas Kategorikal (*Cardinality Assessment*)**: Menghitung jumlah kategori unik. Variabel kategori dengan ribuan kategori unik (misal: ID Pengguna) memerlukan strategi khusus (target encoding atau embedding).
3. **Pencilan dan Skewness Distribusi**: Memeriksa kemencengan data (*skewness*) melalui koefisien momen ketiga. Distribusi dengan ekor panjang (*heavy tail*) memerlukan transformasi logaritmik.
4. **Pemeriksaan Multikolinearitas**: Memeriksa korelasi tinggi antar prediktor menggunakan *Variance Inflation Factor* ($VIF$) atau matriks korelasi Spearman/Pearson.

---

## 7. Definisi dan Filosofi Pembagian Dataset

Pemisahan dataset adalah prosedur membagi kumpulan data historis menjadi himpunan bagian yang saling lepas (*mutually exclusive subsets*). 

Filosofi mendasarnya adalah **generalisasi**: model matematika tidak boleh dievaluasi pada data yang telah digunakannya untuk belajar. Jika model diuji pada data yang dihafalnya, performa yang terukur bukanlah kecerdasan memprediksi pola baru, melainkan kapasitas memori penyimpanan semata.

---

## 8. Peran Krusial Training Set, Validation Set, dan Test Set

Sistem pembelajaran mesin profesional membagi data menjadi tiga partisi:

```text
[============================ KESELURUHAN DATASET ============================]
[             TRAIN SET (60%)            ] [   VAL SET (20%)   ] [ TEST SET (20%) ]
              |                                    |                    |
   Mempelajari Parameter Internal       Tuning Hyperparameter   Audit Evaluasi Final
   (Bobot w, Bias b, Split Tree)        & Model Selection       (Unseen Data Murni)
```

1. **Training Set ($60\% - 80\%$)**: Digunakan secara eksklusif oleh algoritma pengoptimalan (misal: Gradient Descent) untuk menyesuaikan parameter internal model (seperti koefisien regresi $w$ dan bias $b$).
2. **Validation Set ($10\% - 20\%$)**: Digunakan oleh data scientist untuk menentukan arsitektur model, memilih hyperparameter terbaik (seperti derajat polinomial, koefisien regularisasi $\lambda$, atau kedalaman pohon), dan mendeteksi titik awal *overfitting*.
3. **Test Set ($10\% - 20\%$)**: Berfungsi sebagai representasi murni dari dunia nyata (*unseen production environment*). Partisi ini **hanya boleh disentuh satu kali** pada akhir siklus pengembangan untuk memverifikasi performa akhir yang objektif.

---

## 9. Proporsi Pembagian Dataset Standar Industri

Proporsi pembagian bergantung pada skala total sampel data ($N$):

| Ukuran Dataset ($N$) | Proporsi Train / Val / Test | Justifikasi Metodologis |
|---|---|---|
| Kecil ($N < 1.000$) | 70% / 15% / 15% atau Cross-Validation | Setiap sampel sangat berharga; CV lebih disukai untuk mengurangi variansi evaluasi. |
| Menengah ($1.000 \le N \le 100.000$) | 60% / 20% / 20% | Standar baku; memberikan ukuran sampel yang memadai untuk validasi dan pengujian statistik. |
| Sangat Besar ($N > 1.000.000$) | 98% / 1% / 1% | $1\%$ dari satu juta data adalah 10.000 sampel, sudah lebih dari cukup untuk menguji signifikansi statistik performa model. |

---

## 10. Random Splitting: Konsep, Mekanisme, dan Batasan

**Random Splitting** membagi data secara acak seragam menggunakan generator bilangan acak semu (*pseudo-random number generator*) dengan parameter `random_state`.

```text
Indeks Asli: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Pengacakan : [3, 7, 0, 9, 1, 4] -> Train (60%)
             [8, 2]             -> Validation (20%)
             [5, 6]             -> Test (20%)
```

**Batasan Kritis**:
Random splitting berasumsi bahwa seluruh data bersifat *Independent and Identically Distributed* ($i.i.d.$). Jika data memiliki struktur relasional, deret waktu, atau ketidakseimbangan kelas ekstrem, random splitting akan menghasilkan estimasi performa yang bias dan menyesatkan.

---

## 11. Stratified Splitting: Menangani Ketidakseimbangan Distribusi

Ketika variabel target mengalami ketidakseimbangan kelas (misal: hanya $5\%$ transaksi yang merupakan penipuan), pembagian acak biasa berisiko membagi data sedemikian rupa sehingga partisi uji mendapatkan $0\%$ sampel positif.

```text
POPULASI: [++++++++++ --------------------------------------------------] (10 Positif, 50 Negatif = 16.7%)

RANDOM SPLIT (BIAS):
Train : [++++++++ ------------------------------] (21.0% Positif)
Test  : [++ --------------------]                ( 9.1% Positif -> Distorsi Distribusi!)

STRATIFIED SPLIT (PROPORSIONAL):
Train : [+++++++ -----------------------------------] (16.7% Positif)
Test  : [+++ -----------------]                       (16.7% Positif -> Terjaga Sempurna!)
```

`StratifiedKFold` dan `train_test_split(..., stratify=y)` menjamin bahwa setiap partisi mempertahankan proporsi kelas target yang identik dengan populasi aslinya.

---

## 12. Group-Based Splitting: Mengisolasi Entitas Terikat

Dalam skenario industri, beberapa baris data sering kali berasal dari entitas fisik atau subjek yang sama (misal: 100 baris rekaman sensor dari Mesin Alpha, 100 baris dari Mesin Beta).

```text
Bahaya Naive Split:
Mesin Alpha: 80 baris di Train, 20 baris di Test
Mesin Beta : 80 baris di Train, 20 baris di Test
-> MODEL MENGHAFAL CIRI MESIN ALPHA, BUKAN POLA KERUSAKAN UMUM!

Solusi Group-Aware Split (GroupShuffleSplit / GroupKFold):
Train Set: Mesin Alpha, Mesin Gamma, Mesin Delta
Test Set : Mesin Beta (Entitas Baru yang Belum Pernah Dilihat Sama Sekali)
```

Pemisahan berbasis kelompok (*Group-Based Splitting*) menjamin bahwa seluruh data dari satu entitas pengelompokan (*group*) dialokasikan secara eksklusif ke satu partisi saja, mencegah *pseudoreplication* dan kebocoran identitas.

---

## 13. Time-Series Splitting: Eliminasi Look-Ahead Bias

Data temporal (telemetri berkala, harga saham, log cuaca) memiliki keterikatan waktu yang ketat. Mengacak urutan data waktu berarti membiarkan model menggunakan data masa depan ($t+1$) untuk memprediksi data masa lalu ($t$), sebuah cacat fatal yang dikenal sebagai **Look-Ahead Bias**.

```text
Time-Series Split (Expanding Window):
Fold 1: [== Train ==] [ Test ]
Fold 2: [==== Train ====] [ Test ]
Fold 3: [====== Train ======] [ Test ]
Fold 4: [======== Train ========] [ Test ]
Waktu : t0 -----------------------------------------------------> tn
```

Metode `TimeSeriesSplit` memastikan bahwa evaluasi selalu bergerak maju dalam waktu (*forward walk*), mencerminkan skenario penerapan operasional nyata di mana model hanya mengetahui data historis hingga saat ini.

---

## 14. Definisi dan Bahaya Fatal Data Leakage

**Data Leakage** (kebocoran data) adalah kondisi patologis di mana informasi dari luar data latih (*out-of-sample information*) secara tidak sengaja terserap ke dalam proses pelatihan model. 

Dampaknya sangat berbahaya: metrik pelatihan dan validasi tampak sangat luar biasa tinggi (misal akurasi $99\%$), menciptakan optimisme palsu di mata pemangku kepentingan. Namun ketika model dideploy ke sistem produksi, performanya anjlok drastis dan sering kali menghasilkan keputusan yang salah dan merugikan secara finansial.

---

## 15. Taksonomi dan Tipe-Tipe Data Leakage

Terdapat empat tipe kebocoran data yang paling umum terjadi dalam rekayasa ML:

```text
+-------------------------------------------------------------------------------------------------+
|                                   TAKSONOMI DATA LEAKAGE                                        |
+-------------------------------------------------------------------------------------------------+
| 1. Pre-Split Preprocessing Leakage: Menghitung mean, median, min, max pada seluruh dataset      |
|    sebelum membaginya menjadi Train dan Test.                                                   |
| 2. Target Leakage: Mengikutsertakan fitur yang hanya tercipta SETELAH peristiwa target terjadi  |
|    (misal: fitur 'nomor tiket komplain perbaikan' untuk memprediksi 'kerusakan mesin').        |
| 3. Feature Engineering Leakage: Menghitung agregasi global atau frekuensi target lintas subset   |
|    tanpa isolasi fold.                                                                          |
| 4. Temporal / Look-Ahead Leakage: Menggunakan data masa depan untuk merekonstruksi data masa    |
|    lampau akibat pengacakan urutan waktu (random shuffle pada time series).                     |
+-------------------------------------------------------------------------------------------------+
```

---

## 16. Aturan Emas Pencegahan Leakage: Fit Strictly pada Training Set

Prinsip paling fundamental yang harus dipegang teguh oleh setiap praktisi AI adalah:

$$\text{Parameter Pra-pemrosesan } (\mu, \sigma, \text{median}, \text{kategori}) = f(\mathcal{D}_{\text{train}})$$

```text
+------------------------------------------------------------------------------------+
|                             ATURAN EMAS PRA-PEMROSESAN                             |
|                                                                                    |
|  1. transformer.fit(X_train)          <-- HITUNG STATISTIK HANYA DARI TRAIN        |
|  2. X_train_proc = transformer.transform(X_train)                                  |
|  3. X_val_proc   = transformer.transform(X_val)    <-- GUNAKAN STATISTIK TRAIN    |
|  4. X_test_proc  = transformer.transform(X_test)   <-- GUNAKAN STATISTIK TRAIN    |
|                                                                                    |
|  JANGAN PERNAH MEMANGGIL .fit() ATAU .fit_transform() PADA VAL ATAU TEST SET!      |
+------------------------------------------------------------------------------------+
```

---

## 17. Pra-pemrosesan Data Numerik: Imputasi Nilai Hilang

Nilai yang hilang (*missing values*) dapat muncul akibat kegagalan sensor, koneksi jaringan terputus, atau kesalahan manusia. Tipe kehilangan data diklasifikasikan menjadi:
- *Missing Completely at Random* (MCAR)
- *Missing at Random* (MAR)
- *Missing Not at Random* (MNAR)

### Strategi Imputasi Standar
1. **Mean Imputation**: Mengganti NaN dengan rata-rata. Efektif jika distribusi data mendekati normal (lonceng simetris). Rentan rusak oleh outlier.
2. **Median Imputation**: Mengganti NaN dengan nilai tengah. Sangat direkomendasikan untuk distribusi menceng (*skewed*) dan data industri yang sarat lonjakan sensor.
3. **Constant Imputation**: Mengganti NaN dengan konstanta tertentu (misal: $0$ atau $-999$) yang memiliki makna domain spesifik.
4. **KNN / Iterative Imputer**: Mengestimasi nilai hilang menggunakan relasi multi-variat terhadap fitur-fitur lain di sekitarnya.

---

## 18. Penskalaan Fitur Numerik: Standard, Min-Max, dan Robust Scaler

Sebagian besar algoritma optimasi gradien dan algoritma berbasis jarak (k-NN, SVM, Regresi Linier/Logistik ter-regularisasi, Jaringan Saraf Tiruan) sangat sensitif terhadap rentang skala numerik tiap fitur.

### 1. StandardScaler (Z-Score Standardization)
Mentransformasikan fitur sehingga memiliki rata-rata $\mu = 0$ dan simpangan baku $\sigma = 1$:

$$z = \frac{x - \mu}{\sigma}$$

### 2. MinMaxScaler
Mentransformasikan data ke dalam interval tertutup $[0, 1]$:

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Sangat sensitif jika terdapat pencilan (*outliers*) ekstrem yang menekan data non-outlier ke rentang yang sangat sempit.

### 3. RobustScaler
Menggunakan median dan rentang interkuartil (*Interquartile Range*, $IQR = Q_3 - Q_1$):

$$x_{\text{robust}} = \frac{x - \text{median}}{IQR}$$

Skaler ini kebal terhadap pencilan ekstrem karena kuartil tidak terpengaruh oleh titik data di ujung terjauh distribusi.

---

## 19. Transformasi Distribusi Numerik: Logaritmik, Box-Cox, dan Yeo-Johnson

Banyak model parametrik mengasumsikan galat berdistribusi normal dengan varians konstan (*homoscedasticity*). Jika data sangat menceng ke kanan (*right-skewed*), transformasi non-linear diperlukan:

1. **Log Transformation**: $x' = \log(x + c)$, sangat baik untuk data nilai uang atau waktu pemrosesan di mana $x > 0$.
2. **Box-Cox Transformation**: Transformasi berbasis parameter $\lambda$, menuntut seluruh nilai $x > 0$.
3. **Yeo-Johnson Transformation**: Perluasan dari Box-Cox yang dapat menangani data bernilai nol maupun negatif.

---

## 20. Penanganan Outlier: Winsorizing, Truncation, dan Masking

Outlier tidak selalu harus dihapus. Menghapus baris sering kali membuang informasi berharga dan mengacaukan keselarasan ukuran partisi.
- **Winsorizing (Clipping)**: Membatasi nilai di luar persentil tertentu (misal persentil ke-1 dan ke-99). Nilai di atas persentil 99 diganti nilainya tepat dengan nilai ambang persentil 99.
- **Outlier Flagging / Masking**: Menambahkan kolom biner baru `is_outlier = 1` untuk memberi sinyal khusus kepada model bahwa sampel tersebut berada pada kondisi ekstrem.

---

## 21. Pra-pemrosesan Data Kategorikal: Nominal vs Ordinal

Data kategorikal merepresentasikan atribut kualitatif:
- **Kategorikal Nominal**: Kategori yang tidak memiliki relasi urutan atau hierarki intrinsik. Contoh: `cooling_status` $\in$ {`ACTIVE`, `STANDBY`, `OFF`}. Tidak boleh dianggap `OFF` > `ACTIVE` secara numerik.
- **Kategorikal Ordinal**: Kategori yang memiliki hierarki peringkat atau tingkatan nilai yang jelas. Contoh: `warning_flag` $\in$ {`LOW`, `MEDIUM`, `HIGH`} di mana $\text{LOW} < \text{MEDIUM} < \text{HIGH}$.

---

## 22. One-Hot Encoding, Dummy Trap, dan Penanganan Unseen Categories

Untuk data nominal, **One-Hot Encoding** (OHE) memecah variabel berkategori $K$ menjadi $K$ vektor biner:

```text
cooling_status -> [cooling_ACTIVE, cooling_STANDBY, cooling_OFF]
ACTIVE         -> [       1      ,        0       ,      0     ]
STANDBY        -> [       0      ,        1       ,      0     ]
OFF            -> [       0      ,        0       ,      1     ]
```

### Dummy Variable Trap
Jika seluruh $K$ kolom dipertahankan dalam model regresi linier tanpa regularisasi, penjumlahan kolom-kolom tersebut tepat bernilai 1 (matriks singular akibat multikolinearitas sempurna). Solusinya adalah menjatuhkan satu kolom (`drop='first'`). Pada model dengan regularisasi atau pohon keputusan, mempertahankan seluruh $K$ kolom adalah praktik standar.

### Penanganan Unseen Categories
Dalam sistem produksi, data baru bisa saja memunculkan kategori yang belum pernah ada pada masa pelatihan (misal sensor melaporkan status `EMERGENCY_VENT`). Menetapkan parameter `handle_unknown='ignore'` pada `OneHotEncoder` Scikit-learn memastikan kategori baru diubah menjadi vektor semua-nol, mencegah program mengalami pengecualian (*runtime crash*).

---

## 23. Ordinal Encoding dan Pemetaan Hierarkis Terstruktur

Variabel ordinal harus dipetakan ke bilangan bulat yang mempertahankan relasi urutan:
```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[["LOW", "MEDIUM", "HIGH"]],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)
```
Pemetaan manual eksplisit memastikan bahwa urutan alfabetis tidak merusak logika domain (misal tanpa konfigurasi, `HIGH` bisa salah diurutkan sebelum `LOW`).

---

## 24. Target Encoding (Mean Encoding) dan Mitigasi Overfitting

Untuk fitur kategorikal dengan kardinalitas sangat tinggi (misal 5.000 kode pos atau 10.000 ID suku cadang), One-Hot Encoding akan menciptakan ribuan kolom renggang (*sparse matrix*) yang memicu *Curse of Dimensionality*.

**Target Encoding** mengganti setiap kategori dengan nilai rata-rata target ($y$) untuk kategori tersebut:

$$\hat{x}_i = \frac{n \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n + m}$$

Di mana $m$ adalah parameter penghalusan (*smoothing weight*) untuk mencegah kategori dengan frekuensi kecil mengalami overfitting ekstrem.

---

## 25. Filosofi Rekayasa Fitur (Feature Engineering)

Pepatah klasik dalam ilmu komputer *"Garbage In, Garbage Out"* berlaku mutlak pada Machine Learning. Algoritma paling canggih sekalipun tidak dapat memprediksi target jika data input tidak memiliki korelasi atau sinyal informasi yang relevan.

Rekayasa Fitur (*Feature Engineering*) adalah seni dan sains mentransformasikan representasi ruang fitur mentah menjadi variabel baru yang menonjolkan struktur esensial dari masalah yang ingin diselesaikan, mempermudah model dalam mempelajari batas keputusan (*decision boundaries*).

---

## 26. Pembuatan Fitur Domain-Spesifik Berbasis Fisika dan Operasional

Dalam rekayasa peralatan industri, nilai sensor individu sering kali kurang bermakna dibandingkan kombinasi fisik antarfaktor:
- **Rasio Stres Fluida-Mekanik**: $\text{Vibration} / \text{Pressure}$. Loncatan rasio ini mengindikasikan kavitasi pompa.
- **Gradien Suhu per Jam Operasi**: $\text{Temperature} / (\text{Operating Hours} / 1000)$. Mengukur laju degradasi termal akibat keausan bantalan.
- **Indeks Kerusakan Termomekanikal**: $\text{Vibration} \times \text{Temperature}$. Memperkuat sinyal kegagalan ketika suhu ekstrem dan getaran tinggi terjadi bersamaan.

---

## 27. Rekayasa Fitur Temporal dan Siklus Kalender

Data waktu memiliki sifat siklikal periodik (jam 23 ke jam 00 berdekatan secara fisik, namun berjarak 23 secara angka diskrit).

1. **Dekomposisi Kalender**: `hour`, `dayofweek`, `is_weekend`, `quarter`.
2. **Transformasi Trigonometri Siklis**:

$$\sin_{\text{hour}} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos_{\text{hour}} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$

3. **Fitur Lag dan Rolling Window**:
   - `lag_1`: Nilai sensor 1 jam sebelumnya.
   - `rolling_mean_6h`: Rata-rata getaran selama 6 jam terakhir.
   - `rolling_std_6h`: Volatilitas getaran selama 6 jam terakhir.

---

## 28. Suku Interaksi (Interaction Terms) dan Non-Linearitas

Model linier sederhana mengasumsikan efek tiap fitur bersifat independen aditif: $\hat{y} = w_1 x_1 + w_2 x_2$.
Namun di dunia nyata, pengaruh $x_1$ sering kali bergantung pada nilai $x_2$.

Dengan menambahkan suku interaksi perkalian $x_1 x_2$, model linier dapat menangkap interaksi non-linear tanpa memerlukan arsitektur non-linier yang rumit:

$$\hat{y} = w_1 x_1 + w_2 x_2 + w_{12}(x_1 x_2) + b$$

---

## 29. Jebakan Akurasi pada Ketidakseimbangan Kelas (Class Imbalance)

Jika suatu dataset telemetri memiliki 990 mesin normal dan 10 mesin rusak:
- Model naif yang memprediksi "NORMAL" untuk setiap mesin akan memperoleh akurasi **$99.0\%$**!
- Namun, model tersebut memiliki nilai **Recall Kerusakan $0.0\%$**, menjadikannya sama sekali tidak berguna dalam mencegah kerusakan pabrik.

Pada dataset tidak seimbang, akurasi mentah (*raw accuracy*) adalah jebakan. Metrik yang wajib dipantau adalah:
- **Precision**: Berapa proporsi alarm kegagalan yang benar-benar akurat?
- **Recall (Sensitivity)**: Berapa proporsi total mesin rusak yang berhasil ditangkap sistem?
- **F1-Score**: Rata-rata harmonik antara Precision dan Recall.
- **Balanced Accuracy**: Rata-rata recall pada tiap kelas secara independen.

---

## 30. Cost-Sensitive Learning dan Penyesuaian Bobot Kelas

Daripada melakukan manipulasi data sintetis pra-split yang berisiko memicu kebocoran (*leakage*), pendekatan yang paling aman dan metodologis adalah **Cost-Sensitive Learning**.

Algoritma memberikan penalti kesalahan lebih berat pada kelas minoritas melalui bobot inversi frekuensi:

$$w_k = \frac{N}{K \cdot n_k}$$

Pada Scikit-learn, parameter `class_weight='balanced'` secara otomatis menghitung penalti ini sehingga gradien optimasi tidak didominasi oleh kelas mayoritas.

---

## 31. Threshold Tuning dan Optimasi Ambang Keputusan Bisnis

Secara default, model klasifikasi biner memprediksi kelas 1 jika probabilitas output $P(y=1|x) \ge 0.5$.
Namun, dalam skenario di mana biaya membiarkan kerusakan terjadi jauh lebih mahal daripada biaya inspeksi teknisi:
- Ambang batas (*decision threshold*) dapat diturunkan ke $0.25$ atau $0.30$.
- Penyesuaian ambang ini **wajib dilakukan pada Validation Set**, kemudian dikunci dan diuji pada Test Set.

---

## 32. Validasi Silang (Cross-Validation) dan Analisis Varians

Ketika dataset berukuran terbatas, pembagian tunggal Train-Validation rentan menghasilkan variansi tinggi akibat ketergantungan pada seed acak.

```text
5-Fold Cross-Validation:
Fold 1: [ Test ] [ Train ] [ Train ] [ Train ] [ Train ] -> Score 1
Fold 2: [ Train ] [ Test ] [ Train ] [ Train ] [ Train ] -> Score 2
Fold 3: [ Train ] [ Train ] [ Test ] [ Train ] [ Train ] -> Score 3
Fold 4: [ Train ] [ Train ] [ Train ] [ Test ] [ Train ] -> Score 4
Fold 5: [ Train ] [ Train ] [ Train ] [ Train ] [ Test ] -> Score 5

Estimasi Akhir = Mean(Scores) +/- Std(Scores)
```

Deviasi standar yang kecil mengindikasikan bahwa model memiliki generalisasi yang stabil di seluruh sebaran data.

---

## 33. Scikit-learn Pipeline: Enkapsulasi Alur Kerja Bebas Kebocoran

Kelas `Pipeline` dari `sklearn.pipeline` menyatukan rantai transformer data dan satu estimator akhir ke dalam satu objek terpadu.

```text
Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

**Keunggulan Utama**:
1. Menjamin bahwa saat `pipeline.fit(X_train)` dipanggil, statistik imputasi dan penskalaan *hanya* dipelajari dari `X_train`.
2. Saat `cross_val_score(pipeline, X, y)` dijalankan, Scikit-learn secara otomatis mengisolasi pra-pemrosesan di dalam masing-masing fold latih tanpa kebocoran ke fold validasi.
3. Objek pipeline dapat diekspor (*serialize*) sebagai satu berkas artefak produksi tunggal.

---

## 34. Scikit-learn ColumnTransformer: Orkestrasi Data Heterogen

Dalam dataset tabular nyata, kolom-kolom memiliki tipe data yang berbeda dan membutuhkan perlakuan transformasi yang berlainan. `ColumnTransformer` memungkinkan pemetaan jalur pra-pemrosesan berbeda ke kolom yang bersesuaian secara paralel:

```text
                           +---> [SimpleImputer + StandardScaler] ---> Kolom Numerik
                           |
Data Mentah Tabular (X) ---+---> [OneHotEncoder(handle_unknown)]  ---> Kolom Nominal
                           |
                           +---> [OrdinalEncoder(categories)]     ---> Kolom Ordinal
                                                    |
                                                    v
                                      [Hasil Gabungan Fitur 2D]
                                                    |
                                                    v
                                         [Estimator / Classifier]
```

Seluruh alur ini terikat rapi dalam satu pemanggilan antarmuka `.fit()` dan `.predict()`.

---

## 35. Dokumentasi Eksperimen dan Penetapan Baseline Produksi

Sebelum melangkah ke algoritma rumit (seperti Deep Neural Networks atau Gradient Boosted Trees), data scientist wajib membangun **Baseline Model Terstandarisasi**.

Model dasar (seperti Logistic Regression dengan pra-pemrosesan bersih) memberikan patokan biaya komputasi dan performa minimum. Dokumentasi eksperimen wajib mencakup:
- Versi kode dan snapshot dataset (*data hash*).
- Konfigurasi parameter pra-pemrosesan.
- Metrik validasi silang (Mean $\pm$ Std) dan metrik uji final.
- Analisis latensi inferensi per sampel.

---

## 36. Latihan Pemahaman Mandiri

Uji pemahaman Anda terhadap konsep bab ini dengan menjawab pertanyaan-pertanyaan berikut:

### Kategori A: Pemahaman Konseptual (Tingkat Dasar)
1. Jelaskan perbedaan mendasar antara peran *Validation Set* dan *Test Set*. Mengapa kita tidak boleh melakukan tuning hyperparameter langsung pada *Test Set*?
2. Mengapa fungsi `.fit()` tidak boleh dipanggil pada data uji (*Test Set*)? Apa konsekuensi matematisnya jika aturan ini dilanggar?
3. Sebutkan satu skenario di mana *Random Splitting* biasa akan menyebabkan kegagalan fatal pada estimasi performa model!

### Kategori B: Aplikasi Metodologis (Tingkat Menengah)
4. Diberikan dataset sensor mesin turbin yang dicatat setiap 10 menit selama 6 bulan. Anda diminta memprediksi kegagalan mesin. Jelaskan strategi pembagian dataset yang paling tepat dan sebutkan jenis data leakage apa yang ingin Anda cegah!
5. Jelaskan mekanisme kerja `handle_unknown='ignore'` pada `OneHotEncoder` dan mengapa parameter ini sangat krusial saat model beroperasi di lingkungan produksi nyata!

### Kategori C: Analisis Tingkat Lanjut (Tingkat Lanjut)
6. Sebuah tim data science melaporkan akurasi model sebesar $98\%$ pada data deteksi penipuan kartu kredit menggunakan validasi silang 10-fold. Namun, setelah ditelusuri, mereka melakukan SMOTE (oversampling data minoritas) pada seluruh dataset sebelum memanggil `KFold`. Analisis secara kritis mengapa angka akurasi $98\%$ tersebut merupakan ilusi semata!

---

## 37. Mini Project: Reproducible Machine Learning Data Pipeline

### Deskripsi Tugas
Bangunlah sebuah arsitektur *end-to-end* **Reproducible Machine Learning Data Pipeline** untuk mendeteksi potensi kerusakan peralatan industri berbasis data telemetri `equipment_telemetry.csv`.

### Persyaratan Teknis
1. **Pemisahan Partisi**: Gunakan skema *3-Way Stratified Split* (Train 60%, Val 20%, Test 20%) dengan verifikasi ketiadaan tumpang tindih indeks (*zero-overlap assertion*).
2. **Pra-pemrosesan Heterogen**: Bangun `ColumnTransformer` yang mengarahkan fitur numerik ke pipeline imputasi median + standardisasi, fitur nominal ke `OneHotEncoder`, dan fitur ordinal ke `OrdinalEncoder`.
3. **Rekayasa Fitur Domain**: Ciptakan minimal dua fitur baru berbasis kombinasi domain (misal rasio stres atau indeks termomekanikal).
4. **Mitigasi Imbalance**: Terapkan penanganan ketidakseimbangan kelas menggunakan `class_weight='balanced'` atau *threshold tuning* pada Validation Set.
5. **Enkapsulasi Pipeline**: Seluruh proses transformasi dan klasifikasi wajib terbungkus rapi dalam satu objek `sklearn.pipeline.Pipeline`.
6. **Laporan Evaluasi**: Sajikan Classification Report lengkap (Precision, Recall, F1, Balanced Accuracy) pada Test Set final dan dokumentasikan analisis kestabilannya.

### Rubrik Penilaian Mini Project (100 Poin)

| Komponen Penilaian | Bobot | Kriteria Evaluasi Capaian |
|---|---|---|
| **1. Problem & Dataset Understanding** | 15 Poin | Mampu mengidentifikasi struktur data, tipe variabel, proporsi ketidakseimbangan kelas, dan menguraikan risiko teknis proyek secara komprehensif. |
| **2. Splitting & Leakage Prevention** | 25 Poin | Menerapkan partisi 3-way terisolasi secara sempurna; menjamin ketiadaan kebocoran informasi (*zero-leakage*) dengan pembuktian programmatic. |
| **3. Preprocessing Correctness** | 25 Poin | Imputasi dan penskalaan parameter statistik fit strictly hanya pada data latih; penanganan kategori baru (*unseen*) terbukti tangguh tanpa crash. |
| **4. Pipeline & Validation Integrity** | 20 Poin | Mengintegrasikan seluruh komponen dalam Scikit-learn Pipeline dan ColumnTransformer modular; evaluasi menggunakan Stratified K-Fold. |
| **5. Documentation & Interpretation** | 15 Poin | Menyajikan tabel hasil numerik aktual, grafik korelasi, dan analisis kritis terhadap trade-off performa model secara terstruktur dan profesional. |

---

## 38. Referensi Utama

1. **Géron, A. (2022).** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd Edition. O'Reilly Media. [https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
2. **Kuhn, M., & Johnson, K. (2019).** *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press. [https://bookdown.org/max/FES/](https://bookdown.org/max/FES/)
3. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd Edition. Springer. [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/)
4. **Zheng, A., & Casari, A. (2018).** *Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists*. O'Reilly Media. [https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
5. **ACM / IEEE-CS / AAAI. (2024).** *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. ACM Digital Library. [https://csed.acm.org/](https://csed.acm.org/)

