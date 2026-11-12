# Chapter 19: Model Evaluation, Tuning & Explainability

Selamat datang pada Chapter 19 dari kurikulum resmi kecerdasan buatan AI Learning Hub. Bab ini mendalami disiplin rekayasa sistem yang memisahkan praktisi pemula dari insinyur pembelajaran mesin profesional: **Evaluasi Model, Penyetelan Hyperparameter (*Tuning*), dan Keterpahaman Model (*Explainability*)**. Membangun model yang mencetak skor tinggi pada data latih adalah hal yang mudah, namun memastikan model tersebut memiliki kemampuan generalisasi yang kokoh pada data dunia nyata, terkalibrasi secara probabilitas, dan dapat dipertanggungjawabkan secara etis adalah tantangan rekayasa yang sesungguhnya.

---

## 1. Fondasi Evaluasi Model & Kemampuan Generalisasi (*Generalization*)

### 1.1 Mengapa Evaluasi Model Sangat Krusial?
Tujuan fundamental dari *machine learning* bukanlah menghafal observasi masa lalu, melainkan mengekstrak pola invarian yang mampu melakukan inferensi akurat pada data baru yang belum pernah dilihat sebelumnya (*unseen data*). Fenomena di mana model memiliki galat sangat rendah pada data latih namun gagal total saat menghadapi data uji disebut sebagai **overfitting** (varians tinggi). Sebaliknya, model yang terlalu kaku dan gagal menangkap struktur dasar data disebut mengalami **underfitting** (bias tinggi).

```
+-----------------------------------------------------------------------+
|                    KAPASITAS MODEL & DILEMA GENERALISASI              |
+-----------------------------------------------------------------------+
|  Galat (Error)                                                        |
|     ^                                                                 |
|     |     \                                 /  Galat Validasi / Uji   |
|     |      \       Zona Optimal            /   (Overfitting)          |
|     |       \       Sweet Spot            /                           |
|     |        \     *-------------*       /                            |
|     |         \   /               \     /                             |
|     |          \_/                 \___/                              |
|     |           \                                                     |
|     |            \____________________________ Galat Pelatihan (Train)|
|     +---------------------------------------------> Kompleksitas Model|
|        (Underfitting / High Bias)       (Overfitting / High Variance) |
+-----------------------------------------------------------------------+
```

### 1.2 Tiga Partisi Data Terisolasi (*Three-Way Data Splitting*)
Untuk mengevaluasi model tanpa bias optimisme semu, dataset wajib dipartisi menjadi tiga himpunan yang saling lepas (*mutually exclusive*):

1. **Training Set (60–80%)**: Digunakan secara eksklusif oleh algoritma optimasi untuk mempelajari parameter internal model (seperti bobot $w$ dan bias $b$).
2. **Validation Set (10–20%)**: Digunakan untuk membandingkan berbagai algoritma kandidat, mendiagnosis kurva pembelajaran, dan menyetel hyperparameter.
3. **Test Set Terisolasi (10–20%)**: **Haram disentuh** selama siklus eksperimen pelatihan dan penyetelan hyperparameter. Himpunan ini hanya dievaluasi SATU KALI di akhir siklus pengembangan untuk memberikan estimasi performa generalisasi final yang tidak bias.

```
+-----------------------------------------------------------------------+
|                    ARSITEKTUR PARTISI DATASTANDAR                     |
+-----------------------------------------------------------------------+
|  [========================== SELURUH DATASET ==========================]
|                                    │                                  |
|            ┌───────────────────────┴───────────────────────┐          |
|            ▼                                               ▼          |
|  [====== DATA PENGEMBANGAN (80%) ======]         [=== TEST SET (20%) =|
|            │                                     (Hanya untuk Audit   |
|     ┌──────┴──────┐                               Akhir - ZERO LEAK!) |
|     ▼             ▼                                                   |
| [Train (80%)]  [Validation (20%)]                                     |
| (Belajar w,b)  (Pilih Hyperparameter)                                 |
+-----------------------------------------------------------------------+
```

### 1.3 Model Baseline: Titik Tolak Ilmiah
Sebelum menyimpulkan bahwa model *Random Forest* atau *Gradient Boosting* yang rumit bernilai guna, model tersebut harus dibandingkan dengan **model dasar (*baseline model*)**:
- **Baseline Klasifikasi**: *Dummy Majority Classifier* yang selalu memprediksi kelas terbanyak. Jika pada dataset deteksi penipuan kelas mayoritas berjumlah 98%, maka model cerdas yang hanya mencetak akurasi 98% sebenarnya sama sekali tidak memiliki kemampuan prediktif!
- **Baseline Regresi**: *Mean/Median Regressor* yang selalu memprediksi nilai rata-rata target. Nilai $R^2$ model kompleks harus secara signifikan melampaui $0.0$.

---

## 2. Metrik Evaluasi Regresi

Dalam pemodelan regresi dengan target kontinu $y \in \mathbb{R}$, deviasi antara nilai aktual $y_i$ dan nilai prediksi $\hat{y}_i$ diukur menggunakan berbagai metrik dengan karakteristik penalti yang berbeda:

### 2.1 Mean Absolute Error (MAE)
Rata-rata magnitudo selisih absolut:
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
*Karakteristik*: Bersifat linier terhadap besar galat. Setiap unit deviasi diberi bobot yang sama persis, sehingga sangat kokoh (*robust*) terhadap pencilan ekstrem.

### 2.2 Mean Squared Error (MSE) & Root Mean Squared Error (RMSE)
Rata-rata kuadrat galat dan akar kuadratnya:
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2, \quad \text{RMSE} = \sqrt{\text{MSE}}$$
*Karakteristik*: Penalti kuadratik menghukum galat besar secara eksponensial. Jika terdapat satu prediksi yang meleset jauh (misal deviasi 10 poin vs 1 poin), RMSE akan melonjak tajam dibandingkan MAE. RMSE memiliki satuan dimensi yang sama dengan variabel target asli.

### 2.3 Koefisien Determinasi ($R^2$ Score)
Proporsi varians target yang berhasil dijelaskan oleh model dibandingkan model rata-rata acuan:
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
- $R^2 = 1.0$: Model memprediksi setiap sampel secara sempurna tanpa galat.
- $R^2 = 0.0$: Performa model setara dengan selalu memprediksi nilai rata-rata $\bar{y}$.
- $R^2 < 0.0$: Performa model lebih buruk daripada sekadar menebak rata-rata.

---

## 3. Metrik Evaluasi Klasifikasi & Asimetri Biaya Keputusan

### 3.1 Dekomposisi Matriks Kebingungan (*Confusion Matrix*)
Pada klasifikasi biner dengan label kelas Positif ($1$) dan Negatif ($0$), prediksi model dibagi menjadi empat kuadran:

```
+-----------------------------------------------------------------------+
|                    STRUKTUR CONFUSION MATRIX                          |
+-----------------------------------------------------------------------+
|                         Kondisi Aktual (Kebenaran)                    |
|                        Positif (1)            Negatif (0)             |
| Prediksi Positif (1)   True Positive (TP)     False Positive (FP)     |
|                        (Deteksi Tepat)        (Alarm Palsu / Type I)  |
|                                                                       |
| Prediksi Negatif (0)   False Negative (FN)    True Negative (TN)      |
|                        (Lolos / Type II Kritis) (Penolakan Tepat)     |
+-----------------------------------------------------------------------+
```

### 3.2 Metrik Turunan Standar
- **Akurasi (Accuracy)**: Rasio total prediksi benar:
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
- **Presisi (Precision)**: Proporsi prediksi positif yang benar-benar terbukti positif:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \\text{FP}}$$
- **Sensitivitas / Recall (True Positive Rate)**: Proporsi kasus positif aktual yang berhasil dideteksi:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Spesifisitas (True Negative Rate)**: Proporsi kasus negatif aktual yang berhasil ditolak:
  $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
- **F1-Score**: Rata-rata harmonik antara Presisi dan Recall:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$

### 3.3 Asimetri Biaya & Pemilihan Ambang Batas Keputusan (*Threshold Tuning*)
Secara default, pustaka machine learning menetapkan ambang batas probabilitas keputusan pada $\theta = 0.50$ ($\hat{y} = 1$ jika $P(y=1|\mathbf{x}) \ge 0.50$). Namun pada dunia nyata, biaya kesalahan bersifat asimetris:

```
+-----------------------------------------------------------------------+
|                    ANALISIS ASIMETRI BIAYA SISTEM                     |
+-----------------------------------------------------------------------+
|  Skenario Pemeliharaan Prediktif IoT:                                 |
|  - Biaya False Positive (FP): Inspeksi mekanik preventif = $500       |
|  - Biaya False Negative (FN): Kerusakan fatal mendadak  = $10,000      |
|                                                                       |
|  Strategi Optimal:                                                    |
|  Turunkan threshold ke theta = 0.35! Model menjadi lebih sensitif     |
|  (Recall naik ke 95%), menerima sedikit kenaikan alarm palsu demi     |
|  mengeliminasi kegagalan katastropik bernilai $10,000.                |
+-----------------------------------------------------------------------+
```

### 3.4 ROC-AUC vs Precision-Recall AUC
- **Kurva ROC (*Receiver Operating Characteristic*)**: Memplot True Positive Rate vs False Positive Rate pada seluruh rentang ambang $\theta \in [0, 1]$. Metrik ROC-AUC mengukur probabilitas bahwa model memberi skor lebih tinggi pada sampel positif acak dibandingkan sampel negatif acak.
- **Kurva Precision-Recall (PR Curve)**: Memplot Precision vs Recall. Ketika data sangat tidak seimbang (*highly imbalanced* di mana kelas positif $< 5\%$), kurva ROC dapat memberikan ilusi optimis karena denominator FPR ($\text{TN} + \text{FP}$) didominasi oleh jumlah TN yang masif. Pada kasus tersebut, **PR-AUC wajib digunakan sebagai metrik utama**.

---

## 4. Strategi Validasi Silang (*Cross-Validation*)

Untuk meminimalkan varians estimasi evaluasi akibat pemilihan split acak tunggal, Cross-Validation mempartisi data menjadi $K$ kelompok secara bergantian.

```
+-----------------------------------------------------------------------+
|                    K-FOLD CROSS-VALIDATION (K = 5)                    |
+-----------------------------------------------------------------------+
|  Iterasi 1: [ VALIDASI ] [ Latih ]    [ Latih ]    [ Latih ] [ Latih ]|
|  Iterasi 2: [ Latih ]    [ VALIDASI ] [ Latih ]    [ Latih ] [ Latih ]|
|  Iterasi 3: [ Latih ]    [ Latih ]    [ VALIDASI ] [ Latih ] [ Latih ]|
|  Iterasi 4: [ Latih ]    [ Latih ]    [ Latih ]    [ VALIDASI ] [ Latih ]|
|  Iterasi 5: [ Latih ]    [ Latih ]    [ Latih ]    [ Latih ] [ VALIDASI ]|
|                                                                       |
|  Skor Akhir = Rerata(Skor_1 ... Skor_5) +/- Deviasi Standar          |
+-----------------------------------------------------------------------+
```

### 4.1 Taksonomi Cross-Validation Khusus

#### 1. Stratified K-Fold
Menjaga agar persentase sebaran kelas target di setiap fold persis identik dengan sebaran populasi asli. Wajib digunakan untuk seluruh problem klasifikasi.

#### 2. Group K-Fold (Mencegah Kebocoran Kelompok Entitas)
Jika dataset memiliki relasi hierarki entitas (misalnya observasi berulang dari 10 perangkat sensor `device_id` atau rekam medis pasien). Jika sampel dari perangkat yang sama terpecah di train dan validation, model akan menghafal karakteristik unik perangkat tersebut (*device fingerprint leakage*). Group K-Fold menjamin seluruh observasi dari satu grup berada secara eksklusif di set latih atau set validasi.

#### 3. Time Series Split (Mencegah Kebocoran Temporal)
Pada data deret waktu, pengacakan (*shuffling*) melanggar kausalitas fisik waktu (*temporal leakage*). TimeSeriesSplit menggunakan pendekatan jendela bertumbuh (*rolling origin*): fold latih hanya menggunakan data masa lalu, dan fold validasi menguji periode masa depan berikutnya.

```
+-----------------------------------------------------------------------+
|                    TIME SERIES SPLIT (ROLLING ORIGIN)                 |
+-----------------------------------------------------------------------+
|  Split 1: [ Train: t1-t10 ] [ Val: t11-t15 ]                          |
|  Split 2: [ Train: t1-t15          ] [ Val: t16-t20 ]                 |
|  Split 3: [ Train: t1-t20                   ] [ Val: t21-t25 ]        |
|  * Tidak pernah melatih dengan masa depan untuk memprediksi masa lalu!|
+-----------------------------------------------------------------------+
```

### 4.2 Nested Cross-Validation (Eliminasi Bias Seleksi Model)
Ketika kita menjalankan penyetelan hyperparameter (mencoba puluhan konfigurasi) pada K-Fold standar, skor validasi terbaik sering kali terpilih karena kebetulan (*optimistic selection bias*). **Nested Cross-Validation** memisahkan siklus evaluasi menjadi dua lapisan:
- **Inner Loop**: Melakukan pencarian hyperparameter terbaik pada data latih outer fold.
- **Outer Loop**: Mengevaluasi model dengan hyperparameter terpilih pada fold validasi independen yang belum pernah melihat proses pencarian parameter.

---

## 5. Penyetelan Hyperparameter (*Hyperparameter Tuning*)

### 5.1 Parameter vs Hyperparameter
- **Model Parameters**: Koefisien internal yang dipelajari selama proses optimasi matematis (contoh: koefisien regresi $w$, bias $b$, ambang split pohon).
- **Hyperparameters**: Konfigurasi eksternal yang ditentukan oleh insinyur sebelum proses pelatihan dimulai (contoh: kedalaman pohon `max_depth`, penalti regularisasi $C$, laju pembelajaran `learning_rate`).

```
+-----------------------------------------------------------------------+
|               GRID SEARCH VS RANDOMIZED SEARCH (2D SPACE)             |
+-----------------------------------------------------------------------+
|  Grid Search (Kombinatorik Kaku):     Randomized Search (Distribusi): |
|     Param 2                               Param 2                     |
|     ^  *   *   *                          ^   *     *                 |
|     |  *   *   *                          |      *     *              |
|     |  *   *   *                          |  *            *           |
|     +---------> Param 1                   +---------> Param 1         |
|  (Hanya mencoba 3 nilai unik per sumbu)  (Mencoba 9 nilai unik per sumbu!|
|                                            Jauh lebih efisien)         |
+-----------------------------------------------------------------------+
```

### 5.2 GridSearchCV vs RandomizedSearchCV
- **GridSearchCV**: Menguji seluruh kombinasi nilai diskrit dalam kisi pencarian secara komprehensif. Menjamin menemukan kombinasi terbaik di dalam kisi, namun kompleksitasnya $\mathcal{O}(\prod_i |P_i| \cdot K)$ meledak secara eksponensial.
- **RandomizedSearchCV (Bergstra & Bengio, 2012)**: Mengambil sampel $N$ kombinasi secara acak dari distribusi probabilitas parameter. Karena pada umumnya hanya sebagian kecil hyperparameter yang benar-benar menentukan performa model (*low effective dimensionality*), penelusuran acak terbukti secara teoritis menemukan konfigurasi optimal jauh lebih cepat dengan anggaran komputasi tetap.

---

## 6. Diagnostik Kurva Pembelajaran & Kurva Validasi

### 6.1 Kurva Pembelajaran (*Learning Curves*)
Learning curve memplot performa model pada data latih dan data validasi terhadap peningkatan ukuran sampel latih ($N_{\text{train}}$).

```
+-----------------------------------------------------------------------+
|                    DIAGNOSTIK LEARNING CURVES                         |
+-----------------------------------------------------------------------+
|  KASUS 1: High Bias (Underfitting)   KASUS 2: High Variance (Overfit) |
|  Skor                                Skor                             |
|  1.0+                                1.0+================= Train      |
|     |                                   |    (Gap Lebar)              |
|     |                                   |----------------- Val        |
|  0.6+----------- Train                  |                             |
|     |----------- Val                    |                             |
|     +------------------> Sampel         +------------------> Sampel   |
|  * Kedua kurva konvergen di skor     * Gap antara latih dan validasi  |
|    rendah. Menambah data SIA-SIA!      lebar. Menambah data SANGAT    |
|    Solusi: Tambah fitur/kapasitas.     MEMBANTU konvergensi!          |
+-----------------------------------------------------------------------+
```

### 6.2 Kurva Validasi (*Validation Curves*)
Validation curve memetakan perubahan skor latih dan validasi terhadap spektrum nilai satu hyperparameter (misalnya `max_depth` atau parameter regulasi $C$). Grafik ini secara visual mengidentifikasi zona underfitting, titik optimal (*sweet spot*), dan awal terjadinya overfitting.

---

## 7. Kalibrasi Probabilitas & Ketidakpastian (*Probability Calibration*)

### 7.1 Kepercayaan Prediksi vs Probabilitas Terkalibrasi
Banyak pengklasifikasi modern (seperti Random Forest, SVM, atau Naive Bayes) menghasilkan skor angka antara $0.0$ dan $1.0$, namun angka tersebut sering kali **tidak terkalibrasi secara probabilitas**:
- **Naive Bayes**: Asumsi independensi fitur yang tidak realistis mendorong probabilitas mendekati $0.0$ atau $1.0$ secara ekstrem (*over-confident*).
- **Random Forest**: Rata-rata voting dari banyak pohon varian menekan skor probabilitas menjauh dari batas $0.0$ dan $1.0$ ke arah tengah ($0.2$ s.d. $0.8$).

### 7.2 Diagram Kalibrasi (*Reliability Diagram*) & Brier Score
Diagram kalibrasi membagi prediksi probabilitas ke dalam sejumlah bin (misal 5 bin: $0.0-0.2, 0.2-0.4, \dots$) dan memplot rata-rata probabilitas prediksi terhadap proporsi positif aktual. Model terkalibrasi sempurna akan jatuh tepat di garis diagonal 45 derajat ($y = x$).

**Brier Score** mengukur galat kuadratik probabilitas:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
Semakin kecil Brier score, semakin handal probabilitas yang dihasilkan model.

### 7.3 Metode Kalibrasi Scikit-Learn (`CalibratedClassifierCV`)
1. **Platt Scaling (Sigmoid)**: Menyesuaikan parameter model logistik pada keluaran model inti. Sangat efektif untuk model berbasis margin linier seperti SVM.
2. **Isotonic Regression**: Menyesuaikan fungsi tangga monoton non-parametrik. Bebas dari asumsi bentuk sigmoid, ideal untuk dataset berukuran besar ($N > 1.000$).

---

## 8. Transparansi & Keterpahaman Model (*Model Explainability*)

### 8.1 Taksonomi Explainability
- **Intrinsic Interpretability**: Model transparan secara desain (Regresi Linier, Decision Tree dangkal).
- **Post-Hoc Explainability**: Metode diagnostik eksternal yang diterapkan setelah model kotak hitam (*black-box*) selesai dilatih.
- **Global Explanations**: Menjelaskan perilaku agregat model di seluruh populasi.
- **Local Explanations**: Menjelaskan mengapa model membuat keputusan tertentu pada satu sampel individual spesifik.

```
+-----------------------------------------------------------------------+
|                    SPEKTRUM MODEL EXPLAINABILITY                      |
+-----------------------------------------------------------------------+
|  Global Explainability:               Local Explainability:           |
|  - Permutation Feature Importance     - SHAP Waterfall Plot           |
|  - Partial Dependence Plots (PDP)     - LIME Local Surrogates         |
|  (Apa yang penting secara umum?)      (Mengapa mesin X diprediksi     |
|                                        rusak hari ini?)               |
+-----------------------------------------------------------------------+
```

### 8.2 Permutation Feature Importance (Breiman, 2001)
Metode evaluasi model-agnostic yang mengukur penurunan performa metrik (misal F1-score) saat nilai satu fitur diacak secara acak pada **data uji**:
1. Hitung skor awal model pada data uji $S_{\text{base}}$.
2. Untuk setiap fitur $j$, acak urutan baris nilai fitur $j$ untuk merusak korelasi prediktifnya dengan target.
3. Hitung skor baru $S_{\text{perm}}$.
4. Kepentingan fitur: $I_j = S_{\text{base}} - S_{\text{perm}}$.

*Peringatan Metodologis*: Feature importance hanya mengukur ketergantungan prediktif statistik di dalam dataset, **bukan bukti kausalitas fisik (*correlation is not causation*)**.

### 8.3 Partial Dependence Plots (PDP) & ICE
- **Partial Dependence Plots (PDP)**: Menunjukkan rata-rata efek marjinal suatu fitur terhadap target prediksi setelah memarginalkan pengaruh seluruh fitur lain. Mengungkap sifat non-linier dan ambang batas kritis.
- **Individual Conditional Expectation (ICE)**: Memplot kurva respons untuk setiap sampel individual secara terpisah. Jika kurva ICE saling bersilangan, ini merupakan indikasi visual adanya interaksi kuat antar-fitur.

### 8.4 Ikhtisar SHAP & LIME
- **SHAP (SHapley Additive exPlanations)**: Berlandaskan Teori Permainan Koperasi (*Cooperative Game Theory*). Menghitung kontribusi marjinal setiap fitur ke seluruh kemungkinan koalisi subset fitur. Memiliki jaminan aksiomatis konsistensi dan efisiensi lokal yang elegan.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Membangun model linier pengganti lokal (*local surrogate*) di sekitar tetangga terdekat sampel yang ingin dijelaskan.

---

## 9. Analisis Kegagalan Model (*Model Error Analysis*) & Responsible AI

Evaluasi agregat (seperti akurasi 85%) sering kali menyembunyikan kelemahan kritis sistem. Analisis kegagalan sistematis membedah sampel-sampel di mana model melakukan kesalahan:

```
+-----------------------------------------------------------------------+
|                    ALUR KERJA ERROR ANALYSIS                          |
+-----------------------------------------------------------------------+
|  [Hasil Prediksi Model pada Test Set]                                 |
|         │                                                             |
|         ├───────────────────────────────┐                             |
|         ▼                               ▼                             |
|  [Kasus False Positives (FP)]    [Kasus False Negatives (FN)]         |
|  - Alarm Palsu                   - GAGAL DETEKSI KRITIS!              |
|  - Hitung Biaya Inspeksi Semu    - Periksa Nilai Probabilitas         |
|                                  - Identifikasi Noise/Sensor Rusak    |
|                                  - Audit Subgrup Perangkat Lemah      |
+-----------------------------------------------------------------------+
```

### 9.1 Audit Disparitas Subgrup (*Subgroup Auditing*)
Menghitung metrik performa secara terpisah untuk setiap kategori entitas (misal per tipe mesin atau per wilayah sensor). Jika sebuah model memiliki akurasi global 85%, namun memiliki akurasi 95% pada perangkat baru dan hanya 50% pada perangkat tipe lama, model tersebut memiliki disparitas performa yang membahayakan operasional.

### 9.2 Pergeseran Data (*Dataset Shift*)
1. **Covariate Shift**: Distribusi input $P(\mathbf{x})$ bergeser seiring waktu, sementara hubungan target $P(y|\mathbf{x})$ tetap konstan.
2. **Concept Drift**: Hubungan fungsional antara fitur dan target $P(y|\mathbf{x})$ berubah akibat dinamika dunia nyata (misal aus mekanis tipe baru).
3. **Out-of-Distribution (OOD)**: Model menerima masukan dengan nilai di luar batas domain pelatihan (misal suhu $> 100^\circ\text{C}$). Model harus dilengkapi mekanisme deteksi OOD untuk menolak inferensi berisiko tinggi.

---

## 10. Laporan Evaluasi Model Reproducible (*Model Cards*)

Standar dokumentasi sistem machine learning profesional merangkum seluruh temuan eksperimen ke dalam kartu model (*Model Card*) yang mencakup:
1. **Ringkasan Masalah & Sumber Data**: Definisi target, ukuran data latih/uji terisolasi.
2. **Benchmark Terhadap Baseline**: Bukti keunggulan atas *Dummy Classifier*.
3. **Hasil Validasi Silang**: Rerata metrik F1-Score beserta deviasi standarnya.
4. **Hasil Uji Akhir**: Metrik Precision, Recall, F1, ROC-AUC, dan Brier score pada test set tak tersentuh.
5. **Explainability & Batasan Operasional**: Fitur paling menentukan dan batasan kondisi lingkungan operasional yang diizinkan.

---

## 11. Latihan Bertingkat (Tiered Exercises)

### Tier 1: Pemahaman Konseptual
1. Mengapa menyetel hyperparameter langsung pada data uji (*test set*) merupakan pelanggaran fatal dalam machine learning?
   - *Jawaban*: Data uji akan kehilangan sifat netralnya sebagai pengukur generalisasi objektif. Hyperparameter yang dipilih akan mengalami overfitting terselubung terhadap data uji, menghasilkan estimasi performa optimis semu yang gagal saat diterapkan di dunia nyata (*data leakage*).
2. Sebutkan perbedaan esensial antara *Stratified K-Fold* dan *Group K-Fold*!
   - *Jawaban*: Stratified K-Fold menjaga agar persentase sebaran kelas target seimbang di setiap fold (digunakan untuk klasifikasi). Group K-Fold menjamin seluruh observasi dari kelompok/entitas yang sama (seperti ID sensor atau pasien) berada secara eksklusif di fold latih atau fold validasi untuk mencegah kebocoran informasi antar-entitas.
3. Mengapa *Brier Score* lebih informatif daripada sekadar akurasi pada sistem pengambilan keputusan berbasis risiko?
   - *Jawaban*: Akurasi hanya menilai apakah prediksi label biner benar atau salah pada threshold tetap, sedangkan Brier Score mengukur akurasi kuadratik dari estimasi probabilitas kontinu ($p_i - y_i)^2$. Probabilitas yang terkalibrasi akurat sangat krusial untuk menghitung ekspektasi biaya kerugian finansial atau risiko keselamatan.

### Tier 2: Perhitungan Matematis Manual
Diberikan Confusion Matrix dari model deteksi kerusakan sensor pada 100 sampel uji:
- $\text{TP} = 40$
- $\text{FP} = 10$
- $\text{FN} = 5$
- $\text{TN} = 45$

1. **Hitung Akurasi, Presisi, dan Recall**:
   $$\text{Accuracy} = \frac{40 + 45}{100} = \frac{85}{100} = 85.00\%$$
   $$\text{Precision} = \frac{40}{40 + 10} = \frac{40}{50} = 80.00\%$$
   $$\text{Recall} = \frac{40}{40 + 5} = \frac{40}{45} \approx 88.89\%$$
2. **Hitung F1-Score**:
   $$F_1 = 2 \cdot \frac{0.80 \cdot 0.8889}{0.80 + 0.8889} = \frac{1.4222}{1.6889} \approx 0.8421 \quad (84.21\%)$$
3. **Analisis Biaya Asimetris**: Jika biaya $\text{FN} = \$1.000$ dan biaya $\text{FP} = \$100$, berapakah total biaya kesalahan model?
   $$\text{Total Biaya} = (5 \times \$1.000) + (10 \times \$100) = \$5.000 + \$1.000 = \$6.000$$

### Tier 3: Rekayasa Sistem & Desain Arsitektur
Sebuah pabrik semikonduktor memiliki 50 mesin litografi yang memproduksi chip 24/7. Tim data science ingin menerapkan model pendeteksi cacat chip berbasis sensor.
1. Strategi validasi silang apa yang wajib diterapkan dan mengapa?
   - *Jawaban*: Wajib menggunakan kombinasi `GroupKFold` (berdasarkan ID mesin litografi) dan `TimeSeriesSplit`. Jika menggunakan Random K-Fold, model akan menghafal karakteristik kebisingan fisik mesin tertentu dan mengalami kebocoran temporal, sehingga saat dipasang pada mesin baru di masa depan, akurasi akan anjlok drastis.
2. Jika prevalensi chip cacat hanya 0.5% (5 dari 1.000 chip), metrik apa yang harus digunakan untuk memilih model terbaik selama GridSearchCV?
   - *Jawaban*: Wajib menggunakan **PR-AUC (Precision-Recall AUC)** atau **F1-Score makro**, BUKAN akurasi atau ROC-AUC. Model yang selalu memprediksi chip normal akan mencetak akurasi 99.5%, namun PR-AUC akan mendeteksi bahwa model tersebut sama sekali tidak berguna.

---

## 12. Mini Project Mandiri: Comprehensive Evaluation and Explainability (Rubrik 100 Poin)

Mahasiswa diwajibkan menyusun pipeline evaluasi, penyetelan hyperparameter, kalibrasi, dan explainability menyeluruh pada dataset armada telemetri baru.

### Rubrik Penilaian Proyek
- **Desain Evaluasi & Pembagian Data Bebas Bocor (25 Poin)**: Pembagian train-test terisolasi, implementasi Cross-Validation yang sesuai dengan struktur data (Stratified/Group), zero data leakage pada pipeline.
- **Penyetelan Hyperparameter & Diagnostik Kurva (25 Poin)**: Menerapkan GridSearchCV / RandomizedSearchCV di dalam lingkungan CV, menyajikan kurva pembelajaran dan kurva validasi untuk mendiagnosis bias-varians.
- **Kalibrasi Probabilitas & Analisis Kegagalan (25 Poin)**: Menghitung Brier Score, diagram kalibrasi, ekstraksi dan bedah kasus False Negative serta audit disparitas subgrup.
- **Explainability & Dokumentasi Model Card (25 Poin)**: Menghitung Permutation Feature Importance pada data uji, Partial Dependence Plot, serta menyusun laporan Model Card reproducible.

---

## 13. Referensi Ilmiah Primer

1. **Breiman, L. (2001)**. *Random Forests*. Machine Learning, 45(1), 5-32.
2. **Bergstra, J., & Bengio, Y. (2012)**. *Random search for hyper-parameter optimization*. Journal of Machine Learning Research, 13(Feb), 281-305.
3. **Platt, J. (1999)**. *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. Advances in Large Margin Classifiers, 10(3), 61-74.
4. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016)**. *'Why should I trust you?': Explaining the predictions of any classifier*. KDD-2016, 1135-1144.
5. **Lundberg, S. M., & Lee, S. I. (2017)**. *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765-4774.

