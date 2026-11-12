# Bab 20: Machine Learning Capstone Project

## Puncak Integrasi Siklus Hidup Rekayasa Pembelajaran Mesin (Level 2 Finalization)

---

## 1. Hakikat dan Filosofi Capstone Rekayasa Pembelajaran Mesin

Selamat datang di puncak pembelajaran **Level 2: Machine Learning**. Bab 20 dirancang bukan sekadar sebagai tugas pemrograman biasa, melainkan sebagai wadah pembuktian komprehensif atas seluruh kompetensi yang telah Anda bangun dari Bab 11 hingga Bab 19.

Dalam praktik rekayasa industri modern, model pembelajaran mesin tidak pernah hadir dalam ruang hampa. Pelatihan algoritma hanyalah sebagian kecil dari keseluruhan siklus hidup sistem kecerdasan buatan (*AI System Lifecycle*). Keberhasilan sebuah inisiatif *Machine Learning* diukur dari kemampuannya menjawab kebutuhan bisnis nyata, ketahanannya saat menghadapi data tak terduga (*robustness*), keadilan perilakunya lintas berbagai kelompok pengguna (*fairness*), serta transparansi proses pengambil keputusannya (*explainability*).

```
+---------------------------------------------------------------------------------------------------------+
|                                    SIKLUS HIDUP LENGKAP SISTEM ML LEVEL 2                               |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [1. FORMULASI MASALAH]  ───> Definisikan metrik bisnis, kerugian finansial, & batasan operasional      |
|            │                                                                                            |
|            v                                                                                            |
|  [2. ISOLASI DATA UJI]   ───> Kunci 20% untouched test set SEBELUM transformasi apa pun                 |
|            │                                                                                            |
|            v                                                                                            |
|  [3. REKAYASA FITUR]     ───> Ekstraksi parameter domain fisik di dalam Pipeline bebas kebocoran       |
|            │                                                                                            |
|            v                                                                                            |
|  [4. MODEL CANDIDATES]   ───> Benchmark arsitektur linier, SVM kernel, Random Forest, & Gradient Boost  |
|            │                                                                                            |
|            v                                                                                            |
|  [5. TUNING HIPERPARAM]  ───> Optimasi 5-Fold Stratified Cross-Validation pada data latih               |
|            │                                                                                            |
|            v                                                                                            |
|  [6. EVALUASI TEST SET]  ───> Uji generalisasi pada data terisolasi: Akurasi, F1, Recall, ROC-AUC       |
|            │                                                                                            |
|            v                                                                                            |
|  [7. ANALISIS BIAYA]     ───> Diseksi galat False Negative & optimasi decision threshold industri       |
|            │                                                                                            |
|            v                                                                                            |
|  [8. RESPONSIBLE AI]     ───> Audit disparitas armada mesin, Model Card, & batas keselamatan            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Formulasi Masalah Bisnis Menjadi Masalah Teknis Terukur

Langkah awal dari setiap proyek Capstone adalah mengubah pernyataan masalah bisnis yang sering kali ambigu menjadi formulasi matematis yang terukur dan dapat dioptimalkan oleh algoritma pembelajaran mesin.

```
                    +------------------------------------------+
                    |       Tantangan Bisnis Industri          |
                    | "Bagaimana mencegah mesin kompresor      |
                    |  pabrik meledak atau mengalami breakdown |
                    |  mendadak tanpa menimbulkan biaya        |
                    |  pemeliharaan preventif yang boros?"     |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    |       Formulasi Teknis Pembelajaran      |
                    | Tipe Masalah : Klasifikasi Biner         |
                    | Target Y     : machine_failure in {0, 1} |
                    | Fitur X      : Sinyal sensor telemetri   |
                    | Fungsi Rugi  : Cost-Weighted Binary Loss |
                    | Metrik Kunci : F1-Score & Recall (Kerusakan) |
                    +------------------------------------------+
```

### Pemetaan Nilai Bisnis ke Fungsi Biaya (*Cost Function*)
Dalam skenario pemeliharaan mesin pabrik petrokimia:
1. **False Negative (FN)**: Mesin mengalami kerusakan kritis, namun model memprediksinya sebagai "Normal". Mesin beroperasi hingga hancur total, menyebabkan penghentian jalur produksi pabrik (*unplanned plant shutdown*). Kerugian operasional diestimasi sebesar **\$5,000**.
2. **False Positive (FP)**: Mesin sebenarnya sehat, namun model memprediksinya sebagai "Kerusakan". Teknisi dikirim untuk melakukan inspeksi visual dan pengecekan vibrasi. Biaya inspeksi ini hanya memakan waktu kerja teknisi, diestimasi sebesar **\$200**.

Biaya finansial total yang diharapkan (*Expected Monetary Loss*) dimodelkan sebagai:
$$\text{Total Financial Risk} = C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP} = 5000 \cdot \text{FN} + 200 \cdot \text{FP}$$

Formula ini membuktikan bahwa mengejar akurasi tinggi semata bukanlah tujuan akhir. Model dengan akurasi 95% yang melewatkan 5 kegagalan mesin jauh lebih merugikan daripada model dengan akurasi 92% yang menangkap seluruh indikasi kegagalan mesin.

---

## 3. Disiplin Partisi Data & Pencegahan Kebocoran Informasi

Salah satu kesalahan fatal dalam proyek rekayasa pembelajaran mesin adalah terjadinya kebocoran informasi (*data leakage*). Kebocoran data terjadi ketika informasi dari himpunan evaluasi (data uji) merembes masuk ke dalam proses pelatihan model.

```
+---------------------------------------------------------------------------------------------------------+
|                                    ARSITEKTUR PARTISI DATA BEBAS KEBOCORAN                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                                       POPULASI DATA TELEMETRI                                           |
|                                            (N = 600)                                                    |
|                                                │                                                        |
|                       ┌────────────────────────┴────────────────────────┐                               |
|                       │                                                 │                               |
|                       v (80%)                                           v (20%)                         |
|             TRAINING SET (N = 480)                            UNTOUCHED TEST SET (N = 120)              |
|                       │                                                 │                               |
|       ┌───────────────┴───────────────┐                                 │ (DIKUNCI RAPAT)               |
|       │  5-Fold Stratified CV         │                                 │ (Dilarang keras disentuh      |
|       │  - Pipeline Fit Transform     │                                 │  selama eksplorasi, tuning,   |
|       │  - Seleksi Hyperparameter     │                                 │  maupun rekayasa fitur!)      |
|       │  - Cross-Validation F1 Tuning │                                 │                               |
|       └───────────────┬───────────────┘                                 │                               |
|                       │                                                 │                               |
|                       v                                                 │                               |
|               CHAMPION ESTIMATOR ───────────────────────────────────────┘                               |
|                                     │                                                                   |
|                                     v Evaluasi Generalisasi Akhir                                       |
|                            FINAL PERFORMANCE AUDIT                                                      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Golden Rules Isolasi Data Capstone:
1. **Pemisahan di Gerbang Pertama**: Pemisahan data latih ($80\%$) dan data uji ($20\%$) dilakukan sebelum visualisasi distribusi lanjutan, sebelum imputasi nilai hilang, dan sebelum rekayasa fitur apa pun.
2. **Kapsulisasi Pipeline**: Seluruh operasi pembersihan data, penskalaan (*StandardScaler*), dan transformasi rekayasa fitur wajib dibungkus di dalam `sklearn.pipeline.Pipeline`.
3. **Fit HANYA pada Data Latih**: Pemanggilan metode `.fit()` hanya diperbolehkan pada `X_train`. Pada data uji `X_test`, sistem HANYA boleh memanggil `.transform()` atau `.predict()`.

---

## 4. Rekayasa Fitur Domain Berbasis Pengetahuan Fisik

Pembelajaran mesin modern yang sukses selalu memadukan algoritma dengan pengetahuan domain (*domain knowledge*). Dalam sistem pemantauan kompresor industri, sensor telemetri mentah merekam 6 parameter fisik:

```
+---------------------------------------------------------------------------------------------------------+
|                                  PARAMETER SENSOR TELEMETRI MENTAH                                      |
+---------------------------------------------------------------------------------------------------------+
| 1. vibration_rms_g       : Akselerasi getaran efektif RMS (g)                                           |
| 2. bearing_temperature_c : Suhu bantalan kompresor (°C)                                                |
| 3. rotational_speed_rpm  : Kecepatan putaran poros motor (RPM)                                          |
| 4. hydraulic_pressure_bar: Tekanan oli hidrolik pelumas (bar)                                           |
| 5. motor_current_amp     : Konsumsi arus listrik motor penggerak (A)                                    |
| 6. oil_acidity_tan       : Total Acid Number (TAN) tingkat keasaman oli pelumas                         |
+---------------------------------------------------------------------------------------------------------+
```

Berdasarkan teori mekanika rotasi dan tribologi pelumasan mesin, dirumuskan 3 fitur komposit tingkat lanjut:

```
                                  [FITUR TELEMETRI MENTAH]
                                             │
      ┌──────────────────────────────────────┼──────────────────────────────────────┐
      │                                      │                                      │
      v                                      v                                      v
 [Suhu & Tekanan]                    [Getaran & Kecepatan]                  [Arus & Keasaman]
      │                                      │                                      │
      v                                      v                                      v
(Suhu / Tekanan)                     (Getaran^2 * RPM/1000)                   (Arus * Asam)
      │                                      │                                      │
      v                                      v                                      v
+-----------------------+      +---------------------------+      +----------------------------+
| Thermal-Pressure      |      | Vibration Power Index     |      | Lubricant Stress Index     |
| Ratio                 |      |                           |      |                            |
| Menangkap deplesi     |      | Mengukur kepadatan daya   |      | Mendeteksi gesekan mekanis |
| viskositas pelumas    |      | kinetik destruktif        |      | akibat asam oli pekat      |
+-----------------------+      +---------------------------+      +----------------------------+
```

### Formulasi Matematika Fitur Rekayasa:
1. **Rasio Beban Termal-Tekanan**:
   $$f_{\text{thermal-pressure}} = \frac{T_{\text{bearing}}}{P_{\text{hydraulic}}}$$
2. **Indeks Daya Getaran Mekanis**:
   $$f_{\text{vib-power}} = (a_{\text{RMS}})^2 \times \left(\frac{\omega_{\text{RPM}}}{1000}\right)$$
3. **Indeks Stres Pelumasan Listrik**:
   $$f_{\text{lubricant-stress}} = I_{\text{motor}} \times \text{TAN}_{\text{oil}}$$

Ketiga fitur baru ini secara dramatis meningkatkan daya pisang model terhadap anomali non-linier yang sulit dipisahkan oleh kombinasi linier biasa.

---

## 5. Taksonomi Model Kandidat & Validasi Silang Kredibel

Untuk menemukan solusi terbaik, Capstone Level 2 menguji 4 keluarga algoritma yang merepresentasikan spektrum pembelajaran mesin terawasi (*Supervised Learning*):

```
+---------------------------------------------------------------------------------------------------------+
|                               SPEKTRUM 4 KELUARGA ALGORITMA LEVEL 2                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [1. MODEL LINIER]            [2. KERNEL MARGIN]            [3. BAGGING ENSEMBLE]    [4. BOOSTING]      |
|  Logistic Regression          Support Vector Machine        Random Forest            Gradient Boosting  |
|  (L2 Regularized)             (RBF Kernel)                  (100-150 Trees)          (Stochastic Grad)  |
|                                                                                                         |
|  Kelebihan:                   Kelebihan:                    Kelebihan:               Kelebihan:         |
|  - Cepat dilatih              - Efektif dimensi tinggi      - Tahan overfitting      - Menitikberatkan  |
|  - Sangat mudah dipahami      - Batas non-linier fleksibel  - Menangkap interaksi      sampel sulit     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Protokol Validasi Silang: 5-Fold Stratified Cross-Validation
Data latih (480 sampel) dibagi menjadi 5 lipatan (*folds*) secara berimbang:

```
  Fold 1: [ UJI (20%) ] [ LATIH (80%) .................................................. ]
  Fold 2: [ LATIH (20%) ] [ UJI (20%) ] [ LATIH (60%) .................................. ]
  Fold 3: [ LATIH (40%) ................. ] [ UJI (20%) ] [ LATIH (40%) ................. ]
  Fold 4: [ LATIH (60%) .................................. ] [ UJI (20%) ] [ LATIH (20%) ]
  Fold 5: [ LATIH (80%) .................................................. ] [ UJI (20%) ]
```

### Hasil Komparasi Awal Model Kandidat (Terverifikasi):
| Model Kandidat | Rata-rata Akurasi | Recall (Kerusakan) | Rata-rata F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression (L2)** | 92.08% | 93.36% | 0.9336 | 0.9818 |
| **Support Vector Machine (RBF)** | 92.50% | 94.05% | 0.9373 | 0.9697 |
| **Random Forest Classifier** | 92.29% | 93.01% | 0.9350 | 0.9762 |
| **Gradient Boosting Classifier** | 91.88% | 93.00% | 0.9319 | 0.9747 |

*Keputusan Rekayasa*: Model ensemble Random Forest dan Gradient Boosting dipilih untuk melangkah ke tahap penyetelan hyperparameter intensif karena stabilitas variansinya yang unggul pada data sensor industri.

---

## 6. Optimasi Hyperparameter Sistematis & Trade-off Bias-Variansi

Optimasi hyperparameter dilakukan menggunakan `GridSearchCV` dengan fungsi objektif memaksimalkan `f1` pada 5-Fold Stratified CV.

```
                    +------------------------------------------+
                    |        Ruang Parameter Random Forest     |
                    | n_estimators      : [50, 100, 150]       |
                    | max_depth         : [5, 8, 12, None]     |
                    | min_samples_split : [2, 5, 10]           |
                    | Total Kombinasi   : 36 Model             |
                    +--------------------+---------------------+
                                         │
                                         v
                    +------------------------------------------+
                    |          Hasil GridSearchCV              |
                    | Konfigurasi Terbaik:                     |
                    | - n_estimators      : 150                |
                    | - max_depth         : 8                  |
                    | - min_samples_split : 10                 |
                    | F1-Score Validasi   : 0.9390 (Meningkat) |
                    +------------------------------------------+
```

Membatasi `max_depth=8` dan menaikkan `min_samples_split=10` terbukti secara empiris mencegah pohon keputusan melakukan penghafalan derau (*noise memorization*), menghasilkan model yang kokoh dan berdaya generalisasi tinggi.

---

## 7. Evaluasi Data Uji Terisolasi & Analisis Biaya Operasional

Setelah model champion (Tuned Random Forest) terkunci, barulah data uji terisolasi (*Untouched Test Set*, $N=120$) dibuka untuk evaluasi verifikasi akhir.

```
+---------------------------------------------------------------------------------------------------------+
|                               CONFUSION MATRIX EVALUASI TEST SET (N = 120)                              |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                                           PREDIKSI MODEL                                                |
|                                   Normal (0)           Kerusakan (1)                                    |
|                              +-----------------------+-----------------------+                          |
|             Normal (0)       |   TN = 44             |   FP = 5              |   Total Aktual = 49      |
|  AKTUAL                      |   (Operasi Stabil)    |   (Inspeksi Tambahan) |                          |
|                              +-----------------------+-----------------------+                          |
|             Kerusakan (1)    |   FN = 3              |   TP = 68             |   Total Aktual = 71      |
|                              |   (Kerusakan Lolos)   |   (Deteksi Sukses)    |                          |
|                              +-----------------------+-----------------------+                          |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Metrik Kinerja Akhir Terverifikasi:
- **Akurasi Keseluruhan**: **93.33%**
- **Presisi Kerusakan**: **93.15%**
- **Recall Kerusakan**: **95.77%** (68 dari 71 potensi kerusakan fatal berhasil dicegah)
- **F1-Score Harmonik**: **0.9444**
- **ROC-AUC Diskriminasi**: **0.9859**

### Penyetelan Ambang Batas Berbasis Biaya (*Cost-Sensitive Threshold Tuning*)
Secara default, Scikit-Learn menggunakan ambang batas $\theta = 0.50$. Namun, dengan biaya $\text{FN} = \$5,000$ dan $\text{FP} = \$200$, kita melakukan simulasi pergeseran ambang batas probabilitas:

```
+---------------------------------------------------------------------------------------------------------+
|                              SIMULASI TRADE-OFF AMBANG BATAS OPERASIONAL                                |
+---------------------------------------------------------------------------------------------------------+
|  Threshold    Recall       Presisi      FN       FP       Total Risiko Biaya ($)                        |
|  ---------------------------------------------------------------------------                            |
|  0.20           98.59%       85.37%     1        12       $ 7,400                                       |
|  0.30           98.59%       92.11%     1        6        $ 6,200  <-- AMBANG BATAS OPTIMAL FINANSIAL   |
|  0.35           98.59%       92.11%     1        6        $ 6,200                                       |
|  0.40           97.18%       92.00%     2        6        $11,200                                       |
|  0.50           95.77%       93.15%     3        5        $16,000  <-- AMBANG DEFAULT SKLEARN           |
|  0.60           92.96%       94.29%     5        4        $25,800                                       |
|  0.70           90.14%       95.52%     7        3        $35,600                                       |
+---------------------------------------------------------------------------------------------------------+
```

```
     Risiko Finansial ($)
     $40,000 |                                              * (Th=0.70)
     $30,000 |                                   * (Th=0.60)
     $20,000 |                        * (Th=0.50 Default = $16,000)
     $10,000 |             * (Th=0.40)
      $6,200 |    * (Th=0.30 OPTIMAL = $6,200)
             +-------------------------------------------------------->
               0.20   0.30   0.40   0.50   0.60   0.70  Threshold
```

*Temuan Strategis*: Dengan menurunkan ambang aktivasi peringatan dari $0.50$ ke **$0.30$**, jumlah False Negative ditekan dari 3 kasus menjadi hanya 1 kasus. Meskipun False Positive naik tipis dari 5 ke 6 kasus, penghematan finansial pabrik mencapai **\$9,800 (penurunan risiko sebesar 61.25%)**.

---

## 8. Transparansi Model (Explainability) & Validasi Fisika

Model yang digunakan di lingkungan industri berisiko tinggi wajib memiliki penalaran yang dapat dijelaskan (*explainable*). Melalui *Permutation Feature Importance*, diukur seberapa besar performa F1 merosot ketika nilai masing-masing fitur diacak secara acak pada data uji:

```
+---------------------------------------------------------------------------------------------------------+
|                            PERMUTATION FEATURE IMPORTANCE PADA DATA UJI                                 |
+---------------------------------------------------------------------------------------------------------+
|  Fitur Fisik                Mean Drop in F1     Visualisasi Pengaruh                                    |
|  -----------------------------------------------------------------------------------------------------  |
|  bearing_temperature_c         0.1278           [########################################] (Suhu Kritis)|
|  vibration_rms_g               0.1217           [######################################  ] (Getaran)    |
|  oil_acidity_tan               0.0161           [#####                                   ] (Asam Oli)   |
|  hydraulic_pressure_bar        0.0085           [###                                     ] (Tekanan)    |
|  motor_current_amp             0.0052           [##                                      ] (Arus)       |
|  rotational_speed_rpm         -0.0014           [                                        ] (Kecepatan)  |
+---------------------------------------------------------------------------------------------------------+
```

### Profil Ketergantungan Parsial (*Partial Dependence Plot - PDP*)
Analisis ketergantungan parsial pada fitur dominan `bearing_temperature_c` membuktikan adanya transisi fase fisik yang tajam:

```
  Peluang Kerusakan
     1.0 |
     0.8 |                                      *---*---*---*---*---* (Peluang > 82%)
     0.6 |                                 *
     0.4 |     *---*---*---*          *
     0.2 |
     0.0 +-------------------------------------------------------->
              58°C   62°C   66°C   70°C   75°C   79°C   85°C  Suhu Bantalan (°C)
```

Ketika suhu bantalan melampaui ambang batas $75^\circ\text{C}$, estimasi peluang kerusakan melonjak tajam dari $45\%$ menjadi $>80\%$. Fenomena ini selaras sempurna dengan hukum termodinamika degradasi film pelumas sintetik ISO VG-46, membuktikan bahwa model tidak menghafal relasi semu melainkan merefleksikan hukum fisik nyata.

---

## 9. Responsible AI, Audit Disparitas & Model Card Governance

Sebelum model dideploy ke lingkungan produksi, wajib dilakukan audit etika dan tata kelola sistem kecerdasan buatan (*Responsible AI*).

```
+---------------------------------------------------------------------------------------------------------+
|                            AUDIT DISPARITAS PERFORMA 12 UNIT POMPA KOMPRESOR                            |
+---------------------------------------------------------------------------------------------------------+
|  Unit Mesin      Sampel Uji     Akurasi Test       F1-Score       Status Audit                          |
|  -----------------------------------------------------------------------------------------------------  |
|  PUMP_01            7             85.71%            0.8889        Lolos                                 |
|  PUMP_02           10             80.00%            0.8571        Lolos                                 |
|  PUMP_03            8             87.50%            0.8000        Lolos (Batas Bawah)                   |
|  PUMP_04            9            100.00%            1.0000        Sempurna                              |
|  PUMP_05           13             92.31%            0.9412        Lolos                                 |
|  PUMP_06           14            100.00%            1.0000        Sempurna                              |
|  PUMP_07            5            100.00%            1.0000        Sempurna                              |
|  PUMP_08            9             88.89%            0.9091        Lolos                                 |
|  PUMP_09            7            100.00%            1.0000        Sempurna                              |
|  PUMP_10           11            100.00%            1.0000        Sempurna                              |
|  PUMP_11           10            100.00%            1.0000        Sempurna                              |
|  PUMP_12           17             88.24%            0.9000        Lolos                                 |
|  =====================================================================================================  |
|  Disparitas F1 Maksimum Antar Unit: 0.2000 (Toleransi Sistem: < 0.3000) -> AUDIT KELAYAKAN LOLOS        |
+---------------------------------------------------------------------------------------------------------+
```

### Model Card Spesifikasi Tata Kelola
```
+---------------------------------------------------------------------------------------------------------+
|                                   MODEL CARD: INDUSTRIAL PREDICTIVE MAINTENANCE                         |
+---------------------------------------------------------------------------------------------------------+
| 1. Model Details       : Tuned Random Forest Classifier (150 trees, max_depth=8). Rilis Nov 2026.       |
| 2. Intended Use        : Sistem pendukung keputusan (Decision Support) peringatan dini keausan mesin.   |
| 3. Out-of-Scope Use    : Dilarang digunakan sebagai pengendali pemutus otomatis (Emergency Shutdown)   |
|                          tanpa integrasi sistem instrumentasi keselamatan bersertifikat SIL-3.          |
| 4. Human-in-the-Loop   : Peringatan model wajib dikonfirmasi inspeksi teknisi dalam kurun 2 jam kerja.  |
| 5. Maintenance Cadence : Evaluasi drift data wajib dilakukan setiap 30 hari kalender.                   |
+---------------------------------------------------------------------------------------------------------+
```

---

## 10. Katalog 5 Jalur Proyek Capstone Terbuka

Peserta Capstone dipersilakan memilih salah satu dari 5 jalur proyek komprehensif berikut:

### Jalur 1 (Proyek Acuan Repositori): Pemeliharaan Prediktif Kompresor Industri
- **Domain**: Pemantauan keandalan aset mesin rotasi petrokimia.
- **Tipe Data**: Telemetri getaran, suhu bantalan, tekanan hidrolik, dan keasaman pelumas.
- **Tujuan**: Memprediksi kegagalan mesin sebelum terjadi kerusakan fatal untuk menekan biaya downtime.

### Jalur 2: Deteksi Transaksi Keuangan Mencurigakan (*Financial Fraud Detection*)
- **Domain**: Layanan perbankan digital dan gerbang pembayaran (*payment gateway*).
- **Tipe Data**: Fitur anonim transaksi keuangan, riwayat frekuensi, geolokasi, dan deviasi pola belanja.
- **Tujuan**: Mengklasifikasikan transaksi penipuan pada distribusi kelas yang sangat timpang (*extreme class imbalance*).

### Jalur 3: Prediksi Daya Keluaran Pembangkit Tenaga Surya (*Solar Power Forecasting*)
- **Domain**: Jaringan listrik cerdas (*Smart Grid*) dan energi terbarukan.
- **Tipe Data**: Radiasi sinar matahari (DNI/GHI), tutupan awan, kelembapan udara, kecepatan angin, dan suhu sel panel.
- **Tujuan**: Regresi estimasi daya listrik kilowatt-hour (kWh) untuk perencanaan beban jaringan PLN.

### Jalur 4: Stratifikasi Risiko Kardiometabolik Pasien Rawat Jalan
- **Domain**: Sistem pendukung keputusan klinis (*Clinical Decision Support*).
- **Tipe Data**: Tekanan darah, profil lipid, indeks massa tubuh (BMI), biomarker HbA1c, dan demografi.
- **Tujuan**: Stratifikasi risiko komplikasi kardiovaskular dalam 5 tahun untuk intervensi preventif dokter.

### Jalur 5: Prediksi Churn Pelanggan & Estimasi Nilai Seumur Hidup (*Customer LTV*)
- **Domain**: E-Commerce dan platform SaaS berbasis langganan.
- **Tipe Data**: Frekuensi login, interval transaksi, sentimen komplain tiket, dan volume keranjang belanja.
- **Tujuan**: Memprediksi probabilitas berhenti berlangganan sekaligus memperkirakan nilai moneter pelanggan.

---

## 11. Rubrik Penilaian Proyek Capstone (100 Poin)

| Kategori | Bobot | Kriteria Evaluasi Skor Maksimal |
|---|:---:|---|
| **Formulasi Masalah & Baseline** | 15% | Masalah bisnis diterjemahkan tajam ke metrik ML terukur. Terdapat evaluasi baseline sederhana (Dummy & Simple Logistic Regression) sebagai batas acuan bawah. |
| **Pipeline Bebas Bocor & Fitur** | 20% | Pemisahan data uji 20% dilakukan secara terisolasi tanpa kebocoran (*zero data leakage*). Seluruh prapemrosesan dikapsulisasi dalam Scikit-Learn Pipeline. Menghasilkan minimal 3 fitur domain kreatif. |
| **Pemodelan & Tuning** | 25% | Membandingkan minimal 3 arsitektur algoritma Level 2 yang berbeda secara adil menggunakan Stratified Cross-Validation. Melakukan tuning hyperparameter sistematis (GridSearchCV). |
| **Evaluasi Test Set & Galat** | 20% | Evaluasi jujur pada *untouched test set*. Menyajikan analisis matriks konfusi, diseksi mendalam False Negative/Positive, dan simulasi penyesuaian ambang batas berbasis matriks biaya industri. |
| **Responsible AI & Dokumentasi** | 20% | Analisis transparansi fitur (Permutation Importance/PDP). Audit keadilan performa subkelompok. Menyusun Model Card operasional. Seluruh kode modular, bersih, dan reproducible 100%. |

---

## 12. Latihan Mandiri 3 Tingkat (Exercises)

### Tingkat 1: Pemahaman Konseptual
1. Mengapa keberadaan baseline sederhana (*Dummy Classifier*) wajib disertakan dalam laporan Capstone meskipun performanya dipastikan buruk?
2. Jelaskan perbedaan mendasar antara kebocoran data (*data leakage*) yang terjadi pada tahap penskalaan fitur (*scaling*) versus kebocoran data pada tahap seleksi hyperparameter!

### Tingkat 2: Analisis Matematis
1. Diberikan matriks biaya industri: biaya $\text{FN} = \$8,000$ dan biaya $\text{FP} = \$150$. Jika model Anda menghasilkan 2 FN dan 10 FP pada ambang batas 0.50, serta 0 FN dan 35 FP pada ambang batas 0.25, hitunglah total risiko finansial masing-masing ambang batas dan tentukan ambang batas mana yang wajib dipilih manajemen!
2. Buktikan secara matematis mengapa metrik ROC-AUC tidak sensitif terhadap pergeseran ambang batas klasifikasi ($\theta$), sedangkan metrik F1-Score sangat sensitif terhadap perubahan nilai $\theta$!

### Tingkat 3: Rekayasa Komputasional
1. Modifikasi berkas `candidate_models.py` untuk menambahkan algoritma K-Nearest Neighbors ($k=7$) dan Decision Tree Classifier. Bandingkan hasil validasi silangnya dengan Random Forest!
2. Rancang modul deteksi pergeseran konsep (*concept drift detector*) sederhana yang menghitung Wasserstein Distance atau Kolmogorov-Smirnov test antara distribusi data latih dan data telemetri baru!

---

## 13. Daftar Pustaka Terverifikasi

1. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830. [https://scikit-learn.org/](https://scikit-learn.org/)
2. **Mitchell, M., et al.** (2019). *Model Cards for Model Reporting*. Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* '19), 220-229. [https://arxiv.org/abs/1810.03993](https://arxiv.org/abs/1810.03993)
3. **Sculley, D., et al.** (2015). *Hidden Technical Debt in Machine Learning Systems*. Advances in Neural Information Processing Systems (NeurIPS 2015). [https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)
4. **International Organization for Standardization.** (2018). *ISO 10816: Mechanical vibration - Evaluation of machine vibration by measurements on non-rotating parts*. ISO Standards Repository. [https://www.iso.org/standard/27641.html](https://www.iso.org/standard/27641.html)
5. **Géron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd Edition)*. O'Reilly Media. ISBN: 978-1098125967.

