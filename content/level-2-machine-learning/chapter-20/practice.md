# Panduan Praktik Laboratorium: Machine Learning Capstone Project

Selamat datang di modul panduan praktik **Machine Learning Capstone Project** (Bab 20). Modul ini merupakan kulminasi akhir dari seluruh materi **Level 2: Machine Learning**, di mana Anda akan mengintegrasikan pemahaman teoretis dan keterampilan rekayasa kode dari Bab 11 hingga Bab 19 ke dalam sebuah proyek rekayasa pembelajaran mesin end-to-end yang memenuhi standar industri.

---

## 1. Ikhtisar & Ketentuan Proyek

Proyek Capstone menuntut eksekusi disiplin rekayasa pembelajaran mesin (*machine learning engineering*) yang ketat. Anda tidak hanya diminta untuk melatih model, melainkan membangun alur kerja yang modular, bebas kebocoran data (*data leakage-free*), teruji pada data terisolasi, dioptimalkan berdasarkan biaya bisnis riil, dan diaudit aspek keadilan serta tata kelolanya (*Responsible AI*).

### Batasan Teknis & Larangan:
1. **Fokus Level 2**: Seluruh solusi wajib diselesaikan menggunakan arsitektur pembelajaran mesin klasik/tradisional (Regresi Linier/Logistik, Support Vector Machines, K-Nearest Neighbors, Decision Trees, Random Forest, Gradient Boosting, dsb.).
2. **Larangan Deep Learning**: Tidak diperkenankan menggunakan Neural Networks multi-layer, PyTorch, TensorFlow/Keras, arsitektur CNN, YOLO, Transformer, atau model LLM.
3. **Pemisahan Terisolasi**: Pembagian data uji (minimal 20%) harus dilakukan di awal dan TIDAK PERNAH disentuh selama rekayasa fitur atau seleksi hyperparameter.
4. **Reproducibility**: Seluruh kode harus menggunakan random state tetap (`random_state=42`) dan dapat dieksekusi ulang dari awal hingga akhir dengan hasil yang konsisten.

---

## 2. Pilihan Jalur Topik Proyek Capstone

Peserta diberikan kebebasan memilih 1 dari 5 jalur proyek industri di bawah ini, atau dapat menggunakan Proyek Acuan (Jalur 1) yang disediakan secara lengkap dalam repositori:

```
+---------------------------------------------------------------------------------------------------------+
|                                    5 PILIHAN JALUR PROYEK CAPSTONE ML                                  |
+---------------------------------------------------------------------------------------------------------+
| [Jalur 1 - Acuan]  Sistem Pemeliharaan Prediktif Kompresor Sentrifugal Industri                         |
|                    Tipe: Klasifikasi Biner | Domain: IoT Pabrik / Reliability Engineering               |
+---------------------------------------------------------------------------------------------------------+
| [Jalur 2]          Deteksi Transaksi Keuangan Mencurigakan (Financial Fraud Detection)                  |
|                    Tipe: Klasifikasi Imbalanced | Domain: Fintech & Cyber Security                     |
+---------------------------------------------------------------------------------------------------------+
| [Jalur 3]          Prediksi Daya Keluaran Pembangkit Tenaga Surya (Solar Power Forecasting)             |
|                    Tipe: Regresi Kontinu | Domain: Energi Terbarukan & Smart Grid                       |
+---------------------------------------------------------------------------------------------------------+
| [Jalur 4]          Stratifikasi Risiko Kardiometabolik Pasien Rawat Jalan                               |
|                    Tipe: Klasifikasi Multi-kelas / Biner | Domain: Kesehatan Digital & Bioinformatika    |
+---------------------------------------------------------------------------------------------------------+
| [Jalur 5]          Prediksi Churn Pelanggan & Estimasi Nilai Seumur Hidup (Customer LTV)                |
|                    Tipe: Hibrida Klasifikasi & Regresi | Domain: E-Commerce & Retail Analytics          |
+---------------------------------------------------------------------------------------------------------+
```

---

## 3. Delapan Tahapan Baku Siklus Hidup Capstone

Setiap laporan dan repositori Capstone wajib menyajikan 8 tahapan terstruktur:

```
  [1. Formulasi & Baseline] ──> [2. Partisi & Isolasi Data] ──> [3. Rekayasa Fitur Domain]
                                                                            │
  [6. Evaluasi Test Set]    <── [5. Tuning Hyperparameter]  <── [4. Benchmark Kandidat Model]
             │
             └──> [7. Analisis Galat & Biaya] ──> [8. Responsible AI & Model Card]
```

---

## 4. Panduan Eksekusi Langkah-demi-Langkah (Proyek Acuan)

Proyek acuan yang diimplementasikan dalam repositori ini bertajuk:
**"Sistem Pemeliharaan Prediktif Kompresor Sentrifugal Industri Berbasis Telemetri Multi-Sensor"**

### Langkah 1: Eksplorasi Data & Formulasi Masalah
Pahami parameter fisik yang direkam oleh sensor telemetri:
- Berkas data: `content/level-2-machine-learning/chapter-20/code/data/predictive_maintenance_telemetry.csv`
- Total sampel: 600 data lintas 12 unit mesin kompresor (`PUMP_01` hingga `PUMP_12`).
- Jalankan skrip pembangkit:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/generate_capstone_dataset.py
  ```

### Langkah 2: Membangun Model Acuan Awal (Baseline)
Sebelum membuat model kompleks, tetapkan patokan performa minimum:
- Dummy Classifier (tebak mayoritas): Akurasi 59.17%, ROC-AUC 0.5000.
- Simple Logistic Regression: Akurasi 95.83%, F1-Score 0.9655, ROC-AUC 0.9914.
- Jalankan evaluasi baseline:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/baseline_models.py
  ```

### Langkah 3: Rekayasa Fitur Fisik Bebas Kebocoran
Implementasikan custom transformer `IndustrialFeatureEngineer` di dalam `Scikit-Learn Pipeline`:
1. *Thermal-Pressure Ratio*: $\text{Suhu} / \text{Tekanan}$ (penurunan viskositas pelumas).
2. *Vibration Power Index*: $(\text{Vibrasi})^2 \times (\text{RPM} / 1000)$ (energi destruktif bantalan).
3. *Lubricant Stress Index*: $\text{Arus Motor} \times \text{Keasaman Minyak}$ (beban gesekan mekanik).
- Uji pipeline:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/preprocessing_pipeline.py
  ```

### Langkah 4: Benchmark 4 Arsitektur Kandidat Model Level 2
Uji performa 4 paradigma algoritma pada data latih dengan 5-Fold Stratified CV:
- Logistic Regression (L2)
- Support Vector Machine (RBF Kernel)
- Random Forest Classifier (Bagging)
- Gradient Boosting Classifier (Boosting)
- Jalankan komparasi:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/candidate_models.py
  ```

### Langkah 5: Penyetelan Hyperparameter Sistematis
Lakukan GridSearchCV untuk mencari kedalaman pohon dan jumlah estimator optimal pada Random Forest dan Gradient Boosting:
- Jalankan tuning:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/cross_validation_tuning.py
  ```
- *Hasil Terverifikasi*: Konfigurasi terbaik Random Forest (`max_depth=8`, `min_samples_split=10`, `n_estimators=150`) mencatatkan F1-Score CV sebesar **0.9390**.

### Langkah 6: Evaluasi Akhir pada Untouched Test Set
Evaluasi performa model terpilih pada 120 sampel uji yang diisolasi sejak awal:
- Jalankan evaluasi akhir:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/final_test_evaluation.py
  ```
- *Metrik Terverifikasi*:
  - Akurasi: **93.33%**
  - Presisi: **93.15%**
  - Recall: **95.77%** (68 dari 71 kegagalan mesin berhasil diantisipasi)
  - F1-Score: **0.9444**
  - ROC-AUC: **0.9859**

### Langkah 7: Analisis Galat & Optimasi Threshold Biaya Operasional
Analisis sampel False Negative dan False Positive serta integrasikan matriks biaya industri:
- Biaya False Negative (Downtime pabrik tak terduga): **$5,000**
- Biaya False Positive (Inspeksi teknisi tanpa kerusakan): **$200**
- Jalankan diagnostik galat:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/error_analysis_diagnostics.py
  ```
- *Temuan*: Menggeser ambang batas dari 0.50 ke **0.30** menurunkan risiko finansial dari $16,000 menjadi **$6,200** (penghematan $9,800 atau 61.25%).

### Langkah 8: Transparansi Penalaran & Tata Kelola Model (Explainability & Governance)
Audit feature importance, parsial dependence plot, dan keadilan performa antar unit mesin:
- Jalankan analisis keterpahaman:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/explainability_analysis.py
  ```
- Jalankan audit tata kelola Responsible AI:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/responsible_ai_assessment.py
  ```
- Jalankan seluruh pipeline secara terpadu:
  ```bash
  python content/level-2-machine-learning/chapter-20/code/capstone_pipeline_runner.py
  ```

---

## 5. Rubrik Penilaian Capstone (Total: 100 Poin)

| Komponen | Bobot | Kriteria Skor Sempurna (A) |
|---|:---:|---|
| **1. Formulasi Masalah & Baseline** | 15% | Spesifikasi teknis terdefinisi tajam, metrik keberhasilan relevan dengan domain, baseline acuan (dummy & simple model) dievaluasi secara terukur. |
| **2. Prapemrosesan Bebas Bocor & Fitur** | 20% | Pemisahan test set terisolasi secara sempurna, seluruh tahapan transformasi dibungkus rapi dalam Scikit-Learn Pipeline, minimal 3 fitur turunan domain berbasis fisika/logika bisnis. |
| **3. Pemodelan & Tuning Hyperparameter** | 25% | Menguji minimal 3 keluarga algoritma Level 2 yang berbeda, strategi validasi silang tepat (Stratified/Group CV), penyetelan hyperparameter sistematis via Grid/RandomizedSearchCV. |
| **4. Evaluasi Akhir & Analisis Galat** | 20% | Evaluasi komprehensif pada test set terisolasi, diseksi matriks konfusi, analisis mendalam pada kasus False Negative/Positive, dan simulasi penyesuaian threshold berbasis biaya. |
| **5. Responsible AI, Model Card & Kode** | 20% | Menghitung feature importance dan PDP, audit disparitas performa subkelompok, menyusun Model Card dengan batas operasional aman, serta kode modular dan reproducible 100%. |

---

## 6. Daftar Periksa Kesiapan Penyerahan (Submission Checklist)

Sebelum menyerahkan laporan Capstone Anda, pastikan seluruh item berikut telah terpenuhi:
- [ ] Kode sumber dapat dijalankan dengan sekali perintah tanpa galat (`python capstone_pipeline_runner.py`).
- [ ] Menggunakan data latih dan data uji yang terpisah secara terisolasi tanpa ada *data leakage*.
- [ ] Seluruh skrip menggunakan `random_state=42` untuk menjamin reproduktibilitas.
- [ ] Menyertakan notebook pendamping (`chapter_20_companion.ipynb`) yang telah dieksekusi secara penuh.
- [ ] Laporan akhir menyajikan Model Card dengan spesifikasi batasan penggunaan etis.

