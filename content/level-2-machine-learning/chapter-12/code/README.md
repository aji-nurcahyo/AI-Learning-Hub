# Katalog Kode Praktik Chapter 12: ML Workflow, Dataset & Feature Engineering

Katalog ini berisi skrip Python modular yang mengimplementasikan seluruh tahapan pra-pemrosesan data, pembagian dataset, rekayasa fitur (feature engineering), mitigasi ketidakseimbangan kelas, validasi silang (cross-validation), dan pencegahan kebocoran data (*data leakage*) menggunakan ekosistem Scikit-learn dan Pandas.

---

## 1. Lingkungan Eksekusi & Prasyarat

Seluruh skrip diuji dan diverifikasi menggunakan Python 3.12 dengan pustaka standar data science:
- `numpy >= 1.26.0`
- `pandas >= 2.2.0`
- `scikit-learn >= 1.4.0`

Dataset telemetri industri sintetis dibangkitkan secara otomatis oleh skrip `generate_dataset.py` dan disimpan di direktori `data/equipment_telemetry.csv`.

---

## 2. Struktur Berkas & Ringkasan Skrip

| No | Berkas Skrip | Fokus Materi & Metodologi | Status Eksekusi |
|---|---|---|---|
| 00 | `generate_dataset.py` | Pembangkitan dataset telemetri industri (250 baris, 8 atribut, target `failure` berimbang ~10.8%, missing values terkontrol). | Terverifikasi (Exit 0) |
| 01 | `dataset_splitting.py` | Pembagian dataset 3-way (Train 60%, Validation 20%, Test 20%) dengan verifikasi integritas indeks (nol tumpang tindih). | Terverifikasi (Exit 0) |
| 02 | `stratified_split.py` | Perbandingan Random Split vs Stratified Split pada dataset imbalanced; stabilisasi proporsi kelas minoritas lintas fold. | Terverifikasi (Exit 0) |
| 03 | `group_split.py` | Pencegahan kebocoran entitas (*group leakage*) menggunakan `GroupShuffleSplit` dan `GroupKFold` berdasarkan `machine_id`. | Terverifikasi (Exit 0) |
| 04 | `time_series_split.py` | Pencegahan bias masa depan (*look-ahead bias*) dengan `TimeSeriesSplit` (expanding window) pada data sekuensial temporal. | Terverifikasi (Exit 0) |
| 05 | `numeric_preprocessing.py` | Imputasi nilai hilang median dan penskalaan standar (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) dengan aturan fit *hanya* pada Train. | Terverifikasi (Exit 0) |
| 06 | `categorical_preprocessing.py` | `OneHotEncoder(handle_unknown='ignore')` dan `OrdinalEncoder`, diuji ketahanannya terhadap kemunculan kategori baru (*unseen categories*). | Terverifikasi (Exit 0) |
| 07 | `feature_engineering.py` | Pembuatan fitur domain spesifik (rasio getaran-tekanan, laju suhu/jam) dan suku interaksi non-linear (`thermal_mechanical_index`). | Terverifikasi (Exit 0) |
| 08 | `class_imbalance_demo.py` | Penanganan ketidakseimbangan kelas secara metodologis: `class_weight='balanced'` vs penyesuaian ambang batas (*threshold tuning*). | Terverifikasi (Exit 0) |
| 09 | `cross_validation.py` | Evaluasi generalisasi menggunakan `KFold` dan `StratifiedKFold` lintas metrik F1, ROC-AUC, dan Balanced Accuracy. | Terverifikasi (Exit 0) |
| 10 | `sklearn_pipeline.py` | Pembangunan end-to-end `Pipeline` Scikit-learn yang menyatukan imputasi, skalasi, dan estimator tanpa resiko kebocoran data. | Terverifikasi (Exit 0) |
| 11 | `column_transformer.py` | Pra-pemrosesan data heterogen (numerik, nominal, ordinal) secara simultan menggunakan `ColumnTransformer`. | Terverifikasi (Exit 0) |
| 12 | `leakage_prevention_demo.py` | Eksperimen komparatif pembuktian ilusi performa akibat *data leakage* (akurasi palsu 73.33% vs akurasi realistis 58.33% pada noise). | Terverifikasi (Exit 0) |

---

## 3. Hasil Pengujian Numerik Terverifikasi

Seluruh nilai di bawah ini dihasilkan secara aktual dari eksekusi skrip:

```text
1. DATASET SPLIT INTEGRITY (dataset_splitting.py):
   - Training Set   : 150 sampel (60.0%) | 18 positif (12.00%)
   - Validation Set :  50 sampel (20.0%) |  6 positif (12.00%)
   - Test Set       :  50 sampel (20.0%) |  3 positif ( 6.00%)
   - Overlap Index  : 0 sampel (100% disjoint)

2. STRATIFIED VS RANDOM (stratified_split.py):
   - Standar Deviasi Proporsi Positif K-Fold Standar : 4.1183%
   - Standar Deviasi Proporsi Positif StratifiedKF   : 0.9798%

3. GROUP LEAKAGE MITIGATION (group_split.py):
   - Naive Split Overlap Mesin  : 5 mesin bocor ke test set
   - GroupSplit Overlap Mesin   : 0 mesin bocor (eksklusif)

4. TIME-SERIES INTEGRITY (time_series_split.py):
   - Random Split Lookahead Leak : 100% sampel test terjadi sebelum batas akhir train
   - TimeSeriesSplit Lookahead    : 0% pelanggaran urutan waktu

5. HETEROGENEOUS PIPELINE (column_transformer.py):
   - F1-Score (Failure) : 0.6667
   - Balanced Accuracy  : 0.9444
   - Recall Failure     : 1.0000

6. LEAKAGE IMPACT ON PURE NOISE (leakage_prevention_demo.py):
   - Leaky Preprocessing Test Accuracy : 73.33% (Ilusi Performa Palsu)
   - Clean In-Pipeline Test Accuracy   : 58.33% (Estimasi Realistis)
```

---

## 4. Cara Menjalankan

Dari direktori root proyek:
```bash
# Mengaktifkan virtual environment
.\.venv\Scripts\Activate.ps1

# Menjalankan seluruh rangkaian skrip
python content/level-2-machine-learning/chapter-12/code/dataset_splitting.py
python content/level-2-machine-learning/chapter-12/code/column_transformer.py
python content/level-2-machine-learning/chapter-12/code/leakage_prevention_demo.py
```

