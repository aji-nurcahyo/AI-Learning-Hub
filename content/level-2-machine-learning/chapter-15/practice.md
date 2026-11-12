# Panduan Praktikum Laboratorium — Chapter 15: Decision Trees & Ensemble Learning

Panduan laboratorium terpadu ini menyajikan petunjuk langkah demi langkah (*step-by-step*) untuk mengeksplorasi, melatih, memangkas (*pruning*), dan mengagregasi model pohon keputusan (*ensemble*) menggunakan Python dan Scikit-learn pada data telemetri sensor industri.

---

## 1. Sasaran Pembelajaran Praktikum

Setelah menyelesaikan sesi laboratorium ini, mahasiswa diharapkan mampu:
1. Menghitung dan memvalidasi nilai **Entropy**, **Gini Impurity**, dan **Information Gain** secara manual serta mencocokkannya dengan API Scikit-learn.
2. Melatih **DecisionTreeClassifier** dan **DecisionTreeRegressor**, serta mengekstrak representasi aturan logika **IF-THEN**.
3. Menganalisis trade-off bias-varians melalui teknik **Pre-Pruning** dan **Minimal Cost-Complexity Post-Pruning** (`ccp_alpha`).
4. Mengimplementasikan arsitektur **Bagging** dan **Random Forest** dengan evaluasi **Out-of-Bag (OOB)**.
5. Menerapkan teknik sekuensial **Boosting** (**AdaBoost** dan **Gradient Boosting**) serta menganalisis penurunan fungsi *loss*.
6. Membandingkan performa seluruh model ensemble secara objektif melalui **5-Fold Stratified Cross-Validation**.

---

## 2. Persiapan Lingkungan & Struktur Proyek

Pastikan Anda berada di direktori akar repositori dan mengaktifkan virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\activate

# Pindah ke direktori kode Chapter 15
cd content/level-2-machine-learning/chapter-15/code
```

Verifikasi ketersediaan pustaka pendukung:
```python
import numpy as np
import pandas as pd
import sklearn
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__} | Scikit-learn: {sklearn.__version__}")
```

---

## 3. Langkah-Langkah Praktikum Terpandu

### Langkah 1: Pembangkitan Data Telemetri Sensor (`generate_dataset.py`)
Jalankan skrip pembangkit data untuk membuat 360 sampel telemetri deterministik:
```bash
python generate_dataset.py
```
- **Target**: `fault_category` (`NORMAL`, `WARNING`, `FAULT`).
- **Fitur**: `temperature_c`, `vibration_mm_s`, `voltage_v`, `current_a`, `operating_hours`, `load_percentage`.

### Langkah 2: Verifikasi Manual Impurity Node (`entropy_gini_information_gain.py`)
Pahami bagaimana algoritma pohon memilih titik pemisahan (*splitting point*) terbaik:
```bash
python entropy_gini_information_gain.py
```
*Hasil Verifikasi:*
- Shannon Entropy Node Induk (4 Normal, 6 Fault): $H(S) = 0.9710$ bits.
- Gini Impurity Node Induk: $Gini(S) = 0.4800$.
- Information Gain kandidat split: $0.6710$ bits.

### Langkah 3: Pelatihan Baseline Decision Tree Classifier (`decision_tree_classifier.py`)
Bandingkan kriteria pemisahan Gini Impurity versus Entropy:
```bash
python decision_tree_classifier.py
```
*Hasil Verifikasi Aktual Data Uji:*
- Akurasi Gini Impurity (`max_depth=4`): **92.22%** (Macro F1: 0.9260).
- Akurasi Entropy (`max_depth=4`): **90.00%**.

### Langkah 4: Visualisasi Struktur Pohon & Aturan Keputusan (`tree_visualization.py`)
Inspeksi aturan IF-THEN yang terbentuk pada pohon keputusan:
```bash
python tree_visualization.py
```
Perhatikan bagaimana fitur `vibration_mm_s` dan `temperature_c` terpilih sebagai pemisah utama di node teratas (*root node*).

### Langkah 5: Eksperimen Pencegahan Overfitting via Pruning (`tree_pruning.py`)
Amati bahaya pohon tanpa batas (*unpruned tree*) yang menghafal data latih:
```bash
python tree_pruning.py
```
*Temuan Utama:*
- Pohon tanpa batas mengalami overfitting: Akurasi Latih $100\%$, Akurasi Uji $90.00\%$ (gap $10.00\%$).
- Post-pruning dengan $\alpha_{\text{optimal}} \approx 0.008$ memangkas jumlah node dari 45 menjadi 15 tanpa mengorbankan akurasi uji.

### Langkah 6: Mengurangi Varians dengan Bagging (`bagging_classifier.py`)
Amati bagaimana agregasi voting dari 50 pohon varians-tinggi meningkatkan stabilitas:
```bash
python bagging_classifier.py
```
*Hasil Uji:* Akurasi meningkat dari $90.00\%$ (single tree) menjadi **91.11%** pada 50 estimator.

### Langkah 7: Random Forest & Evaluasi Out-of-Bag (`random_forest_classifier.py` & `out_of_bag_evaluation.py`)
Gunakan *feature subsampling* ($\sqrt{p} = \sqrt{6} \approx 2$ fitur per split) dan evaluasi OOB:
```bash
python random_forest_classifier.py
python out_of_bag_evaluation.py
```
*Hasil Verifikasi:*
- Skor OOB ($93.70\%$) sangat selaras dengan rata-rata 5-Fold Stratified Cross-Validation ($94.81\%$).
- MDI Feature Importance mengidentifikasi `vibration_mm_s` ($35.4\%$) dan `temperature_c` ($28.1\%$) sebagai prediktor paling dominan.

### Langkah 8: Boosting Sekuensial (`adaboost_classifier.py` & `gradient_boosting.py`)
Pelajari bagaimana model ensemble memperbaiki kesalahan sampel secara bertahap:
```bash
python adaboost_classifier.py
python gradient_boosting.py
```
*Hasil Uji:*
- Gradient Boosting mencapai akurasi uji tertinggi sebesar **93.33%** dengan penurunan log-loss bertahap dari iterasi 1 ($0.98$) ke iterasi 100 ($0.12$).

### Langkah 9: Komparasi Komprehensif Seluruh Model (`ensemble_model_comparison.py`)
Bandingkan seluruh model di bawah protokol evaluasi 5-Fold Stratified Cross-Validation yang adil:
```bash
python ensemble_model_comparison.py
```

---

## 4. Tabel Hasil Pengujian Numerik Terverifikasi

| Arsitektur Model | CV Accuracy (Mean ± Std) | CV Macro F1 | Test Accuracy | Fit Time (ms) | Karakteristik Utama |
|---|---|---|---|---|---|
| **Dummy (Most Frequent)** | $41.48\% \pm 0.91\%$ | $19.54\%$ | $41.11\%$ | $0.6$ ms | Baseline naif, tidak belajar |
| **Single Tree (Unpruned)** | $92.59\% \pm 4.22\%$ | $92.90\%$ | $90.00\%$ | $3.4$ ms | Rentan overfitting, varians tinggi |
| **Single Tree (Pruned max_d=4)** | $91.48\% \pm 4.32\%$ | $91.20\%$ | $90.00\%$ | $2.6$ ms | Sangat terinterpretasi, sub-milidetik |
| **Bagging (50 Trees)** | $95.19\% \pm 5.44\%$ | $95.32\%$ | $91.11\%$ | $75.7$ ms | Reduksi varians efektif via bootstrap |
| **Random Forest (100 Trees)** | $94.81\% \pm 5.42\%$ | $94.78\%$ | $90.00\%$ | $98.9$ ms | De-korelasi pohon via feature subsampling |
| **AdaBoost (100 Stumps)** | $89.26\% \pm 10.44\%$ | $89.41\%$ | $81.11\%$ | $129.1$ ms | Sensitif terhadap noise & data tumpang tindih |
| **Gradient Boosting (100 Trees)** | **$96.30\% \pm 3.10\%$** | **$96.34\%$** | **$93.33\%$** | $223.2$ ms | **Model Terbaik**: Presisi tinggi lintas kelas |

---

## 5. Tugas Eksplorasi Mandiri & Tantangan Analitis

1. **Tantangan 1: Pengaruh Kardinalitas pada MDI Feature Importance**  
   Tambahkan kolom fitur acak murni berkardinalitas tinggi (`random_id = np.arange(len(df))`). Latih Random Forest dan amati skor kepentingan fitur acak tersebut. Mengapa MDI memberikan bobot tinggi pada fitur yang sama sekali tidak memiliki hubungan fisik dengan target?
2. **Tantangan 2: Pengaruh Learning Rate pada Gradient Boosting**  
   Ubah parameter `learning_rate` pada Gradient Boosting menjadi $0.01$ (sangat kecil) dan $1.5$ (sangat besar). Amati efek underfitting (memerlukan ribuan estimator) versus osilasi divergen.
3. **Tantangan 3: Regresi Pohon Keputusan (`decision_tree_regressor.py`)**  
   Plot kurva prediksi kontinu terhadap fitur `operating_hours`. Jelaskan secara geometris mengapa DecisionTreeRegressor tidak dapat mengekstrapolasi tren linier ke masa depan di luar rentang nilai latihan!

---

## 6. Rubrik Penilaian Praktikum Laboratorium (Total 100 Poin)

| Kriteria Evaluasi | Bobot | Bukti Lengkap (Full Credit) | Bukti Sebagian (Partial Credit) | Kurang Bukti (Insufficient) |
|---|---|---|---|---|
| **Eksekusi & Integritas Lingkungan** | **20** | Seluruh 11 skrip dan notebook tereksekusi tanpa galat, output numerik tercetak lengkap. | Sebagian besar skrip berjalan, terdapat galat minor pada parameter atau path. | Banyak skrip gagal dieksekusi atau lingkungan tidak terisolasi. |
| **Kalkulasi Matematis Impurity** | **20** | Perhitungan manual Entropy, Gini, dan IG dituliskan langkah per langkah dan cocok 100% dengan API. | Perhitungan manual ada namun terdapat salah hitung pada pembobotan cabang. | Tidak ada perhitungan manual, hanya menyalin output Scikit-learn. |
| **Analisis Pruning & Kompleksitas** | **20** | Membandingkan kurva overfitting unpruned tree vs pre-pruning vs post-pruning `ccp_alpha` secara mendalam. | Menerapkan pruning sederhana (`max_depth`) tanpa analisis `ccp_alpha`. | Tidak melakukan analisis pemangkasan pohon sama sekali. |
| **Komparasi Model Ensemble** | **25** | Melakukan validasi silang 5-fold komparatif (Single Tree, Bagging, RF, AdaBoost, Gradient Boosting) secara adil. | Membandingkan 2 model saja tanpa cross-validation terstruktur. | Hanya melatih satu model tanpa komparasi performa. |
| **Dokumentasi & Laporan Analisis** | **15** | Penjelasan fenomena teknis ditulis rapi, menyertakan trade-off latensi vs akurasi serta batasan data. | Laporan singkat, analisis kurang tajam terhadap metrik evaluasi. | Tidak menyertakan analisis, hanya kumpulan kode tanpa pembahasan. |

