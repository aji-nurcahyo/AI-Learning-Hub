# Katalog Praktikum & Eksperimen Kode — Chapter 15: Decision Trees & Ensemble Learning

Direktori ini memuat seluruh implementasi kode Python mandiri (*runnable*), skrip pembangkit data deterministik, visualisasi struktur pohon, teknik pemangkasan (*pruning*), serta komparasi model *ensemble* (Bagging, Random Forest, AdaBoost, Gradient Boosting) untuk klasifikasi dan regresi sensor industri.

---

## 1. Lingkungan Eksekusi & Kebutuhan Dependensi

Seluruh skrip dikembangkan dan diverifikasi menggunakan Python 3.12 pada lingkungan virtual `.venv`:

```bash
# Aktifkan virtual environment
.\.venv\Scripts\activate

# Dependensi utama (terinstal):
# - numpy >= 1.26.0
# - pandas >= 2.2.0
# - scikit-learn >= 1.4.0
# - matplotlib >= 3.8.0
```

---

## 2. Struktur Berkas & Deskripsi Modul

| Berkas Skrip | Deskripsi Algoritmik & Pembelajaran | Keluaran Utama / Metrik |
|---|---|---|
| `generate_dataset.py` | Pembangkit data sintetis telemetri sensor industri deterministik (360 baris) | `data/equipment_fault_sensor_data.csv` (NORMAL, WARNING, FAULT) |
| `decision_tree_classifier.py` | DecisionTreeClassifier dengan kriteria Gini vs Entropy, evaluasi data latih/uji | Akurasi Uji Gini vs Entropy, Confusion Matrix 3x3 |
| `decision_tree_regressor.py` | DecisionTreeRegressor untuk prediksi nilai kontinu suhu berbasis penurunan varians | Piecewise constant, evaluasi RMSE dan $R^2$ lintas kedalaman |
| `entropy_gini_information_gain.py` | Perhitungan manual Shannon Entropy, Gini Impurity, dan Information Gain | Validasi identik 100% dengan internal Scikit-learn tree impurity |
| `tree_visualization.py` | Ekspor aturan IF-THEN tekstual dan representasi diagram alur pohon teratas | Proposisi aturan pengambilan keputusan yang terverifikasi |
| `tree_pruning.py` | Analisis Pre-Pruning (`max_depth`, `min_samples_leaf`) & Minimal Cost-Complexity Post-Pruning (`ccp_alpha`) | Penentuan $\alpha_{\text{optimal}}$ untuk mereduksi overfitting |
| `bagging_classifier.py` | Bootstrap Aggregating (Bagging) berbasis agregasi mayoritas voting | Reduksi varians signifikan dibandingkan single unpruned tree |
| `random_forest_classifier.py` | Random Forest dengan random feature subsampling ($\sqrt{p}$) | Analisis MDI (Mean Decrease in Impurity) Feature Importance |
| `out_of_bag_evaluation.py` | Estimasi generalisasi Out-of-Bag (OOB) sampel sisa bootstrap ($\approx 36.8\%$) | Komparasi OOB Score vs 5-Fold Stratified CV vs Test Set |
| `adaboost_classifier.py` | Adaptive Boosting dengan weak learners (decision stumps, kedalaman 1) | Analisis pengaruh learning rate dan penambahan estimator |
| `gradient_boosting.py` | Gradient Boosting dengan optimasi loss pseudo-residuals bertahap | Analisis konvergensi staged log-loss (iterasi 1 hingga 100) |
| `ensemble_model_comparison.py` | Komparasi komprehensif 7 arsitektur model pada 5-Fold Stratified CV | Tabel komparasi akurasi, F1 Macro, dan latensi inferensi |

---

## 3. Dataset Telemetri Sensor Kerusakan Peralatan

Dataset disimpan pada `data/equipment_fault_sensor_data.csv` dengan metadata:
- **Jumlah Baris**: 360 sampel telemetri.
- **Fitur Masukan (6 Prediktor)**:
  1. `temperature_c`: Suhu operasional mesin dalam derajat Celsius (°C).
  2. `vibration_mm_s`: Kecepatan vibrasi RMS (Root Mean Square) dalam mm/s.
  3. `voltage_v`: Tegangan catu daya fase-3 dalam Volt (V).
  4. `current_a`: Arus listrik operasional dalam Ampere (A).
  5. `operating_hours`: Jam kerja kumulatif mesin (100 - 8000 jam).
  6. `load_percentage`: Persentase kapasitas beban operasi (20% - 100%).
- **Target Klasifikasi**:
  - `fault_category`: Kategori diskret (`NORMAL`: 91 sampel, `WARNING`: 120 sampel, `FAULT`: 149 sampel).
  - `fault_code`: Representasi numerik terenkode (`0`, `1`, `2`).

---

## 4. Cara Menjalankan Skrip

Jalankan skrip secara individual melalui terminal:

```bash
# Menjalankan pembanding model ensemble
python content/level-2-machine-learning/chapter-15/code/ensemble_model_comparison.py

# Menjalankan verifikasi perhitungan matematika Entropy & Gini
python content/level-2-machine-learning/chapter-15/code/entropy_gini_information_gain.py

# Menjalankan eksperimen post-pruning
python content/level-2-machine-learning/chapter-15/code/tree_pruning.py
```

