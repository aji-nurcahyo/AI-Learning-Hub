# Chapter 20: Machine Learning Capstone — Code Catalog

Katalog skrip Python dan panduan eksekusi untuk proyek Capstone Level 2: **Sistem Pemeliharaan Prediktif Kompresor Industri**.

---

## 1. Ikhtisar Dataset

Dataset telemetri tersimpan di `data/predictive_maintenance_telemetry.csv` dan dibangkitkan secara deterministik oleh `generate_capstone_dataset.py`:
- **Ukuran**: 600 observasi telemetri kontinu lintas 12 unit mesin (`PUMP_01` hingga `PUMP_12`).
- **Fitur Fisik**:
  1. `vibration_rms_g`: Akselerasi getaran efektif RMS ($g$).
  2. `bearing_temperature_c`: Suhu bantalan kompresor ($^\circ\text{C}$).
  3. `rotational_speed_rpm`: Kecepatan putaran poros utama (RPM).
  4. `hydraulic_pressure_bar`: Tekanan pelumasan hidrolik ($\text{bar}$).
  5. `motor_current_amp`: Konsumsi arus listrik motor ($A$).
  6. `oil_acidity_tan`: Angka keasaman minyak pelumas ($\text{mg KOH/g}$).
- **Target**: `machine_failure` (0 = Operasi Normal, 1 = Indikasi Kerusakan Kritis). Distribusi: Normal 243 sampel (40.5%), Kerusakan 357 sampel (59.5%).

---

## 2. Struktur Skrip Python

| No | Berkas Skrip | Peran / Deskripsi | Luaran Utama Terverifikasi |
|---|---|---|---|
| 1 | `generate_capstone_dataset.py` | Generator data sintetik deterministik berbasis fisika mesin rotasi | CSV 600 baris, 9 kolom |
| 2 | `baseline_models.py` | Evaluasi Dummy Majority vs Simple Logistic Regression | Baseline LR F1 = 0.9655, ROC-AUC = 0.9914 |
| 3 | `preprocessing_pipeline.py` | Custom transformer rekayasa fitur fisik terintegrasi Scikit-Learn Pipeline | Menghasilkan 3 fitur rasio fisik domain |
| 4 | `candidate_models.py` | Benchmark 4 kandidat model Level 2 (5-Fold Stratified CV) | RF (93.50% F1), SVM (93.73% F1) |
| 5 | `cross_validation_tuning.py` | Optimasi hyperparameter Random Forest & Gradient Boosting | Champion RF F1 CV = 0.9390 |
| 6 | `final_test_evaluation.py` | Evaluasi akhir model terpilih pada untouched test set (120 sampel) | Akurasi 93.33%, Recall 95.77%, ROC-AUC 0.9859 |
| 7 | `error_analysis_diagnostics.py` | Diseksi False Negatives/Positives & optimasi threshold berbasis biaya | Threshold optimal 0.30 menekan risiko ke $6,200 |
| 8 | `explainability_analysis.py` | Permutation Importance & Partial Dependence Plot (PDP) | Suhu bearing & vibrasi sebagai faktor penentu utama |
| 9 | `responsible_ai_assessment.py` | Audit disparitas performa 12 unit pompa & batasan tata kelola etis | Disparitas F1 0.2000 (Lolos batas <0.30) |
| 10 | `capstone_pipeline_runner.py` | Runner orkestrator end-to-end yang mengeksekusi seluruh 8 tahapan | Exit Code 0 terverifikasi |

---

## 3. Cara Menjalankan

### Menjalankan Master Pipeline End-to-End:
```bash
python content/level-2-machine-learning/chapter-20/code/capstone_pipeline_runner.py
```

### Menjalankan Modul Individual:
```bash
python content/level-2-machine-learning/chapter-20/code/baseline_models.py
python content/level-2-machine-learning/chapter-20/code/final_test_evaluation.py
python content/level-2-machine-learning/chapter-20/code/error_analysis_diagnostics.py
```

