# Katalog Kode Praktik Chapter 13: Supervised Learning — Regression Algorithms

Katalog ini memuat implementasi skrip Python modular untuk eksplorasi, pemodelan, regularisasi, dan evaluasi diagnostik algoritma regresi (Supervised Learning) menggunakan ekosistem NumPy, Pandas, dan Scikit-learn.

---

## 1. Lingkungan Eksekusi & Prasyarat

Seluruh skrip diuji dan diverifikasi menggunakan Python 3.12 dengan pustaka standar:
- `numpy >= 1.26.0`
- `pandas >= 2.2.0`
- `scikit-learn >= 1.4.0`

Dataset telemetri konsumsi energi industri (`equipment_power_demand.csv`) dibangkitkan secara deterministik menggunakan skrip `generate_dataset.py` dan disimpan di direktori `data/`.

---

## 2. Struktur Berkas & Ringkasan Skrip

| No | Berkas Skrip | Fokus Materi & Metodologi | Status Eksekusi |
|---|---|---|---|
| 00 | `generate_dataset.py` | Pembangkitan dataset telemetri daya (300 baris, 6 prediktor fisik, target kontinu `power_consumption_kw`). | Terverifikasi (Exit 0) |
| 01 | `simple_linear_regression.py` | Regresi linier sederhana univariat ($y = wx + b$), estimasi OLS, ekstraksi koefisien, dan galat residual. | Terverifikasi (Exit 0) |
| 02 | `multiple_linear_regression.py` | Regresi linier multivariat 6 fitur operasional, interpretasi efek marginal, komparasi vs Mean Dummy Regressor. | Terverifikasi (Exit 0) |
| 03 | `polynomial_regression.py` | Penanganan kelengkungan non-linear via `PolynomialFeatures` (Derajat 1, 2, dan 4) terisolasi dalam Pipeline. | Terverifikasi (Exit 0) |
| 04 | `ridge_regression.py` | Regularisasi L2 (Ridge), efek penalti $\alpha$ terhadap penyusutan norma $\|w\|_2$, dan keharusan penskalaan fitur. | Terverifikasi (Exit 0) |
| 05 | `lasso_regression.py` | Regularisasi L1 (Lasso), pembuktian sifat kelangkaan (*sparsity* di mana $w=0$), dan seleksi fitur terintegrasi. | Terverifikasi (Exit 0) |
| 06 | `elasticnet_regression.py` | Kombinasi penalti konveks L1 dan L2, analisis kestabilan multikolinearitas terhadap parameter `l1_ratio`. | Terverifikasi (Exit 0) |
| 07 | `regression_metrics.py` | Perhitungan matematis manual vs API untuk MAE, MSE, RMSE, $R^2$, dan pembuktian $R^2 < 0$ pada data uji. | Terverifikasi (Exit 0) |
| 08 | `residual_analysis.py` | Diagnostik residual: mean galat mendekati nol, uji homoskedastisitas tertil, normalitas skewness, dan deteksi outlier. | Terverifikasi (Exit 0) |
| 09 | `regression_model_comparison.py` | Benchmark 6 model regresi dengan 5-Fold Cross-Validation dan evaluasi akhir pada *unseen test data*. | Terverifikasi (Exit 0) |

---

## 3. Hasil Pengujian Numerik Terverifikasi

Seluruh nilai di bawah ini dihasilkan secara aktual dari eksekusi skrip:

```text
1. SIMPLE LINEAR REGRESSION (simple_linear_regression.py):
   - Slope (w)        : 3.6295 kW / % beban
   - Intercept (b)    : 98.4008 kW
   - Test RMSE        : 8.07 kW
   - Test R^2         : 0.9888

2. MULTIPLE LINEAR REGRESSION (multiple_linear_regression.py):
   - Dummy Baseline RMSE : 76.65 kW (R^2 = -0.0054)
   - MLR Test RMSE       :  5.37 kW (R^2 =  0.9951)
   - Penurunan Galat     : 92.99% penurunan RMSE

3. POLYNOMIAL REGRESSION TRADEOFF (polynomial_regression.py):
   - Derajat 1 (Linier) : 3 fitur  | Train RMSE: 6.90 kW | Test RMSE: 6.83 kW | R^2: 0.9920
   - Derajat 2 (Kuadrat): 9 fitur  | Train RMSE: 5.87 kW | Test RMSE: 5.62 kW | R^2: 0.9946 (Optimal)
   - Derajat 4 (Tinggi) : 34 fitur | Train RMSE: 5.66 kW | Test RMSE: 5.92 kW | R^2: 0.9940 (Mulai Overfit)

4. REGULARISASI COEFFICIENT SHRINKAGE:
   - Ridge (alpha=0.001 -> 500) : ||w||_2 menyusut dari 59.87 ke 27.28 (koefisien tidak pernah 0).
   - Lasso (alpha=0.05 -> 8.0)  : Fitur aktif terpangkas dari 6 -> 5 -> 3 -> 1 (sparsity terbukti).

5. RESIDUAL DIAGNOSTICS (residual_analysis.py):
   - Mean Residual       : -0.5619 kW (Mendekati 0.0000)
   - Skewness Residual   : -0.2049 (Simetris mendekati normal)
   - Varians per Segmen  : Rendah=4.16, Sedang=4.50, Tinggi=5.67 (Homoskedastik terpenuhi)

6. BENCHMARK AKHIR (regression_model_comparison.py):
   - Polynomial Reg (Deg 2) : CV RMSE = 5.47 (+/- 0.61) kW | Test RMSE = 4.12 kW | Test R^2 = 0.9971
   - Linear Regression      : CV RMSE = 6.17 (+/- 0.90) kW | Test RMSE = 5.37 kW | Test R^2 = 0.9951
   - Lasso Regression (L1)  : CV RMSE = 6.22 (+/- 1.10) kW | Test RMSE = 5.51 kW | Test R^2 = 0.9948
```

---

## 4. Cara Menjalankan

Dari direktori root proyek:
```bash
# Mengaktifkan virtual environment
.\.venv\Scripts\Activate.ps1

# Menjalankan seluruh rangkaian skrip
python content/level-2-machine-learning/chapter-13/code/simple_linear_regression.py
python content/level-2-machine-learning/chapter-13/code/polynomial_regression.py
python content/level-2-machine-learning/chapter-13/code/regression_model_comparison.py
```

