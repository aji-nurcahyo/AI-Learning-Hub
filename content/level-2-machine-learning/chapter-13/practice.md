# Panduan Praktik Laboratorium — Chapter 13: Supervised Learning — Regression Algorithms

Selamat datang di laboratorium **Chapter 13: Supervised Learning — Regression Algorithms**. Praktikum ini memandu Anda menguasai formulasi matematis, implementasi komputasi, regularisasi, dan evaluasi diagnostik model regresi untuk memprediksi variabel kontinu pada sistem fisik dunia nyata.

---

## 1. Tujuan Praktik

Setelah menyelesaikan seluruh rangkaian praktikum laboratorium mandiri ini, Anda diharapkan mampu:
1. Mengimplementasikan Simple Linear Regression menggunakan algoritma Ordinary Least Squares (OLS) dan mengekstrak koefisien $w$ serta intersep $b$.
2. Membangun Multiple Linear Regression dengan 6 prediktor fisik dan membandingkan performanya terhadap *Mean Dummy Regressor*.
3. Menerapkan `PolynomialFeatures` dalam pipeline untuk memodelkan kelengkungan termal non-linear tanpa merusak sifat linieritas parameter model.
4. Menerapkan Regularisasi L2 (Ridge) untuk mengatasi multikolinearitas dan mengendalikan varians koefisien via penalti $\alpha$.
5. Menerapkan Regularisasi L1 (Lasso) untuk menciptakan model yang renggang (*sparse model*) dengan seleksi fitur otomatis ($w = 0$).
6. Mengonfigurasi ElasticNet untuk menyeimbangkan penalti L1 dan L2 secara simultan menggunakan parameter `l1_ratio`.
7. Menghitung dan memvalidasi metrik evaluasi MAE, MSE, RMSE, dan $R^2$ secara manual serta membuktikan fenomena $R^2 < 0$.
8. Melakukan analisis diagnostik residual (uji nilai harapan nol, homoskedastisitas lintas segmen prediksi, dan normalitas distribusi galat).
9. Melakukan benchmark komparatif multi-model menggunakan 5-Fold Cross-Validation dan pengujian final pada *unseen test data*.

---

## 2. Struktur Direktori Praktikum

Seluruh kode sumber mandiri, dataset telemetri, dan notebook interaktif tersimpan dalam struktur berikut:

```text
content/level-2-machine-learning/chapter-13/
├── code/
│   ├── data/
│   │   └── equipment_power_demand.csv # Dataset telemetri konsumsi daya (300 baris, 6 fitur)
│   ├── generate_dataset.py            # Skrip pembangkit dataset sintetis deterministik
│   ├── simple_linear_regression.py    # Regresi linier sederhana univariat & OLS
│   ├── multiple_linear_regression.py  # Regresi linier multivariat & komparasi baseline
│   ├── polynomial_regression.py       # Ekspansi kuadratik & bias-variance tradeoff
│   ├── ridge_regression.py            # Regularisasi L2 & coefficient shrinkage
│   ├── lasso_regression.py            # Regularisasi L1 & sparse feature selection
│   ├── elasticnet_regression.py       # Kombinasi konveks L1/L2 & l1_ratio tuning
│   ├── regression_metrics.py          # Verifikasi rumus manual MAE/MSE/RMSE/R2 & fenomena R2<0
│   ├── residual_analysis.py           # Diagnostik residual: mean, homoskedastisitas, skewness
│   ├── regression_model_comparison.py # Benchmark 6 model regresi dengan 5-Fold CV
│   └── README.md                      # Dokumentasi katalog skrip & hasil pengujian aktual
├── notebooks/
│   └── chapter_13_companion.ipynb     # Interactive Jupyter Companion Notebook (Format v4)
├── practice.md                        # Panduan praktikum laboratorium (berkas ini)
├── index.md                           # Modul teori komprehensif (Single Source of Truth)
├── slides/
│   └── deck.json                      # Slide presentasi pengajaran Reveal.js (30 slide)
├── audit.md                           # Laporan audit mutu publikasi bab
└── metadata.json                      # Registri metadata resmi kurikulum
```

---

## 3. Prasyarat Lingkungan & Persiapan

Praktikum ini menggunakan lingkungan virtual Python terpadu (`.venv`):
- `python >= 3.10` (diverifikasi pada Python 3.12.10)
- `numpy >= 1.26.0`
- `pandas >= 2.2.0`
- `scikit-learn >= 1.4.0`

### Langkah Aktivasi Lingkungan
Buka terminal PowerShell pada direktori repositori:
```powershell
# 1. Masuk ke root direktori
cd "D:\Workstation\Repo My Corses\Kecerdasan Buatan"

# 2. Aktifkan virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Pastikan dependensi siap
python -c "import sklearn, pandas, numpy; print('Environment Chapter 13 Siap!')"
```

---

## 4. Rangkaian Praktikum Laboratorium Terpandu

Lakukan praktikum ini langkah demi langkah sesuai urutan skrip modular di bawah ini:

### Lab 1: Simple Linear Regression & Estimasi OLS
Jalankan skrip regresi linier sederhana:
```powershell
python content/level-2-machine-learning/chapter-13/code/simple_linear_regression.py
```
**Analisis Kunci:**
- Amati koefisien kemiringan $w \approx 3.63$ kW/% beban dan intersep $b \approx 98.40$ kW.
- Perhatikan bahwa satu variabel beban operasi saja sudah mampu menjelaskan $98.88\%$ variasi daya, membuktikan korelasi fisik yang sangat kuat.

### Lab 2: Multiple Linear Regression & Komparasi Baseline
Jalankan skrip regresi linier multivariat:
```powershell
python content/level-2-machine-learning/chapter-13/code/multiple_linear_regression.py
```
**Analisis Kunci:**
- Bandingkan performa regresi berganda (RMSE $5.37$ kW) terhadap Dummy Mean Baseline (RMSE $76.65$ kW). Terjadi penurunan galat drastis sebesar **$92.99\%$**.
- Interpretasikan efek marginal tiap prediktor (misal: setiap kenaikan $1^\circ\text{C}$ suhu ambien meningkatkan daya sebesar $+0.63$ kW).

### Lab 3: Polynomial Regression & Kelengkungan Non-Linear
Jalankan skrip regresi polinomial:
```powershell
python content/level-2-machine-learning/chapter-13/code/polynomial_regression.py
```
**Analisis Kunci:**
- Derajat 1 mengasumsikan efisiensi linier murni (Test RMSE $6.83$ kW).
- Derajat 2 menangkap suku kuadratik pemanasan pada beban tinggi, menurunkan Test RMSE menjadi $5.62$ kW ($R^2 = 0.9946$).
- Derajat 4 memperluas dimensi hingga 34 fitur; Train RMSE turun menjadi $5.66$ kW, namun Test RMSE berbalik naik menjadi $5.92$ kW (tanda awal overfitting).

### Lab 4: Regularisasi L2 (Ridge Regression)
Jalankan skrip Ridge Regression:
```powershell
python content/level-2-machine-learning/chapter-13/code/ridge_regression.py
```
**Analisis Kunci:**
- Amati penyusutan norma $\|w\|_2$ dari $59.87$ pada $\alpha=0.001$ menjadi $27.28$ pada $\alpha=500.0$.
- Buktikan bahwa Ridge tidak pernah memusnahkan koefisien menjadi tepat nol, melainkan menyusutkannya secara halus dan proporsional.

### Lab 5: Regularisasi L1 (Lasso Regression & Sparsity)
Jalankan skrip Lasso Regression:
```powershell
python content/level-2-machine-learning/chapter-13/code/lasso_regression.py
```
**Analisis Kunci:**
- Pada $\alpha=0.5$, jumlah fitur aktif terpangkas dari 6 menjadi 5 fitur (fitur dengan kontribusi marginal kecil dimatikan menjadi $0.0000$).
- Pada $\alpha=2.0$, hanya tersisa 3 fitur terkuat; pada $\alpha=8.0$, hanya tersisa 1 fitur (`operating_load_pct`).
- Buktikan peran Lasso sebagai seleksi fitur otomatis (*embedded feature selection*).

### Lab 6: ElasticNet Regression (Kombinasi L1 + L2)
Jalankan skrip ElasticNet:
```powershell
python content/level-2-machine-learning/chapter-13/code/elasticnet_regression.py
```
**Analisis Kunci:**
- Amati bagaimana parameter `l1_ratio` mengatur peralihan sifat model: $l_1=0.1$ bertindak mirip Ridge (mempertahankan kestabilan fitur berkorelasi), sedangkan $l_1=0.95$ bertindak mirip Lasso.

### Lab 7: Verifikasi Metrik & Pembuktian $R^2 < 0$
Jalankan skrip verifikasi metrik:
```powershell
python content/level-2-machine-learning/chapter-13/code/regression_metrics.py
```
**Analisis Kunci:**
- Verifikasi bahwa rumus matematis manual MAE, MSE, RMSE, dan $R^2$ menghasilkan angka yang $100\%$ identik dengan implementasi Scikit-learn.
- Pahami skenario di mana model yang memprediksi konstanta keliru menghasilkan **$R^2 = -14.2072$**. Buktikan bahwa $R^2$ pada data uji di luar sampel bukan kuadrat dari angka nyata dan dapat bernilai negatif!

### Lab 8: Analisis Diagnostik Residual
Jalankan skrip diagnostik residual:
```powershell
python content/level-2-machine-learning/chapter-13/code/residual_analysis.py
```
**Analisis Kunci:**
- Rata-rata galat residual bernilai $-0.56$ kW (sangat mendekati nol).
- Simpangan baku residual pada kuartil beban rendah ($4.16$ kW), sedang ($4.50$ kW), dan tinggi ($5.67$ kW) relatif stabil, memenuhi asumsi homoskedastisitas.
- Kemencengan (*skewness*) $-0.20$ mengonfirmasi bahwa distribusi galat mendekati normal tanpa pencilan ekstrem ($0\%$ outlier $>3\sigma$).

### Lab 9: Benchmark Komparatif Multi-Model
Jalankan skrip komparasi menyeluruh:
```powershell
python content/level-2-machine-learning/chapter-13/code/regression_model_comparison.py
```
**Analisis Kunci:**
- Evaluasi komparasi 6 model menggunakan 5-Fold Cross-Validation dan evaluasi akhir pada Test Set.
- Polynomial Regression Derajat 2 keluar sebagai pemenang dengan Test RMSE terendah ($4.12$ kW) dan $R^2 = 0.9971$.

---

## 5. Eksplorasi Mandiri & Tantangan Analisis

Untuk memperdalam keahlian pemodelan regresi Anda, selesaikan 3 tantangan berikut:

### Tantangan 1: Tuning Hiperparameter $\alpha$ Menggunakan `RidgeCV` dan `LassoCV`
Modifikasi skrip `ridge_regression.py` dan `lasso_regression.py`:
1. Gunakan `RidgeCV(alphas=np.logspace(-3, 3, 50))` dan `LassoCV(cv=5)`.
2. Temukan nilai $\alpha$ optimal secara otomatis berdasarkan skor validasi silang.
3. Evaluasi performa model optimal tersebut pada Test Set dan bandingkan hasilnya dengan nilai $\alpha$ manual.

### Tantangan 2: Deteksi Multikolinearitas dengan Variance Inflation Factor (VIF)
Sebelum fitting Multiple Linear Regression:
1. Hitung nilai $VIF$ untuk keenam fitur menggunakan `statsmodels.stats.outliers_influence.variance_inflation_factor`.
2. Identifikasi apakah ada pasangan fitur (misal `operating_load_pct` dan `machine_speed_rpm`) yang memiliki $VIF > 5$.
3. Bandingkan bagaimana Ridge dan Lasso menangani pasangan fitur tersebut.

### Tantangan 3: Transformasi Target Box-Cox / Logaritmik
Jika target memiliki sebaran menceng:
1. Bungkus pemodelan menggunakan `TransformedTargetRegressor` dari `sklearn.compose`.
2. Gunakan fungsi `func=np.log1p` dan `inverse_func=np.expm1`.
3. Bandingkan metrik RMSE dan sebaran residual terhadap model linier tanpa transformasi target.

---

## 6. Rubrik Penilaian Praktikum Laboratorium (100 Poin)

| Kriteria Penilaian | Bobot | Deskripsi Bukti Capaian |
|---|---|---|
| **Formulasi OLS & Regresi Linier** | 20 Poin | Kode berhasil mengekstrak $w$ dan $b$, menginterpretasikan efek marginal dengan benar, serta membuktikan penurunan galat terhadap baseline. |
| **Implementasi Polinomial & Kontrol Overfitting** | 20 Poin | Mampu mengonfigurasi `PolynomialFeatures` di dalam Pipeline secara bebas kebocoran statistik dan menganalisis titik optimal derajat polinomial. |
| **Penguasaan Regularisasi (L1, L2, ElasticNet)** | 20 Poin | Memahami perbedaan konseptual dan empiris antara penyusutan Ridge, seleksi fitur renggang Lasso ($w=0$), serta sifat kompromi ElasticNet. |
| **Kebenaran Metrik & Diagnostik Residual** | 20 Poin | Menghitung MAE, MSE, RMSE, $R^2$ secara presisi, mampu menjelaskan penyebab $R^2 < 0$, dan menguji asumsi homoskedastisitas residual. |
| **Validasi Silang & Arsitektur Pipeline** | 20 Poin | Menyatukan seluruh alur transformasi dalam Scikit-learn Pipeline; evaluasi dilakukan via 5-Fold Cross-Validation tanpa menyentuh data uji final selama tuning. |

---

## 7. Pelaporan Hasil Praktikum

Dokumentasikan hasil eksperimen mandiri, tabel ringkasan metrik aktual, serta analisis diagnostik ke dalam laporan laboratorium format Markdown:
`content/level-2-machine-learning/chapter-13/lab_report_<NIM>.md`
Pastikan seluruh nilai metrik evaluasi didasarkan pada eksekusi nyata skrip Python.

