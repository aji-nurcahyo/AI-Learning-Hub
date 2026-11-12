# Katalog Skrip Python — Chapter 07: Python Ecosystem for AI

Direktori ini berisi kode sumber mandiri dan data pengamatan untuk **Chapter 07: Python Ecosystem for AI**. Seluruh contoh kode telah diuji menggunakan Python 3.12.10 dengan paket ekosistem modern: **NumPy**, **Pandas**, **Matplotlib**, dan **Scikit-learn**.

---

## 1. Daftar Skrip & Deskripsi Fungsional

| Nama File | Modul / Topik Utama | Fungsi & Cakupan Algoritma |
|:---|:---|:---|
| [`numpy_arrays.py`](./numpy_arrays.py) | NumPy Arrays & Slicing | Pembuatan array ndarray, inspeksi atribut (shape, ndim, dtype), perbandingan efisiensi alokasi memori list vs ndarray, indexing & slicing 2D, boolean masking, dan reshaping. |
| [`numpy_operations.py`](./numpy_operations.py) | NumPy Operasi & Linear Algebra | Operasi aritmetika tervektorisasi (*vectorization*), mekanisme penyesuaian dimensi otomatis (*broadcasting*), agregasi berbasis sumbu (`axis=0` vs `axis=1`), standarisasi z-score, dan perkalian matriks menggunakan operator `@`. |
| [`pandas_dataframe.py`](./pandas_dataframe.py) | Pandas DataFrame Anatomy | Pembuatan Series dan DataFrame dari struktur data dasar, inspeksi struktur data (`.describe()`, `.info()`), pengaksesan berbasis label (`.loc`) vs posisi integer (`.iloc`), dan penyaringan bersyarat. |
| [`pandas_csv.py`](./pandas_csv.py) | Pandas CSV & Data Cleaning | Pembacaan file CSV menggunakan `pathlib`, deteksi nilai hilang (`.isna()`), strategi imputasi mean, agregasi terkelompok (`.groupby()`), rekayasa fitur sederhana, dan ekspor data bersih. |
| [`matplotlib_visualization.py`](./matplotlib_visualization.py) | Matplotlib Multi-Plot Dashboard | Implementasi paradigma Object-Oriented (Figure & Axes), backend non-GUI headless ('Agg'), pembuatan 4 jenis grafik fundamental (Line plot, Scatter plot, Histogram, Bar chart), dan ekspor dashboard ke format PNG. |
| [`sklearn_workflow_demo.py`](./sklearn_workflow_demo.py) | Scikit-learn Estimator API | Pola desain antarmuka Estimator Scikit-learn: pemisahan matriks fitur $\mathbf{X}$ dan vektor target $\mathbf{y}$, `train_test_split`, `StandardScaler` (tanpa kebocoran data), pemanggilan `.fit()` dan `.predict()`, serta evaluasi akurasi dan confusion matrix. |
| [`reproducible_project.py`](./reproducible_project.py) | Reproducibility & Project Pipeline | Penetapan global seed (`random`, `numpy`), manajemen jalur berkas relatif independen platform, eksekusi pipeline deterministik, verifikasi kesamaan luaran antar siklus uji, dan catatan batas reproduksibilitas. |

---

## 2. Dataset Pengamatan

- **Berkas**: [`data/sensor_readings.csv`](./data/sensor_readings.csv)
- **Sumber**: Dataset sintetis pengukuran sensor IoT industri (suhu, kelembaban, getaran pada 4 lokasi fasilitas: Server-Room-A, Server-Room-B, Turbine-Bay-1, Turbine-Bay-2).
- **Lisensi**: Creative Commons Zero (CC0) / Public Domain — Bebas digunakan untuk keperluan edukasi dan riset.
- **Ukuran**: 20 rekaman data pengamatan dengan 7 kolom fitur.

---

## 3. Cara Menjalankan Skrip

Pastikan virtual environment telah aktif atau pustaka NumPy, Pandas, Matplotlib, dan Scikit-learn telah terpasang:

```bash
# Menjalankan skrip demonstrasi masing-masing
python content/level-1-ai-foundations/chapter-07/code/numpy_arrays.py
python content/level-1-ai-foundations/chapter-07/code/numpy_operations.py
python content/level-1-ai-foundations/chapter-07/code/pandas_dataframe.py
python content/level-1-ai-foundations/chapter-07/code/pandas_csv.py
python content/level-1-ai-foundations/chapter-07/code/matplotlib_visualization.py
python content/level-1-ai-foundations/chapter-07/code/sklearn_workflow_demo.py
python content/level-1-ai-foundations/chapter-07/code/reproducible_project.py
```

---

## 4. Ringkasan Verifikasi Eksekusi

| Skrip | Luaran Utama | Status Validasi |
|:---|:---|:---:|
| `numpy_arrays.py` | Slicing 2D, boolean masking (`grid > 20`), memory efficiency ~39.1 KB vs 351.6 KB | **VERIFIED (0 Error)** |
| `numpy_operations.py` | Broadcasting (4,3)+(3,) -> (4,3), Axis-0 mean, MatMul `@` layer output (4,2) | **VERIFIED (0 Error)** |
| `pandas_dataframe.py` | DataFrame 6x5, pengaksesan `.loc` vs `.iloc` identik 31.2 °C, filtering anomali | **VERIFIED (0 Error)** |
| `pandas_csv.py` | CSV loading 20 baris, 2 missing values terimputasi, groupby location, ekspor clean CSV | **VERIFIED (0 Error)** |
| `matplotlib_visualization.py` | Dashboard 2x2 dirender dengan backend Agg, diekspor ke `sensor_analysis_plot.png` (198 KB) | **VERIFIED (0 Error)** |
| `sklearn_workflow_demo.py` | Split 70:30, StandardScaler train fit only, LogisticRegression fit/predict, Akurasi 100% | **VERIFIED (0 Error)** |
| `reproducible_project.py` | Pengujian 2 siklus pipeline dengan seed 42 menghasilkan kesamaan mutlak (selisih = 0.0) | **VERIFIED (0 Error)** |

