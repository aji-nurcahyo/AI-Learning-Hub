# Panduan Praktik Laboratorium — Chapter 07: Python Ecosystem for AI

Selamat datang di laboratorium **Chapter 07: Python Ecosystem for AI**. Modul praktikum ini membimbing Anda menguasai empat pustaka pilar komputasi kecerdasan buatan dalam bahasa pemrograman Python: **NumPy**, **Pandas**, **Matplotlib**, dan **Scikit-learn**, serta menerapkan rekayasa perangkat lunak untuk riset AI yang dapat direproduksi (*reproducible workflows*).

---

## 1. Tujuan Praktik

Setelah menyelesaikan seluruh langkah praktikum mandiri ini, Anda diharapkan mampu:
1. Memanfaatkan array multidimensi NumPy (`ndarray`), operasi tervektorisasi (*vectorization*), dan aturan ekspansi dimensi otomatis (*broadcasting*).
2. Memuat, menginspeksi, membersihkan nilai hilang (*missing values*), dan melakukan agregasi terkelompok (*GroupBy*) menggunakan Pandas.
3. Membangun visualisasi data profesional (Line plot, Scatter plot, Histogram, Bar chart) menggunakan paradigma *Figure and Axes* Matplotlib dalam mode *headless*.
4. Menerapkan pola desain antarmuka Estimator API Scikit-learn (`fit` dan `predict`), melakukan pemisahan data latih/uji (*train/test split*), dan penyekalaan fitur (*StandardScaler*) tanpa kebocoran data (*data leakage*).
5. Membangun pipeline eksperimen yang deterministik dan konsisten menggunakan kontrol *random seed* dan referensi jalur berkas relatif (`pathlib`).

---

## 2. Struktur Direktori Praktikum

Seluruh kode sumber mandiri, dataset pengamatan, dan notebook interaktif tersimpan pada struktur direktori berikut:

```text
content/level-1-ai-foundations/chapter-07/
├── code/
│   ├── data/
│   │   ├── sensor_readings.csv          # Dataset pengamatan sensor industri mentah (20 baris)
│   │   ├── sensor_readings_clean.csv    # Dataset bersih hasil imputasi dan rekayasa fitur
│   │   └── sensor_analysis_plot.png     # Dashboard visual 4 plot hasil render Matplotlib
│   ├── numpy_arrays.py                  # Pembuatan array, dimensi, efisiensi memori, slicing
│   ├── numpy_operations.py              # Operasi tervektorisasi, broadcasting, perkalian matriks @
│   ├── pandas_dataframe.py              # Anatomi DataFrame, pengaksesan .loc vs .iloc, filtering
│   ├── pandas_csv.py                    # Penanganan missing values, GroupBy, rekayasa fitur, ekspor
│   ├── matplotlib_visualization.py      # Dashboard multi-grafik dengan backend headless 'Agg'
│   ├── sklearn_workflow_demo.py         # Pola Estimator API Scikit-learn, scaling, train/test split
│   ├── reproducible_project.py          # Uji determinisme seed dan verifikasi pipeline
│   └── README.md                        # Katalog kode dan tabel hasil eksekusi aktual
├── notebooks/
│   └── chapter_07_companion.ipynb       # Interactive Jupyter Companion Notebook (Format v4)
├── practice.md                          # Panduan praktikum laboratorium (berkas ini)
└── index.md                             # Modul teori utama (Single Source of Truth)
```

---

## 3. Prasyarat Lingkungan & Persiapan

### Dependensi Python
Praktikum ini menggunakan pustaka ekosistem data science standar yang tercatat pada `requirements.txt`:
```bash
# Jika Anda menggunakan virtual environment (.venv)
.venv\Scripts\activate.bat       # Windows CMD
.venv\Scripts\Activate.ps1       # Windows PowerShell
source .venv/bin/activate        # macOS / Linux
```

Verifikasi ketersediaan paket:
```bash
python -c "import numpy, pandas, matplotlib, sklearn; print('Seluruh paket siap!')"
```

---

## 4. Panduan Eksekusi Langkah demi Langkah

### Langkah 1: Eksplorasi NumPy Array & Efisiensi Memori
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-07/code/numpy_arrays.py
```
**Hal penting yang diamati:**
- Bandingkan alokasi memori antara list Python (~351 KB untuk 10.000 elemen) dengan NumPy array (39.1 KB). NumPy menyimpan bilangan langsung dalam *contiguous memory buffer C* tanpa *pointer overhead*.
- Amati teknik *boolean masking* (`grid[grid > 20]`) yang memfilter elemen tanpa loop `for`.

### Langkah 2: Vektorisasi & Mekanisme Broadcasting
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-07/code/numpy_operations.py
```
**Hal penting yang diamati:**
- Matriks sensor berukuran $(4 \times 3)$ dijumlahkan dengan vektor kalibrasi berukuran $(3,)$. NumPy secara otomatis mereplikasi vektor sepanjang baris (*broadcasting*) sehingga seluruh sensor terkalibrasi simultan.
- Pembedaan agregasi sumbu: `axis=0` mereduksi baris menghasilkan rata-rata per fitur, sedangkan `axis=1` mereduksi kolom menghasilkan nilai per sampel.
- Perkalian matriks formal menggunakan operator `@` (`sensor_matrix @ weights`).

### Langkah 3: Anatomi Pandas DataFrame & Pengaksesan Data
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-07/code/pandas_dataframe.py
```
**Hal penting yang diamati:**
- Perhatikan tiga komponen DataFrame: Index baris, Kolom fitur, dan Array data 2D underlying.
- Bandingkan `.loc[3, 'temperature_c']` (berbasis label) dengan `.iloc[3, 2]` (berbasis posisi integer 0-indexed). Keduanya mengembalikan nilai presisi yang sama.

### Langkah 4: Pemrosesan CSV, Missing Values & GroupBy
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-07/code/pandas_csv.py
```
**Hal penting yang diamati:**
- Deteksi 2 nilai hilang pada kolom `temperature_c` dan `humidity_pct`.
- Strategi imputasi menggunakan nilai rata-rata kolom masing-masing (`fillna`), mempertahankan integritas rekaman tanpa membuang baris data.
- Analisis *GroupBy* berdasarkan `location` menunjukkan lokasi `Turbine-Bay-2` memiliki suhu dan getaran tertinggi (potensi sumber anomali).

### Langkah 5: Dashboard Visualisasi Matplotlib Headless
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-07/code/matplotlib_visualization.py
```
**Hal penting yang diamati:**
- Skrip menggunakan backend non-GUI `matplotlib.use('Agg')` yang menjamin eksekusi lancar di terminal server tanpa memerlukan display window.
- Berkas gambar `sensor_analysis_plot.png` tercipta di folder `data/` dengan resolusi tajam (DPI 150) memuat 4 plot: Line plot, Scatter plot, Histogram, dan Horizontal Bar chart.

### Langkah 6: Alur Estimator API Scikit-learn
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-07/code/sklearn_workflow_demo.py
```
**Hal penting yang diamati:**
- Pemisahan data latih ($70\%$) dan data uji ($30\%$) dengan parameter `stratify=y` dan `random_state=42`.
- `StandardScaler` dipasang (*fit*) **hanya** pada data latih (`X_train`), lalu mentransformasikan `X_train` dan `X_test`. Teknik ini mutlak wajib untuk mencegah *data leakage* (kebocoran informasi masa depan ke model).
- Siklus antarmuka model: `model.fit(X_train, y_train)` lalu `model.predict(X_test)`.

### Langkah 7: Pengujian Reproduksibilitas Pipeline
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-07/code/reproducible_project.py
```
**Hal penting yang diamati:**
- Menetapkan `seed=42` pada `random` dan `numpy.random`.
- Menjalankan dua siklus pipeline berturut-turut menghasilkan matriks luaran dengan selisih maksimum $0.0e+00$ (identik mutlak).

---

## 5. Pertanyaan Refleksi Laboratorium

1. Mengapa operasi tervektorisasi pada NumPy jauh lebih cepat daripada perulangan `for` konvensional di Python?
2. Dalam kondisi apa teknik pengisian nilai hilang (*mean imputation*) dapat merusak atau membias pola statistik data?
3. Mengapa `StandardScaler().fit()` tidak boleh dipanggil pada keseluruhan dataset sebelum proses *train/test split*?
4. Mengapa penetapan *random seed* tidak selalu menjamin reproduksibilitas 100% jika kode dijalankan pada arsitektur prosesor yang berbeda (misal Intel x86 vs Apple Silicon ARM64)?

