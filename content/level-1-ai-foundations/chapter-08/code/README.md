# Katalog Skrip Python — Chapter 08: Data Preparation & Exploratory Analysis

Direktori ini berisi kode sumber mandiri dan dataset telemetri sensor industri untuk **Chapter 08: Data Preparation & Exploratory Analysis**. Seluruh implementasi telah diverifikasi menggunakan Python 3.12.10 dan pustaka data science standar: **NumPy**, **Pandas**, **Matplotlib**, dan **Scikit-learn**.

---

## 1. Daftar Skrip & Deskripsi Fungsional

| Nama File | Topik / Fokus Modul | Fungsi & Cakupan Algoritma |
|:---|:---|:---|
| [`data_inspection.py`](./data_inspection.py) | Data Inspection & Profiling | Pemeriksaan struktur DataFrame mentah, ukuran memori, deteksi tipe data, penghitungan missing values per kolom, deteksi rekaman duplikat, dan identifikasi inkonsistensi teks kategori. |
| [`missing_values.py`](./missing_values.py) | Missing Data Handling | Evaluasi mekanisme pelesapan (MCAR, MAR, MNAR), komparasi Listwise Deletion vs Imputasi, evaluasi ketahanan Median vs Mean terhadap outlier, imputasi berbasis grup, dan pembuatan flag indikator missing. |
| [`duplicate_cleaning.py`](./duplicate_cleaning.py) | Duplicates & Validation | Identifikasi dan penghapusan exact duplicates (`drop_duplicates`), pemeriksaan kunci primer parsial, standardisasi kapitalisasi kategori (Title Case), validasi rentang fisik data, dan pencatatan audit trail. |
| [`outlier_detection.py`](./outlier_detection.py) | Outlier & Anomaly Analysis | Deteksi pencilan univariat metode Interquartile Range (IQR Tukey) dan Z-Score ($|Z| > 2.5$), analisis kontekstual multi-fitur (kegagalan turbin vs glitch sensor), serta teknik pembatasan nilai (*Winsorization / Capping*). |
| [`data_transformation.py`](./data_transformation.py) | Data Transformation | Min-Max Scaling ($[0, 1]$), Standarisasi Z-Score ($\mu=0, \sigma=1$), analisis sensitivitas outlier pada scaling, dan transformasi logaritmik $\log(x)$ untuk mereduksi kemiringan sebaran (*skewness* $1.39 \to 0.18$). |
| [`categorical_encoding.py`](./categorical_encoding.py) | Categorical Encoding | Pembedaan variabel Nominal vs Ordinal, pemetaan berperingkat Ordinal Encoding pada `cooling_status`, jebakan ordinal pada data nominal, dan penerapan One-Hot Encoding pada `zone` dengan `OneHotEncoder`. |
| [`exploratory_analysis.py`](./exploratory_analysis.py) | Exploratory Data Analysis | Analisis univariat (histogram suhu) dan bivariat (boxplot sebaran getaran per zona, scatter plot suhu vs getaran, bar chart daya), diekspor ke gambar multi-panel [`data/telemetry_eda_dashboard.png`](./data/telemetry_eda_dashboard.png) (mode headless). |
| [`correlation_analysis.py`](./correlation_analysis.py) | Correlation Analysis | Perhitungan matriks korelasi Pearson ($r$), visualisasi heatmap matriks korelasi tersimpan ke [`data/telemetry_correlation_heatmap.png`](./data/telemetry_correlation_heatmap.png), serta penjelasan prinsip kritis korelasi vs kausalitas. |
| [`data_leakage_demo.py`](./data_leakage_demo.py) | Data Leakage Prevention | Demonstrasi bahaya prapemrosesan sebelum `train_test_split` (kebocoran statistik data uji ke data latih) vs urutan yang benar (fit hanya pada data latih, transform ke data uji), dilengkapi checklist pencegahan kebocoran data. |

---

## 2. Dataset Pengamatan

- **Berkas Mentah**: [`data/telemetry_raw.csv`](./data/telemetry_raw.csv) (24 rekaman pengamatan sensor industri telemetri, memuat missing values, duplikasi, inkonsistensi string, dan anomali suhu).
- **Berkas Bersih**: [`data/telemetry_clean.csv`](./data/telemetry_clean.csv) (23 rekaman bersih hasil penanganan duplikat, harmonisasi kategori zona, dan imputasi median).
- **Lisensi**: Creative Commons Zero (CC0) / Public Domain — Bebas digunakan untuk riset dan pendidikan.

---

## 3. Cara Menjalankan Skrip

Pastikan virtual environment telah aktif:

```bash
# Menjalankan masing-masing modul praktik
python content/level-1-ai-foundations/chapter-08/code/data_inspection.py
python content/level-1-ai-foundations/chapter-08/code/missing_values.py
python content/level-1-ai-foundations/chapter-08/code/duplicate_cleaning.py
python content/level-1-ai-foundations/chapter-08/code/outlier_detection.py
python content/level-1-ai-foundations/chapter-08/code/data_transformation.py
python content/level-1-ai-foundations/chapter-08/code/categorical_encoding.py
python content/level-1-ai-foundations/chapter-08/code/exploratory_analysis.py
python content/level-1-ai-foundations/chapter-08/code/correlation_analysis.py
python content/level-1-ai-foundations/chapter-08/code/data_leakage_demo.py
```

---

## 4. Ringkasan Verifikasi Eksekusi Numerik

| Skrip | Luaran Kunci yang Diverifikasi | Status Validasi |
|:---|:---|:---:|
| `data_inspection.py` | Shape $24 \times 8$, 3 missing values, 1 exact duplicate, zone memiliki 6 variasi kapitalisasi | **VERIFIED (0 Error)** |
| `missing_values.py` | Mean suhu ($29.61\text{ }^\circ\text{C}$) terdistorsi outlier vs Median ($27.00\text{ }^\circ\text{C}$) stabil, imputasi 100% tuntas | **VERIFIED (0 Error)** |
| `duplicate_cleaning.py` | 1 baris duplikat terhapus, kategori zone tersinkronisasi menjadi 3 nilai standar Title Case | **VERIFIED (0 Error)** |
| `outlier_detection.py` | Nilai suhu $88.5\text{ }^\circ\text{C}$ terdeteksi IQR ($>38.9\text{ }^\circ\text{C}$) & Z-Score ($Z=4.39$), teridentifikasi sebagai anomali mekanis nyata | **VERIFIED (0 Error)** |
| `data_transformation.py` | Skewness daya berkurang drastis dari $+1.3959$ menjadi $+0.1829$ setelah transformasi logaritmik | **VERIFIED (0 Error)** |
| `categorical_encoding.py` | Ordinal map `LOW`=1, `MED`=2, `HIGH`=3; One-Hot Encoding menghasilkan 3 kolom biner bebas bias urutan | **VERIFIED (0 Error)** |
| `exploratory_analysis.py` | Dashboard 4 panel beresolusi tinggi tersimpan ke `telemetry_eda_dashboard.png` (187 KB) | **VERIFIED (0 Error)** |
| `correlation_analysis.py` | $r(\text{temp, vibe}) = +0.886$, $r(\text{vibe, power}) = +0.971$, heatmap tersimpan ke `telemetry_correlation_heatmap.png` (80 KB) | **VERIFIED (0 Error)** |
| `data_leakage_demo.py` | Selisih parameter mean yang dipelajari antara pola bocor vs terisolasi terbukti sebesar $0.7168\text{ }^\circ\text{C}$ | **VERIFIED (0 Error)** |

