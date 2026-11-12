# Panduan Praktik Laboratorium — Chapter 08: Data Preparation & Exploratory Analysis

Selamat datang di laboratorium **Chapter 08: Data Preparation & Exploratory Analysis**. Modul praktikum ini membimbing Anda menguasai alur kerja penyiapan data (*data preparation*) dan analisis data eksploratif (*Exploratory Data Analysis* - EDA) end-to-end menggunakan data telemetri turbin generator industri nyata (`telemetry_raw.csv`).

---

## 1. Tujuan Praktik

Setelah menyelesaikan seluruh langkah praktikum mandiri ini, Anda diharapkan mampu:
1. Menginspeksi struktur dataset, mendeteksi inkonsistensi tipe data, anomali penulisan string, dan distribusi nilai kosong (*missing values*).
2. Menganalisis mekanisme kekosongan data (MCAR, MAR, MNAR) dan menerapkan strategi penanganan yang tepat (imputasi terkelompok *group median* vs eliminasi terkontrol).
3. Mengidentifikasi dan membersihkan duplikasi baris identik serta observasi temporal ganda.
4. Mendeteksi pencilan (*outliers*) menggunakan Tukey Interquartile Range (IQR) dan Z-score, serta membedakan anomali fisik nyata dari *sensor glitch* melalui pemahaman domain (*domain engineering*).
5. Menerapkan transformasi fitur numerik: penyekalaan Min-Max, standardisasi Z-score, dan transformasi logaritma untuk menormalkan distribusi miring (*skewed*).
6. Mengubah variabel kategorikal ke representasi numerik menggunakan Ordinal Encoding dan One-Hot Encoding dengan pemahaman mengenai *dummy variable trap*.
7. Membangun visualisasi analitis terstruktur (histogram, boxplot, scatter plot, dan heatmap korelasi Pearson) menggunakan Matplotlib secara *headless*.
8. Menerapkan protokol pencegahan kebocoran data (*data leakage*) dengan memisahkan partisi *train/test* sebelum komputasi parameter statistik dilakukan.

---

## 2. Struktur Direktori Praktikum

Seluruh kode sumber mandiri, dataset pengamatan, notebook interaktif, dan grafik visual tersimpan pada struktur direktori berikut:

```text
content/level-1-ai-foundations/chapter-08/
├── code/
│   ├── data/
│   │   ├── telemetry_raw.csv                # Dataset telemetri mentah industri (24 baris, anomali nyata)
│   │   ├── telemetry_clean.csv              # Dataset bersih hasil deduplikasi, imputasi, dan standardisasi
│   │   ├── telemetry_eda_dashboard.png      # Dashboard 4 plot visualisasi distribusi & sebaran fitur
│   │   └── telemetry_correlation_heatmap.png # Matriks visual korelasi Pearson antar fitur numerik
│   ├── data_inspection.py                   # Pemeriksaan dimensi, tipe data, anomali string, deskriptif
│   ├── missing_values.py                    # Analisis MCAR/MAR/MNAR & imputasi group median berbasis zona
│   ├── duplicate_cleaning.py                # Deteksi duplikat identik vs subset observasi sensor berulang
│   ├── outlier_detection.py                 # Tukey IQR vs Z-score, domain analysis, capping/Winsorizing
│   ├── data_transformation.py               # Min-Max Scaling, Standard Scaling, dan Log-Transform skewness
│   ├── categorical_encoding.py              # Ordinal Encoding vs One-Hot Encoding & dummy variable trap
│   ├── exploratory_analysis.py              # Visualisasi multi-panel (distribusi, boxplot, scatter)
│   ├── correlation_analysis.py              # Komputasi matriks korelasi Pearson & heatmap visualisasi
│   ├── data_leakage_demo.py                 # Demonstrasi kebocoran data (leaky vs strict pipeline)
│   └── README.md                            # Katalog kode dan tabel hasil verifikasi eksekusi numerik
├── notebooks/
│   └── chapter_08_companion.ipynb           # Interactive Jupyter Companion Notebook (Format v4, 18 sel)
├── practice.md                              # Panduan praktikum laboratorium (berkas ini)
├── index.md                                 # Modul teori utama (Single Source of Truth)
└── metadata.json                            # Registri metadata chapter
```

---

## 3. Prasyarat Lingkungan & Persiapan

### Dependensi Python
Praktikum ini menggunakan lingkungan virtual Python standar (`.venv`) dengan pustaka:
- `pandas` (manipulasi tabel DataFrame dan ekspor CSV)
- `numpy` (komputasi array numerik dan transformasi logaritma)
- `matplotlib` (visualisasi analitis headless `Agg`)
- `scikit-learn` (preprocessing scalers, encoders, train_test_split)
- `scipy` (komputasi nilai statistik skewness)

Aktivasi virtual environment:
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Verifikasi ketersediaan dependensi:
```bash
python -c "import pandas, numpy, matplotlib, sklearn, scipy; print('Lingkungan laboratorium siap!')"
```

---

## 4. Panduan Eksekusi Langkah demi Langkah

### Langkah 1: Inspeksi Awal Dataset Mentah
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/data_inspection.py
```
**Hal penting yang diamati:**
- Dataset mentah memiliki 24 baris dan 8 kolom.
- Kolom `zone` memiliki ketidakkonsistenan huruf (`'Production'` vs `'production'`, `'Server Room'` vs `'server room'`).
- Terdapat 2 *missing values* pada `temperature_c` dan 1 pada `humidity_pct`.
- Ditemukan nilai suhu maksimum sebesar $88.5^\circ\text{C}$ yang jauh di atas kuartil atas.

### Langkah 2: Analisis Nilai Hilang (*Missing Values*) & Imputasi Berbasis Grup
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/missing_values.py
```
**Hal penting yang diamati:**
- Mengapa imputasi rata-rata global (*global mean*) buruk? Karena zona *Production* (suhu rata-rata $\sim 44^\circ\text{C}$) memiliki dinamika termal berbeda drastis dengan *Server Room* ($\sim 24^\circ\text{C}$). Menggunakan rata-rata global ($35.34^\circ\text{C}$) akan menggeser data secara keliru.
- Imputasi median terkelompok berdasarkan `zone` menghasilkan estimasi yang secara fisika konsisten dengan lingkungan operasional masing-masing turbin.

### Langkah 3: Pembersihan Duplikasi (*Deduplication*)
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/duplicate_cleaning.py
```
**Hal penting yang diamati:**
- Ditemukan 1 baris duplikat identik pada indeks 13 (replika indeks 7 untuk `TURB-03` pada `2026-03-01 08:05:00`).
- Melakukan deduplikasi dengan mempertahankan observasi pertama (`keep='first'`), menghasilkan 23 baris unik.

### Langkah 4: Deteksi Pencilan (*Outliers*) & Domain Engineering
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/outlier_detection.py
```
**Hal penting yang diamati:**
- Batas atas Tukey IQR adalah $57.15^\circ\text{C}$. Nilai $88.5^\circ\text{C}$ terdeteksi sebagai outlier statistik ($Z = 3.65$).
- **Pelajaran Rekayasa Domain**: Poin $88.5^\circ\text{C}$ diiringi getaran $3.85\text{ mm/s}$ dan daya $210.5\text{ kW}$. Ini adalah **kondisi beban berlebih nyata (*turbine overheating breakdown*)**, bukan kegagalan transmisi sensor. Menghapusnya akan membuat model AI buta terhadap sinyal kerusakan mekanis!
- Jika algoritma sensitif terhadap magnitudo ekstrim (seperti regresi linear), terapkan *capping/Winsorizing* ke batas persentil 95 ($46.88^\circ\text{C}$) daripada menghapus data sampel.

### Langkah 5: Transformasi Fitur Numerik (*Scaling & Skewness Reduction*)
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/data_transformation.py
```
**Hal penting yang diamati:**
- Distribusi `power_kw` memiliki skewness positif $+1.3959$ (ekor kanan panjang akibat insiden daya tinggi).
- Penerapan transformasi logaritma $\log(1 + x)$ menormalkan skewness menjadi $+0.1829$ (mendekati simetris sempurna).
- Min-Max Scaling mengompresi data ke rentang $[0, 1]$, sedangkan Standard Scaling mentransformasikannya ke $\mu = 0, \sigma = 1$.

### Langkah 6: Encoding Fitur Kategorikal
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/categorical_encoding.py
```
**Hal penting yang diamati:**
- Gunakan *Ordinal Encoding* hanya jika kategori memiliki tingkatan berurutan alami (misal status pendingin: `OFF (0) < STANDBY (1) < ACTIVE (2)`).
- Gunakan *One-Hot Encoding* untuk variabel nominal tanpa hierarki seperti lokasi zona (`Production`, `Server Room`, `Warehouse`).
- Pahami *Dummy Variable Trap*: Jika model menggunakan regresi parametrik, gunakan `drop_first=True` ($k-1$ kolom) untuk mencegah multikolinearitas linear sempurna.

### Langkah 7: Visualisasi Analisis Data Eksploratif (EDA Dashboard)
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/exploratory_analysis.py
```
**Hasil visualisasi:**
- Skrip menghasilkan file gambar dashboard: `content/level-1-ai-foundations/chapter-08/code/data/telemetry_eda_dashboard.png`.
- Menampilkan 4 panel: Histogram distribusi suhu, Boxplot getaran per zona, Scatter plot daya vs getaran, dan Bar chart rata-rata daya per zona.

### Langkah 8: Analisis Korelasi Multivariat
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/correlation_analysis.py
```
**Hasil visualisasi:**
- Skrip menghasilkan file heatmap: `content/level-1-ai-foundations/chapter-08/code/data/telemetry_correlation_heatmap.png`.
- Koefisien korelasi Pearson $r(\text{vibration}, \text{power}) = +0.8921$ dan $r(\text{temperature}, \text{vibration}) = +0.8412$, membuktikan hubungan termomekanis yang kuat.

### Langkah 9: Evaluasi & Pencegahan Kebocoran Data (*Data Leakage*)
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-08/code/data_leakage_demo.py
```
**Hal penting yang diamati:**
- Perbedaan rata-rata suhu antara subset latih murni ($34.625^\circ\text{C}$) dan populasi bocor ($35.342^\circ\text{C}$) menunjukkan bias sebesar $0.7167^\circ\text{C}$.
- Bias ini menyebabkan model memperkirakan variansi data uji lebih optimis secara semu, yang berujung pada penurunan performa drastis saat dideploy di lingkungan produksi.

---

## 5. Eksplorasi Interaktif dengan Jupyter Notebook

Buka dan jalankan Jupyter Companion Notebook:
```bash
jupyter notebook content/level-1-ai-foundations/chapter-08/notebooks/chapter_08_companion.ipynb
```
Setiap sel telah dilengkapi narasi teori dan visualisasi interaktif. Pastikan Anda menjalankan sel secara berurutan (*Kernel -> Restart & Run All*) untuk memastikan determinisme eksperimen.

---

## 6. Ringkasan Sintaks Kunci

| Operasi | Sintaks Pandas / Scikit-Learn | Catatan Penting |
| :--- | :--- | :--- |
| **Standardisasi Teks** | `df['col'] = df['col'].astype(str).str.strip().str.title()` | Hindari duplikasi semantik akibat kapitalisasi |
| **Deduplikasi** | `df.drop_duplicates(subset=['id', 'time'], keep='first')` | Pertahankan observasi valid pertama |
| **Group Imputation** | `df.groupby('zone')['temp'].transform(lambda s: s.fillna(s.median()))` | Pertahankan dinamika fisik lingkungan |
| **Tukey IQR** | `iqr = q3 - q1; lower = q1 - 1.5*iqr; upper = q3 + 1.5*iqr` | Deteksi non-parametrik yang tangguh terhadap skew |
| **Log Transform** | `np.log1p(df['power'])` | Menormalkan distribusi miring positif ($\log(1+x)$) |
| **One-Hot Encoding** | `pd.get_dummies(df, columns=['zone'], drop_first=True)` | Mencegah jebakan multikolinearitas |
| **Pencegahan Leakage** | `scaler.fit(X_train); scaler.transform(X_test)` | Jangan pernah memanggil `fit()` pada seluruh data $X$ |

