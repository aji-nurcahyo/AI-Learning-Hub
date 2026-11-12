# Panduan Praktik Laboratorium — Chapter 12: ML Workflow, Dataset & Feature Engineering

Selamat datang di laboratorium **Chapter 12: ML Workflow, Dataset & Feature Engineering**. Bab ini merupakan pilar metodologis paling krusial dalam **Level 2: Machine Learning**. Sebagian besar kegagalan proyek kecerdasan buatan di industri bukan disebabkan oleh pemilihan algoritma yang salah, melainkan oleh cacat metodologis dalam pemisahan data (*data leakage*), pra-pemrosesan yang tidak terisolasi, dan evaluasi yang bias.

---

## 1. Tujuan Praktik

Setelah menyelesaikan seluruh rangkaian praktikum laboratorium mandiri ini, Anda diharapkan mampu:
1. Menguasai tahapan pembagian dataset *3-way* (*Train*, *Validation*, *Test*) dan memverifikasi ketiadaan tumpang tindih indeks (*zero-leakage*).
2. Membandingkan perilaku pembagian acak (*Random Split*) dengan pembagian terstratifikasi (*Stratified Split*) pada dataset dengan ketidakseimbangan kelas (*imbalanced data*).
3. Mengisolasi entitas fisik pada data menggunakan `GroupShuffleSplit` dan `GroupKFold` guna mencegah *cross-entity leakage*.
4. Menerapkan pemisahan temporal `TimeSeriesSplit` (*expanding window*) guna mengeliminasi bias melihat masa depan (*look-ahead bias*).
5. Menerapkan pra-pemrosesan numerik (imputasi median dan standardisasi) dengan aturan fit strictly hanya pada *Training Set*.
6. Mengonfigurasi pra-pemrosesan kategorikal (`OneHotEncoder` dan `OrdinalEncoder`) yang tangguh terhadap kemunculan kategori baru (*unseen categories*).
7. Merekayasa fitur baru berbasis domain industri (rasio stres fisik, laju perubahan suhu) dan suku interaksi non-linear.
8. Menangani ketidakseimbangan kelas secara bertanggung jawab menggunakan pembobotan biaya (*cost-sensitive learning*) dan *threshold tuning*.
9. Mengevaluasi stabilitas model menggunakan *Stratified K-Fold Cross-Validation* dan menganalisis varians metrik performa.
10. Membangun arsitektur *end-to-end* yang modular dan siap produksi menggunakan Scikit-learn `Pipeline` dan `ColumnTransformer`.
11. Membuktikan secara empiris dampak manipulasi metrik palsu akibat *Data Leakage* pada data kebisingan murni (*pure noise*).

---

## 2. Struktur Direktori Praktikum

Seluruh kode sumber, dataset telemetri, dan notebook interaktif tersimpan dalam struktur berikut:

```text
content/level-2-machine-learning/chapter-12/
├── code/
│   ├── data/
│   │   └── equipment_telemetry.csv    # Dataset telemetri industri sintetis (250 baris, 8 fitur)
│   ├── generate_dataset.py            # Skrip pembangkit dataset telemetri
│   ├── dataset_splitting.py           # 3-Way split (Train 60%, Val 20%, Test 20%)
│   ├── stratified_split.py            # Stratified vs Random Split pada data imbalanced
│   ├── group_split.py                 # Pencegahan cross-entity leakage (GroupKFold)
│   ├── time_series_split.py           # Pencegahan look-ahead bias (TimeSeriesSplit)
│   ├── numeric_preprocessing.py       # Imputasi & Penskalaan terisolasi train-only
│   ├── categorical_preprocessing.py   # OneHot & Ordinal Encoder tangguh unseen category
│   ├── feature_engineering.py         # Fitur domain fisik & suku interaksi non-linear
│   ├── class_imbalance_demo.py        # Cost-sensitive weights & threshold tuning
│   ├── cross_validation.py            # Evaluasi varians StratifiedKFold cross-validation
│   ├── sklearn_pipeline.py            # Pembangunan end-to-end Scikit-learn Pipeline
│   ├── column_transformer.py          # Routing heterogeneous data preprocessing
│   ├── leakage_prevention_demo.py     # Eksperimen pembuktian ilusi data leakage
│   └── README.md                      # Dokumentasi katalog skrip & hasil pengujian aktual
├── notebooks/
│   └── chapter_12_companion.ipynb     # Interactive Jupyter Companion Notebook (Format v4)
├── practice.md                        # Panduan praktikum laboratorium (berkas ini)
├── index.md                           # Modul teori komprehensif (Single Source of Truth)
├── slides/
│   └── deck.json                      # Slide presentasi pengajaran Reveal.js (34 slide)
├── audit.md                           # Laporan audit mutu publikasi bab
└── metadata.json                      # Registri metadata resmi kurikulum
```

---

## 3. Prasyarat Lingkungan & Persiapan

Praktikum ini menggunakan lingkungan Python terpadu (`.venv`):
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

# 3. Pastikan dependensi tersedia
python -c "import sklearn, pandas, numpy; print('Environment Siap!')"
```

---

## 4. Rangkaian Praktikum Laboratorium Terpandu

Lakukan praktikum ini langkah demi langkah sesuai urutan skrip modular di bawah ini:

### Lab 1: Pembangkitan Data & Pemisahan 3-Way Terisolasi
Jalankan skrip pembagian partisi dasar:
```powershell
python content/level-2-machine-learning/chapter-12/code/dataset_splitting.py
```
**Analisis Kunci:**
- Amati bagaimana dataset telemetri dipecah menjadi 150 data latih (60%), 50 data validasi (20%), dan 50 data uji (20%).
- Perhatikan pengecekan interseksi himpunan indeks: nilai tumpang tindih harus tepat 0. Mengapa data uji (*Test Set*) tidak boleh disentuh sedikit pun selama proses tuning hyperparameter?

### Lab 2: Stratifikasi pada Kasus Ketidakseimbangan Kelas
Jalankan skrip pengujian stratifikasi:
```powershell
python content/level-2-machine-learning/chapter-12/code/stratified_split.py
```
**Analisis Kunci:**
- Bandingkan standar deviasi proporsi kegagalan antar fold: K-Fold acak menghasilkan variasi $4.1183\%$, sedangkan StratifiedKFold mereduksinya menjadi $0.9798\%$.
- Jelaskan bahaya yang terjadi bila satu fold validasi tidak memiliki satu pun sampel kegagalan mesin (*zero positive*).

### Lab 3: Isolasi Kelompok Entitas (*Group Leakage*)
Jalankan skrip pemisahan berbasis kelompok:
```powershell
python content/level-2-machine-learning/chapter-12/code/group_split.py
```
**Analisis Kunci:**
- Pada *Naive Random Split*, data dari mesin yang sama muncul pada *Train* dan *Test* (5 mesin bocor). Model akan menghafal ID dan karakteristik individu mesin.
- `GroupShuffleSplit` dan `GroupKFold` membagi mesin secara eksklusif (misal: Mesin Beta hanya diuji setelah dilatih pada mesin Alpha, Gamma, Delta, dan Epsilon).

### Lab 4: Pencegahan Bias Temporal (*Time-Series Leakage*)
Jalankan skrip pemisahan urutan waktu:
```powershell
python content/level-2-machine-learning/chapter-12/code/time_series_split.py
```
**Analisis Kunci:**
- Random shuffle memicu 100% data uji berada di masa lalu sebelum batas akhir data latih (*Look-Ahead Bias*).
- `TimeSeriesSplit` menerapkan jendela berkembang (*expanding window*) di mana masa lalu melatih model untuk memprediksi masa depan yang belum terjadi.

### Lab 5: Pra-pemrosesan Fitur Numerik Bebas Kebocoran
Jalankan skrip pra-pemrosesan numerik:
```powershell
python content/level-2-machine-learning/chapter-12/code/numeric_preprocessing.py
```
**Analisis Kunci:**
- Amati bahwa nilai median yang dihitung untuk imputasi berasal *murni* dari data latih.
- Hasil standardisasi pada data latih menghasilkan rata-rata tepat 0.0000 dan deviasi standar 1.0000. Data uji menghasilkan rata-rata $-0.1530$ dan standar deviasi $1.0509$, mencerminkan realitas variasi populasi unseen data.

### Lab 6: Pra-pemrosesan Fitur Kategorikal & Penanganan Data Asing
Jalankan skrip pra-pemrosesan kategorikal:
```powershell
python content/level-2-machine-learning/chapter-12/code/categorical_preprocessing.py
```
**Analisis Kunci:**
- Saat data produksi membawa kategori baru yang tidak ada pada masa pelatihan (misal `cooling_status="EMERGENCY_COOLING"`), konfigurasi `handle_unknown='ignore'` mengubah kategori tersebut menjadi baris biner semua-nol tanpa memicu program berhenti mendadak (*crash*).
- Pada `OrdinalEncoder`, kategori asing dipetakan ke nilai $-1$ menggunakan `unknown_value=-1`.

### Lab 7: Rekayasa Fitur Domain & Interaksi Non-Linear
Jalankan skrip rekayasa fitur:
```powershell
python content/level-2-machine-learning/chapter-12/code/feature_engineering.py
```
**Analisis Kunci:**
- Fitur interaksi non-linear `thermal_mechanical_index` (perkalian getaran dan temperatur) menghasilkan korelasi Pearson tinggi terhadap kegagalan ($r = 0.4183$), jauh melampaui fitur mentah individu.
- Domain flag `critical_heat_warning` membantu memisahkan kasus darurat operasional.

### Lab 8: Penanganan Ketidakseimbangan Kelas Tanpa Resampling Pra-Split
Jalankan skrip mitigasi class imbalance:
```powershell
python content/level-2-machine-learning/chapter-12/code/class_imbalance_demo.py
```
**Analisis Kunci:**
- Model logistik standar menghasilkan recall rendah terhadap kegagalan pada ambang default 0.5.
- Penggunaan `class_weight='balanced'` atau penyesuaian ambang probabilitas pada *Validation Set* secara signifikan mendongkrak Recall dan Balanced Accuracy tanpa perlu menciptakan data sintetis (SMOTE) secara ceroboh sebelum pemisahan partisi.

### Lab 9: Cross-Validation & Analisis Varians
Jalankan skrip validasi silang:
```powershell
python content/level-2-machine-learning/chapter-12/code/cross_validation.py
```
**Analisis Kunci:**
- Evaluasi performa rata-rata dan deviasi standar lintas 5 fold.
- Mengapa nilai deviasi standar antar fold mengindikasikan tingkat keandalan model di dunia nyata?

### Lab 10: Integrasi Scikit-learn Pipeline
Jalankan skrip pipeline tunggal:
```powershell
python content/level-2-machine-learning/chapter-12/code/sklearn_pipeline.py
```
**Analisis Kunci:**
- `Pipeline` Scikit-learn mengikat proses imputasi median, penskalaan standar, dan regresi logistik menjadi satu objek monolitik.
- Panggilan `fit` secara otomatis mengeksekusi `fit_transform` pada tahapan transformer, sedangkan pemanggilan `predict` secara otomatis hanya mengeksekusi `transform` menggunakan statistik data latih yang telah terkunci.

### Lab 11: Pra-pemrosesan Data Campuran Menggunakan ColumnTransformer
Jalankan skrip pra-pemrosesan data heterogen:
```powershell
python content/level-2-machine-learning/chapter-12/code/column_transformer.py
```
**Analisis Kunci:**
- `ColumnTransformer` mengarahkan kolom numerik ke pipeline numerik, kolom nominal ke `OneHotEncoder`, dan kolom ordinal ke `OrdinalEncoder`.
- Model mencapai Balanced Accuracy $0.9444$ dan Recall Failure $1.0000$ pada data uji mentah.

### Lab 12: Eksperimen Pembuktian Bahaya Fatal Data Leakage
Jalankan skrip demonstrasi ilusi kebocoran data:
```powershell
python content/level-2-machine-learning/chapter-12/code/leakage_prevention_demo.py
```
**Analisis Kunci:**
- Pada eksperimen data kebisingan murni (*pure Gaussian noise* di mana tidak ada hubungan antara $X$ dan $y$), metode bocor (*leaky preprocessing*) memanipulasi akurasi uji menjadi **$73.33\%$** (ilusi performa tinggi).
- Metode bersih berbasis pipeline menghasilkan akurasi realistis **$58.33\%$** (mendekati probabilitas tebakan acak 50%).
- Kesimpulan: Data leakage adalah penipuan metodologis yang paling merusak dalam implementasi AI industri.

---

## 5. Eksplorasi Mandiri & Tantangan Analisis

Untuk memperdalam pemahaman praktis Anda, selesaikan 3 tantangan berikut:

### Tantangan 1: Uji Ketahanan Skaler Terhadap Pencilan Ekstrem
Modifikasi `numeric_preprocessing.py`:
1. Masukkan nilai pencilan buatan pada kolom `vibration_rms` data uji (misal nilai getaran $50.0$ akibat lonjakan sensor).
2. Bandingkan dampak transformasi `StandardScaler` versus `RobustScaler` (berbasis kuartil $IQR$).
3. Dokumentasikan mana skaler yang mempertahankan batas distribusi fitur secara lebih stabil.

### Tantangan 2: Pemilihan Ambang Probabilitas Optimal
Pada `class_imbalance_demo.py`:
1. Buat kurva *Precision-Recall* pada *Validation Set*.
2. Tentukan nilai ambang batas (*threshold*) yang memaksimalkan $F_2$-Score (memberikan bobot dua kali lebih besar pada Recall daripada Precision).
3. Evaluasi performa $F_2$-Score tersebut pada data uji yang belum pernah dilihat (*Test Set*).

### Tantangan 3: Custom Transformer untuk Rekayasa Fitur
Buat kelas transformer khusus menggunakan `BaseEstimator` dan `TransformerMixin` dari `sklearn.base`:
1. Bungkus logika pembentukan fitur `thermal_mechanical_index` dan `vibration_pressure_ratio` ke dalam kelas tersebut.
2. Integrasikan transformer buatan Anda langsung ke dalam `Pipeline` Scikit-learn.
3. Buktikan bahwa seluruh proses dapat dieksekusi dengan satu kali perintah `pipeline.fit_predict(X)`.

---

## 6. Rubrik Penilaian Praktikum Laboratorium (100 Poin)

| Kriteria Penilaian | Bobot | Deskripsi Bukti Capaian |
|---|---|---|
| **Integritas Splitting & Isolasi Data** | 20 Poin | Kode berhasil membagi dataset secara 3-way tanpa ada tumpang tindih indeks ($0$ overlap). Pemilihan strategi split (stratified/group/temporal) tepat sesuai karakteristik data. |
| **Kebersihan Pra-pemrosesan Data** | 20 Poin | Tidak ada pemanggilan `.fit()` pada data validasi maupun data uji. Imputasi dan scaling mengunci parameter yang dipelajari dari data latih secara benar. |
| **Kekokohan Penanganan Data Kategorikal** | 15 Poin | Mampu menangani fitur campuran (nominal dan ordinal) serta membuktikan ketahanan sistem saat disajikan data dengan kategori yang belum pernah ada sebelumnya (*unseen categories*). |
| **Rekayasa Fitur & Mitigasi Imbalance** | 15 Poin | Menghasilkan minimal 2 fitur domain baru yang memberikan peningkatan korelasi prediktif dan menerapkan mitigasi ketidakseimbangan kelas (*cost-sensitive* / *threshold tuning*) tanpa bias. |
| **Arsitektur Pipeline & Modularitas** | 20 Poin | Menggabungkan seluruh tahapan dalam satu Scikit-learn `Pipeline` dan `ColumnTransformer` modular yang dapat menerima data mentah secara langsung. |
| **Analisis & Dokumentasi Data Leakage** | 10 Poin | Mampu menjelaskan secara lugas bahaya data leakage berdasarkan hasil eksekusi eksperimen kuantitatif pada noise acak. |

---

## 7. Pelaporan Hasil Praktikum

Simpan hasil pengerjaan eksplorasi mandiri Anda, tangkapan layar eksekusi skrip, serta analisis numerik ke dalam berkas laporan laboratorium format Markdown:
`content/level-2-machine-learning/chapter-12/lab_report_<NIM>.md`
Pastikan seluruh nilai metrik evaluasi didasarkan pada eksekusi aktual skrip Python.

