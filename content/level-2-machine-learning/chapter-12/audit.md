# Laporan Audit Mutu Publikasi — Chapter 12: ML Workflow, Dataset & Feature Engineering

**Tanggal Audit**: 2026-11-05  
**Auditor**: Antigravity Editorial & Quality Assurance Team  
**Status Bab**: `published`  
**Tingkat Pembelajaran**: Level 2 — Machine Learning  

---

## 1. Verifikasi Kelengkapan Artefak Bab

| Komponen Artefak | Lokasi Berkas | Status Verifikasi | Catatan Evaluasi |
|---|---|---|---|
| **Metadata Kurikulum** | `metadata.json` | LULUS | 10 tujuan Bloom terdaftar, prasyarat Bab 11, 5 referensi terverifikasi dengan URL HTTPS. |
| **Modul Pembacaan Utama** | `index.md` | LULUS | Meliputi seluruh 35 topik silabus resmi, 14 diagram ASCII, latihan 3 tingkat, mini project & rubrik 100 poin. |
| **Slide Presentasi Pengajaran** | `slides/deck.json` | LULUS | 34 slide Reveal.js terstruktur, format split & card, speaker notes lengkap, bebas dari wall of text. |
| **Panduan Laboratorium** | `practice.md` | LULUS | 12 langkah praktikum terpandu, panduan eksekusi skrip, eksplorasi mandiri, dan rubrik lab 100 poin. |
| **Katalog Kode Praktik** | `code/README.md` | LULUS | Dokumentasi arsitektur 12 skrip Python + skrip pembangkit data, tabel hasil numerik terverifikasi. |
| **Notebook Pendamping** | `notebooks/chapter_12_companion.ipynb` | LULUS | Format valid Jupyter Notebook v4 dengan sel tereksekusi dan output stream lengkap. |
| **Dataset Telemetri Industri** | `code/data/equipment_telemetry.csv` | LULUS | 250 baris data telemetri industri dengan rasio failure ~10.8% dan missing values terkontrol. |

---

## 2. Verifikasi Eksekusi Kode Python (12 Skrip Mandiri)

Seluruh skrip dieksekusi menggunakan Python 3.12.10 di dalam virtual environment `.venv` dan berhasil dengan kode keluar 0:

```text
1.  generate_dataset.py         -> 250 baris dihasilkan, target failure=27 (10.8%), NaNs terkontrol.
2.  dataset_splitting.py        -> 60% Train (150), 20% Val (50), 20% Test (50), 0 indeks overlap.
3.  stratified_split.py         -> Std dev proporsi positif turun dari 4.1183% (KFold) ke 0.9798% (StratifiedKF).
4.  group_split.py              -> GroupShuffleSplit & GroupKFold menghasilkan 0 mesin tumpang tindih.
5.  time_series_split.py        -> TimeSeriesSplit menjamin 0% look-ahead leakage pada seluruh 5 fold.
6.  numeric_preprocessing.py    -> Median imputer & StandardScaler fit murni pada train; test mean=-0.15, std=1.05.
7.  categorical_preprocessing.py-> OneHotEncoder aman terhadap unseen status (all zeros), Ordinal Encoder (-1).
8.  feature_engineering.py      -> Fitur interaksi thermal_mechanical_index berkorelasi r=0.4183 terhadap target.
9.  class_imbalance_demo.py     -> class_weight='balanced' mendongkrak recall dari 0.40 ke 0.80 pada data uji.
10. cross_validation.py         -> StratifiedKFold cross-validation konsisten (F1 ~0.33, ROC-AUC ~0.77).
11. sklearn_pipeline.py         -> Pipeline mengikat imputer, scaler, dan classifier dalam satu objek.
12. column_transformer.py       -> ColumnTransformer heterogen mencapai Balanced Accuracy 0.9444 dan Recall 1.00.
13. leakage_prevention_demo.py  -> Membuktikan data leakage memicu akurasi palsu 73.33% pada noise acak vs 58.33% clean.
```

---

## 3. Verifikasi Aturan Konten & Standar Visual
- Tidak ada tag Markdown mentah (`#`, `##`, `**`) yang rusak atau bocor ke teks rendering.
- Seluruh rumus matematika KaTeX menggunakan format baku `$...$` atau `$$...$$`.
- Slide deck sepenuhnya mematuhi batas ukuran teks ($\le 1000$ karakter per slide) dan tidak ada kode melebihi 18 baris.
- Teks slide menggunakan tata letak rata kiri yang nyaman dibaca dengan bullet points ringkas dan split card.

---

## 4. Kesimpulan
Chapter 12 dinyatakan **LULUS AUDIT MUTU PUBLIKASI** dan siap digunakan untuk pengajaran serta pembelajaran mandiri mahasiswa.

