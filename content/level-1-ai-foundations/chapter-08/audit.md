# Audit Publikasi Konten — Chapter 08: Data Preparation & Exploratory Analysis

Dokumen ini mencatat audit editorial, validasi struktural, integrasi platform, dan verifikasi eksekusi kode Python untuk **Chapter 08: Data Preparation & Exploratory Analysis**.

---

## 1. Identitas & Metadata Chapter

- **ID Chapter**: 8
- **Slug**: `chapter-08`
- **Tingkat Kurikulum**: Level 1 — AI Foundations
- **Judul Resmi**: `Data Preparation & Exploratory Analysis`
- **Status Publikasi**: `published`
- **Prerequisites**: `[7]` (Python Ecosystem for AI)
- **Tinjauan Kualitas**: `reviewed: true`, `executionVerified: true`
- **Target Pembelajaran**: 10 measurable objectives berbasis Taksonomi Bloom.
- **Daftar Pustaka Primer**: 5 referensi terverifikasi (Hadley Wickham/Tidy Data, John Tukey/EDA, Little & Rubin/Statistical Analysis with Missing Data, Pedregosa et al./Scikit-learn, Kaufman et al./Leakage in Data Mining ACM TKDD).

---

## 2. Kelengkapan Deliverables Konten

| Komponen | Jalur Berkas | Keterangan & Verifikasi |
| :--- | :--- | :--- |
| **Modul Utama** | `content/level-1-ai-foundations/chapter-08/index.md` | Single source of truth (Part A s/d K, diagram alur, 5 latihan kritis, mini project sensor turbin, rubrik 100 poin, 5 referensi primer). |
| **Slide Presentasi** | `content/level-1-ai-foundations/chapter-08/slides/deck.json` | 28 slide Reveal.js terstruktur, rata kiri, proporsional, tanpa teks mentah, dilengkapi speaker notes mendalam untuk pengajar. |
| **Panduan Praktikum** | `content/level-1-ai-foundations/chapter-08/practice.md` | Panduan laboratorium langkah demi langkah, pemahaman fisika sistem, dan tabel sintaks kunci. |
| **Companion Notebook** | `content/level-1-ai-foundations/chapter-08/notebooks/chapter_08_companion.ipynb` | Notebook interaktif Jupyter format v4 (18 sel), tervalidasi dan tereksekusi tanpa error. |
| **Katalog Kode Sumber** | `content/level-1-ai-foundations/chapter-08/code/README.md` | Dokumentasi 9 skrip mandiri Python, dataset CSV mentah & bersih, serta tabel hasil eksekusi aktual. |
| **Dataset Mentah** | `content/level-1-ai-foundations/chapter-08/code/data/telemetry_raw.csv` | 24 baris data telemetri turbin generator industri (anomali teks, missing, duplikasi, outlier nyata). |
| **Dataset Bersih** | `content/level-1-ai-foundations/chapter-08/code/data/telemetry_clean.csv` | 23 baris dataset bersih hasil deduplikasi, imputasi group median, dan standardisasi teks. |
| **Visualisasi Dashboard** | `content/level-1-ai-foundations/chapter-08/code/data/telemetry_eda_dashboard.png` | 4-panel visualisasi headless mode ('Agg') distribusi suhu, boxplot getaran, scatter daya, dan bar chart. |
| **Heatmap Korelasi** | `content/level-1-ai-foundations/chapter-08/code/data/telemetry_correlation_heatmap.png` | Matriks visualisasi korelasi Pearson antar fitur numerik sensor industri. |

---

## 3. Hasil Verifikasi Eksekusi Skrip Python (3.12.10)

| File Skrip | Pustaka Kunci | Hasil Eksekusi Aktual | Exit Code |
| :--- | :--- | :--- | :---: |
| `data_inspection.py` | Pandas | Pemeriksaan dimensi 24x8, deteksi 2 missing suhu & 1 kelembaban, anomali string zona | 0 |
| `missing_values.py` | Pandas | Imputasi terkelompok *group median* per zona (Production, Server Room, Warehouse), sisa missing 0 | 0 |
| `duplicate_cleaning.py` | Pandas | Deteksi 1 duplikat identik (indeks 13 replika indeks 7), deduplikasi menghasilkan 23 baris bersih | 0 |
| `outlier_detection.py` | NumPy, Pandas | Tukey IQR batas atas 57.15 °C, Z-score 3.65 pada 88.5 °C terbukti kejadian fisik nyata (*mechanical overload*), Winsorizing demo | 0 |
| `data_transformation.py` | NumPy, Scipy, Sklearn | Skewness awal daya listrik +1.3959 berhasil dinormalkan menjadi +0.1829 via log1p, MinMax vs StandardScaler | 0 |
| `categorical_encoding.py` | Pandas, Sklearn | Ordinal Encoding status pendingin (0, 1, 2) vs One-Hot Encoding zona bebas multikolinearitas | 0 |
| `exploratory_analysis.py` | Matplotlib, Pandas | Render headless mode ('Agg'), dashboard 4 panel tersimpan ke `telemetry_eda_dashboard.png` | 0 |
| `correlation_analysis.py` | Matplotlib, Pandas | Matriks Pearson, korelasi getaran-daya (+0.8921) dan suhu-getaran (+0.8412), tersimpan ke `telemetry_correlation_heatmap.png` | 0 |
| `data_leakage_demo.py` | Scikit-learn, Pandas | Pembuktian selisih rata-rata suhu data latih strict vs populasi leaky sebesar 0.7167 °C | 0 |

