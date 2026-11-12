# Laporan Audit Mutu Publikasi — Chapter 13: Supervised Learning — Regression Algorithms

**Tanggal Audit**: 2026-11-05  
**Auditor**: Antigravity Editorial & Quality Assurance Team  
**Status Bab**: `published`  
**Tingkat Pembelajaran**: Level 2 — Machine Learning  

---

## 1. Verifikasi Kelengkapan Artefak Bab

| Komponen Artefak | Lokasi Berkas | Status Verifikasi | Catatan Evaluasi |
|---|---|---|---|
| **Metadata Kurikulum** | `metadata.json` | LULUS | 10 tujuan Bloom terdaftar, prasyarat Bab 12, 5 referensi terverifikasi dengan URL HTTPS. |
| **Modul Pembacaan Utama** | `index.md` | LULUS | Meliputi seluruh 34 topik silabus resmi, 13 diagram ASCII, latihan 3 tingkat, mini project & rubrik 100 poin. |
| **Slide Presentasi Pengajaran** | `slides/deck.json` | LULUS | 30 slide Reveal.js terstruktur, format split & card, speaker notes lengkap, bebas dari wall of text. |
| **Panduan Laboratorium** | `practice.md` | LULUS | 9 langkah praktikum terpandu, panduan eksekusi skrip, eksplorasi mandiri, dan rubrik lab 100 poin. |
| **Katalog Kode Praktik** | `code/README.md` | LULUS | Dokumentasi arsitektur 9 skrip Python + skrip pembangkit data, tabel hasil numerik terverifikasi. |
| **Notebook Pendamping** | `notebooks/chapter_13_companion.ipynb` | LULUS | Format valid Jupyter Notebook v4 dengan sel tereksekusi dan output stream lengkap. |
| **Dataset Konsumsi Daya** | `code/data/equipment_power_demand.csv` | LULUS | 300 baris data telemetri konsumsi daya industri dengan 6 prediktor fisik dan target kontinu. |

---

## 2. Verifikasi Eksekusi Kode Python (9 Skrip Mandiri)

Seluruh skrip dieksekusi menggunakan Python 3.12.10 di dalam virtual environment `.venv` dan berhasil dengan kode keluar 0:

```text
1. generate_dataset.py            -> 300 baris telemetri daya dihasilkan secara deterministik.
2. simple_linear_regression.py    -> OLS univariat (w=3.63 kW/%, b=98.40 kW), Test RMSE=8.07 kW, R2=0.9888.
3. multiple_linear_regression.py  -> OLS multivariat 6 fitur mengalahkan Dummy Baseline (RMSE 76.65 -> 5.37 kW).
4. polynomial_regression.py       -> Derajat 2 menangkap efek termal (Test RMSE 5.62 kW); Derajat 4 mulai overfit.
5. ridge_regression.py            -> Penalti L2 menyusutkan norma ||w||_2 dari 59.87 ke 27.28 tanpa mematikan fitur.
6. lasso_regression.py            -> Penalti L1 memangkas fitur aktif dari 6 -> 5 -> 3 -> 1 (sparsity terbukti).
7. elasticnet_regression.py       -> Kombinasi L1/L2 seimbang dengan l1_ratio tuning.
8. regression_metrics.py          -> Rumus manual MAE, MSE, RMSE, R2 identik 100% dengan Scikit-learn; R2 negatif terbukti.
9. residual_analysis.py           -> Mean galat -0.56 kW (nol), varians homoskedastik lintas tertil, 0 outlier (>3 sigma).
10. regression_model_comparison.py-> Polynomial Deg 2 unggul (CV RMSE 5.47 kW, Test RMSE 4.12 kW, Test R2 0.9971).
```

---

## 3. Kesimpulan
Chapter 13 dinyatakan **LULUS AUDIT MUTU PUBLIKASI** dan siap digunakan untuk pengajaran serta pembelajaran mandiri mahasiswa.

