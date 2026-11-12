# Laporan Audit Mutu Publikasi — Chapter 15: Decision Trees & Ensemble Learning

**Tanggal Audit**: 2026-11-05  
**Auditor**: Antigravity Editorial & Quality Assurance Team  
**Status Bab**: `published`  
**Tingkat Pembelajaran**: Level 2 — Machine Learning  

---

## 1. Verifikasi Kelengkapan Artefak Bab

| Komponen Artefak | Lokasi Berkas | Status Verifikasi | Catatan Evaluasi |
|---|---|---|---|
| **Metadata Kurikulum** | `metadata.json` | LULUS | 11 tujuan Bloom terdaftar, prasyarat Bab 14, 5 referensi terverifikasi dengan URL HTTPS/DOI. |
| **Modul Pembacaan Utama** | `index.md` | LULUS | Meliputi seluruh topik silabus resmi (Part A s/d Part F), 16 diagram alur ASCII, latihan 3 tingkat, mini project & rubrik 100 poin. |
| **Slide Presentasi Pengajaran** | `slides/deck.json` | LULUS | 34 slide Reveal.js terstruktur, format split, card, case study, speaker notes lengkap, bebas dari wall of text. |
| **Panduan Laboratorium** | `practice.md` | LULUS | 9 langkah praktikum terpandu, panduan eksekusi skrip, eksplorasi mandiri, dan rubrik lab 100 poin. |
| **Katalog Kode Praktik** | `code/README.md` | LULUS | Dokumentasi arsitektur 11 skrip Python + skrip pembangkit data, tabel hasil numerik terverifikasi. |
| **Notebook Pendamping** | `notebooks/chapter_15_companion.ipynb` | LULUS | Format valid Jupyter Notebook v4 dengan sel tereksekusi dan output stream lengkap. |
| **Dataset Sensor Kerusakan** | `code/data/equipment_fault_sensor_data.csv` | LULUS | 360 baris data telemetri sensor industri dengan 6 prediktor fisik dan target multikelas (FAULT: 149, WARNING: 120, NORMAL: 91). |

---

## 2. Verifikasi Eksekusi Kode Python (11 Skrip Mandiri)

Seluruh skrip dieksekusi menggunakan Python 3.12.10 di dalam virtual environment `.venv` dan berhasil dengan kode keluar 0:

```text
1. generate_dataset.py                 -> 360 sampel telemetri sensor industri dihasilkan secara deterministik.
2. decision_tree_classifier.py         -> Gini (Akurasi 92.22%) vs Entropy (Akurasi 90.00%), Confusion Matrix 3x3.
3. decision_tree_regressor.py          -> Prediksi suhu kontinu piecewise constant, evaluasi RMSE & R2 lintas kedalaman.
4. entropy_gini_information_gain.py    -> Perhitungan manual Entropy (0.9710) & Gini (0.4800) identik 100% dengan Scikit-learn.
5. tree_visualization.py               -> Ekspor aturan logika IF-THEN tekstual dan visualisasi node teratas.
6. tree_pruning.py                     -> Overfitting pohon unpruned dibuktikan (gap 10%), optimal ccp_alpha diidentifikasi.
7. bagging_classifier.py               -> Agregasi 50 pohon bootstrap mereduksi varians, akurasi uji meningkat ke 91.11%.
8. random_forest_classifier.py         -> Feature subsampling sqrt(p) mendekomposisi korelasi pohon, MDI feature importance dianalisis.
9. out_of_bag_evaluation.py            -> Skor OOB (93.70%) memvalidasi estimasi 5-Fold Stratified CV (94.81%) tanpa bias.
10. adaboost_classifier.py             -> Agregasi 100 decision stumps berbobot adaptif, analisis tuning learning rate.
11. gradient_boosting.py               -> Optimasi loss pseudo-residuals bertahap, staged log-loss menurun dari iterasi 1 ke 100.
12. ensemble_model_comparison.py       -> Evaluasi komparatif 7 model pada 5-Fold Stratified CV (Gradient Boosting terbaik: CV Acc 96.30%).
```

---

## 3. Kesimpulan
Chapter 15 dinyatakan **LULUS AUDIT MUTU PUBLIKASI** dan siap digunakan untuk pengajaran serta pembelajaran mandiri mahasiswa.

