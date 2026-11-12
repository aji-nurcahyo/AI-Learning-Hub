# Laporan Audit Mutu Publikasi — Chapter 16: Support Vector Machines & k-Nearest Neighbors

**Tanggal Audit**: 2026-11-05  
**Auditor**: Antigravity Editorial & Quality Assurance Team  
**Status Bab**: `published`  
**Tingkat Pembelajaran**: Level 2 — Machine Learning  

---

## 1. Verifikasi Kelengkapan Artefak Bab

| Komponen Artefak | Lokasi Berkas | Status Verifikasi | Catatan Evaluasi |
|---|---|---|---|
| **Metadata Kurikulum** | `metadata.json` | LULUS | 12 tujuan Bloom terdaftar, prasyarat Bab 15, 5 referensi terverifikasi dengan URL HTTPS/DOI. |
| **Modul Pembacaan Utama** | `index.md` | LULUS | Meliputi seluruh topik silabus resmi (Part A s/d Part F), 16 diagram alur ASCII, latihan 3 tingkat, mini project & rubrik 100 poin. |
| **Slide Presentasi Pengajaran** | `slides/deck.json` | LULUS | 35 slide Reveal.js terstruktur, format split, card, case study, speaker notes lengkap, bebas dari wall of text. |
| **Panduan Laboratorium** | `practice.md` | LULUS | 10 langkah praktikum terpandu, panduan eksekusi skrip, eksplorasi mandiri, dan rubrik lab 100 poin. |
| **Katalog Kode Praktik** | `code/README.md` | LULUS | Dokumentasi arsitektur 10 skrip Python + skrip pembangkit data, tabel hasil numerik terverifikasi. |
| **Notebook Pendamping** | `notebooks/chapter_16_companion.ipynb` | LULUS | Format valid Jupyter Notebook v4 dengan sel tereksekusi dan output stream lengkap. |
| **Dataset Status Sensor WSN** | `code/data/wireless_sensor_network_state.csv` | LULUS | 360 baris data telemetri jaringan sensor nirkabel dengan 6 prediktor fisik dan target multikelas (CRITICAL: 140, WARNING: 123, NORMAL: 97). |

---

## 2. Verifikasi Eksekusi Kode Python (10 Skrip Mandiri)

Seluruh skrip dieksekusi menggunakan Python 3.12.10 di dalam virtual environment `.venv` dan berhasil dengan kode keluar 0:

```text
1. generate_dataset.py                 -> 360 sampel telemetri simpul sensor nirkabel dihasilkan secara deterministik.
2. linear_svm.py                       -> Linear SVM dengan StandardScaler (Akurasi Uji 62.22%), inspeksi 171 Support Vectors.
3. hard_soft_margin_demo.py            -> Ketidakfisibelan hard-margin pada data ber-derau dibuktikan, kompensasi slack diverifikasi.
4. svm_regularization.py               -> Analisis penalti C in [0.001, 1000], penurunan jumlah SV seiring penyempitan margin.
5. svm_kernel_comparison.py            -> Komparasi fungsi kernel (Linear, Poly d=3, RBF, Sigmoid), RBF mencapai skor CV tertinggi 67.78%.
6. rbf_gamma_analysis.py               -> Analisis parameter gamma in [0.001, 100], identifikasi radius lonceng Gaussian optimal.
7. knn_classifier.py                   -> KNN (k=5) berbasis memori (Akurasi Uji 57.78%, Macro F1 0.5746).
8. knn_distance_metrics.py             -> Perhitungan manual & empiris jarak Euclidean (L2), Manhattan (L1), Chebyshev (L_inf), Minkowski (p=3).
9. knn_k_parameter.py                  -> Evaluasi spektrum k in [1, 45], hafalan k=1 dibuktikan, weighted voting mengungguli uniform.
10. feature_scaling_effect.py          -> Dampak fatal ketiadaan scaling dibuktikan (RBF SVM melonjak dari 58.89% ke 70.00% saat distandarisasi).
11. svm_knn_comparison.py              -> Benchmark komprehensif 6 model (RBF SVM terbaik: Akurasi Uji 70.00%, Macro F1 68.05%).
```

---

## 3. Kesimpulan
Chapter 16 dinyatakan **LULUS AUDIT MUTU PUBLIKASI** dan siap digunakan untuk pengajaran serta pembelajaran mandiri mahasiswa.

