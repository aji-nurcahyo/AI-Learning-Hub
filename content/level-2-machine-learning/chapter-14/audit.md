# Laporan Audit Mutu Publikasi — Chapter 14: Supervised Learning — Classification Algorithms

**Tanggal Audit**: 2026-11-05  
**Auditor**: Antigravity Editorial & Quality Assurance Team  
**Status Bab**: `published`  
**Tingkat Pembelajaran**: Level 2 — Machine Learning  

---

## 1. Verifikasi Kelengkapan Artefak Bab

| Komponen Artefak | Lokasi Berkas | Status Verifikasi | Catatan Evaluasi |
|---|---|---|---|
| **Metadata Kurikulum** | `metadata.json` | LULUS | 14 tujuan Bloom terdaftar, prasyarat Bab 13, 5 referensi ilmiah terverifikasi dengan URL HTTPS. |
| **Modul Pembacaan Utama** | `index.md` | LULUS | Meliputi seluruh 35 topik silabus resmi, 14 diagram ASCII, latihan 3 tingkat, mini project & rubrik 100 poin. |
| **Slide Presentasi Pengajaran** | `slides/deck.json` | LULUS | 32 slide Reveal.js terstruktur, format split & card, speaker notes lengkap, bebas dari wall of text. |
| **Panduan Laboratorium** | `practice.md` | LULUS | 9 langkah praktikum terpandu, panduan eksekusi skrip, eksplorasi mandiri, dan rubrik lab 100 poin. |
| **Katalog Kode Praktik** | `code/README.md` | LULUS | Dokumentasi arsitektur 9 skrip Python + skrip pembangkit data, tabel hasil numerik terverifikasi. |
| **Notebook Pendamping** | `notebooks/chapter_14_companion.ipynb` | LULUS | Format valid Jupyter Notebook v4 dengan sel tereksekusi dan output stream lengkap. |
| **Dataset Deteksi Kerusakan** | `code/data/equipment_fault_records.csv` | LULUS | 350 baris data telemetri vibrasi & suhu industri, target biner (30.9% fault) dan multikelas (4 status). |

---

## 2. Verifikasi Eksekusi Kode Python (9 Skrip Mandiri)

Seluruh skrip dieksekusi menggunakan Python 3.12.10 di dalam virtual environment `.venv` dan berhasil dengan kode keluar 0:

```text
1. generate_dataset.py                 -> 350 sampel telemetri dihasilkan secara deterministik (242 normal, 108 abnormal).
2. binary_logistic_regression.py       -> Sigmoid, log-loss, interpretasi odds ratio (vibration_rms 5.80x), Test Acc 100%, ROC-AUC 1.0000.
3. multiclass_logistic_regression.py   -> Softmax multinomial & OvR, confusion matrix 4x4, Test Acc 98.57%, Macro F1 0.9530.
4. gaussian_naive_bayes.py             -> Likelihood Gaussian univariat, asumsi conditional independence, Test Acc 100%, ROC-AUC 1.0000.
5. multinomial_naive_bayes.py          -> Fitur frekuensi diskret 5 level, perataan Laplace (alpha=1.0) mencegah probabilitas nol, Acc 97.14%.
6. confusion_matrix_metrics.py         -> Implementasi manual TP/FP/FN/TN, Precision, Recall, Specificity, F1 identik 100% dengan Scikit-learn.
7. classification_threshold.py         -> Kalibrasi ambang keputusan T in [0.10, 0.90], analisis matriks kerugian biaya industri ($50.000 vs $200).
8. roc_auc_analysis.py                 -> Kurva ROC parametrik, interpretasi probabilitas Wilcoxon-Mann-Whitney, bahaya discrete predictions.
9. precision_recall_analysis.py        -> Kurva PR tahan imbalance, baseline prevalence 0.3143, Average Precision (AP) 1.0000.
10. classification_model_comparison.py -> Evaluasi komparatif: Dummy baseline (Recall 0.0000) vs Logistic Regression & GNB (Recall 1.0000).
```

---

## 3. Kesimpulan
Chapter 14 dinyatakan **LULUS AUDIT MUTU PUBLIKASI** dan siap digunakan untuk pengajaran serta pembelajaran mandiri mahasiswa.

