# Laporan Audit Kualitas: Chapter 20 — Machine Learning Capstone

**Tanggal Audit**: 5 November 2026  
**Status**: **PASSED (100% MEMENUHI STANDAR)**  
**Auditor**: Tim Penjamin Mutu Kurikulum AI Learning Hub  
**Cakupan**: Bab 20 — Machine Learning Capstone (Level 2 Finalization)

---

## 1. Verifikasi Keselarasan Master Curriculum

| Butir Evaluasi | Standar Kurikulum Resmi | Status Aktual Repositori | Keterangan |
|---|---|:---:|---|
| Nomor Bab | Chapter 20 | 20 | Sesuai |
| Judul Resmi | Machine Learning Capstone | Machine Learning Capstone | Sesuai |
| Level Pembelajaran | Level 2: Machine Learning | Level 2: Machine Learning | Sesuai |
| Prasyarat | Chapter 19 | `[19]` | Sesuai |
| Status Publikasi | `published` | `published` | Sesuai |
| Batasan Arsitektur | Pembelajaran mesin terawasi/tak terawasi klasik | Scikit-Learn (LR, SVM, RF, GB) | Bebas Deep Learning / NN / LLM |

---

## 2. Verifikasi Artefak Pembelajaran

| No | Artefak | Lokasi Berkas | Status Verifikasi |
|---|---|---|:---:|
| 1 | Modul Teori (SSOT) | `chapter-20/index.md` | Lengkap (13 Bagian, 16 Diagram ASCII, 5 Referensi) |
| 2 | Slide Reveal.js | `chapter-20/slides/deck.json` | 21 Slide valid, format direct array JSON |
| 3 | Panduan Praktik Lab | `chapter-20/practice.md` | Lengkap (5 Jalur Proyek, 8 Tahapan, Rubrik 100 Poin) |
| 4 | Dataset Telemetri | `chapter-20/code/data/predictive_maintenance_telemetry.csv` | 600 Sampel, 9 Kolom, 12 Unit Pompa |
| 5 | Skrip Python Lengkap | `chapter-20/code/*.py` (10 skrip) | Seluruh skrip dieksekusi dengan exit code 0 |
| 6 | Notebook Pendamping | `chapter-20/notebooks/chapter_20_companion.ipynb` | Valid Jupyter Notebook v4 (Pre-executed) |
| 7 | Katalog Kode | `chapter-20/code/README.md` | Panduan eksekusi dan tabel skrip terverifikasi |
| 8 | Metadata Bab | `chapter-20/metadata.json` | 12 Bloom Objectives, 5 Referensi terverifikasi |

---

## 3. Verifikasi Metrik Eksperimen & Hasil Komputasi

Seluruh nilai eksperimen berikut telah dieksekusi dan diverifikasi secara langsung menggunakan Python 3.12.10:

```
[Tahap 1: Formulasi & Baseline]
- Dummy Majority Baseline         : Akurasi = 59.17% | F1 = 0.7435 | ROC-AUC = 0.5000
- Simple Logistic Regression      : Akurasi = 95.83% | F1 = 0.9655 | ROC-AUC = 0.9914

[Tahap 3: Benchmark Model Kandidat (5-Fold Stratified CV)]
- Logistic Regression (L2)        : Akurasi = 92.08% | F1 = 0.9336 | ROC-AUC = 0.9818
- Support Vector Machine (RBF)    : Akurasi = 92.50% | F1 = 0.9373 | ROC-AUC = 0.9697
- Random Forest Classifier        : Akurasi = 92.29% | F1 = 0.9350 | ROC-AUC = 0.9762
- Gradient Boosting Classifier    : Akurasi = 91.88% | F1 = 0.9319 | ROC-AUC = 0.9747

[Tahap 4: Hyperparameter Tuning (GridSearchCV)]
- Champion Model                  : Random Forest Classifier (n_estimators=150, max_depth=8, min_samples_split=10)
- Best CV F1-Score                : 0.9390 (Meningkat dari baseline tanpa tuning)

[Tahap 5: Evaluasi Untouched Test Set (N = 120)]
- Akurasi Test Set                : 93.33%
- Presisi Test Set                : 93.15%
- Recall Test Set                 : 95.77% (68 dari 71 kerusakan terdeteksi)
- F1-Score Test Set               : 0.9444
- ROC-AUC Test Set                : 0.9859
- Confusion Matrix                : TN = 44, FP = 5, FN = 3, TP = 68

[Tahap 6: Error Analysis & Cost-Sensitive Threshold Optimization]
- Cost Matrix Industri            : Biaya FN = $5,000 | Biaya FP = $200
- Risiko Finansial Threshold 0.50 : 3 FN, 5 FP -> Total Risiko = $16,000
- Risiko Finansial Threshold 0.30 : 1 FN, 6 FP -> Total Risiko = $6,200
- Efisiensi Finansial             : Penghematan risiko kerugian $9,800 (61.25%)

[Tahap 7: Transparansi Penalaran (Explainability)]
- Fitur Kunci 1                   : bearing_temperature_c (Mean Drop in F1 = 0.1278)
- Fitur Kunci 2                   : vibration_rms_g (Mean Drop in F1 = 0.1217)
- Karakteristik PDP               : Peluang kerusakan melonjak >80% pada suhu bantalan > 75°C

[Tahap 8: Responsible AI & Fairness Audit]
- Disparitas F1 Antar 12 Unit     : 0.2000 (Toleransi sistem < 0.3000 -> LOLOS AUDIT)
- Batasan Tata Kelola Operasional : Wajib intervensi teknisi; dilarang untuk pemutus darurat tunggal
```

---

## 4. Kesimpulan Audit Akhir
Bab 20 telah memenuhi seluruh standar pedagogi, rekayasa kode, dan integrasi kurikulum. Dengan terselesaikannya Bab 19 dan Bab 20, seluruh **Level 2: Machine Learning (Bab 11 hingga 20)** resmi berstatus **PUBLISHED 100%**.

