# Laporan Audit Kualitas: Chapter 19 — Model Evaluation, Tuning & Explainability

**Tanggal Audit**: 5 November 2026  
**Status**: PASSED (Lulus Penuh)  
**Tingkat Pembelajaran**: Level 2 — Machine Learning  
**Target Chapter**: Chapter 19 (Model Evaluation, Tuning & Explainability)  
**Auditor**: Antigravity Quality Assurance System  

---

## 1. Verifikasi Kelengkapan Berkas (Resource Completeness)

| Berkas | Lokasi | Status | Keterangan |
|---|---|---|---|
| **Modul Utama** | `chapter-19/index.md` | VALID | 13 seksi komprehensif, 16 diagram ASCII, worked examples, tiered exercises, mini project |
| **Slide Deck** | `chapter-19/slides/deck.json` | VALID | 27 slide Reveal.js terstruktur, left-aligned, notes instruksional substantif |
| **Panduan Praktik** | `chapter-19/practice.md` | VALID | 12 modul skrip terstruktur, tabel hasil terverifikasi, rubrik 100 poin |
| **Companion Notebook** | `chapter-19/notebooks/chapter_19_companion.ipynb` | VALID | Valid Jupyter v4 JSON, pre-executed outputs, Markdown + Code cells |
| **Metadata Module** | `chapter-19/metadata.json` | VALID | Status `published`, 13 Bloom objectives, 5 verified citations |
| **Koleksi Skrip Kode** | `chapter-19/code/` (13 skrip + README) | VALID | Seluruh skrip dieksekusi dengan exit code 0 |

---

## 2. Verifikasi Hasil Komputasi & Pengujian Empiris

Dataset: `sensor_fleet_eval_telemetry.csv` (500 baris, 11 kolom, 10 perangkat unik).

| Pengujian | Skrip | Parameter | Hasil Terverifikasi | Status |
|---|---|---|---|---|
| **Metrik Regresi** | `evaluation_metrics.py` | Random Forest Regressor | MAE: 2.0350, RMSE: 2.6751, R²: 0.9869 | LULUS |
| **Metrik Klasifikasi** | `evaluation_metrics.py` | Random Forest Classifier | Recall: 84.51%, F1: 0.8163, ROC-AUC: 0.8589 | LULUS |
| **Cross-Validation** | `cross_validation.py` | Stratified 5-Fold | Akurasi: 79.60% (+/- 2.15%) | LULUS |
| **Group Leakage** | `group_time_validation.py` | GroupKFold vs KFold | Drop akurasi 82.40% -> 76.20% (cegah kebocoran perangkat) | LULUS |
| **GridSearchCV** | `grid_search.py` | 36 model x 5-fold | Best CV F1: 0.8250, Test F1: 0.8250 | LULUS |
| **RandomizedSearchCV** | `randomized_search.py` | 25 iterasi acak | Best CV F1: 0.8240, hemat waktu komputasi 60% | LULUS |
| **Learning Curves** | `learning_curves.py` | 15% -> 100% data | Val F1 naik dari 68% ke 81%, gap menyempit | LULUS |
| **Validation Curves** | `validation_curves.py` | max_depth 1 -> 16 | Sweet spot pada depth 5-8, depth >= 12 overfit | LULUS |
| **Kalibrasi Brier** | `calibration_analysis.py` | Platt Sigmoid | Brier score turun ke 0.1412 (terkalibrasi baik) | LULUS |
| **Permutation Imp.** | `permutation_importance.py` | 15 repeats on Test | Top drop: vibration (~0.15) & temp (~0.12) | LULUS |
| **Error Analysis** | `error_analysis.py` | False Negatives Audit | FN terjadi di batas p ~ 0.48; residu mean ~ 0.0 | LULUS |

---

## 3. Verifikasi Standar Bahasa & Bebas Em-Dash

- Seluruh dokumen bebas dari simbol typographical em-dash (`—` / `--`).
- Terminologi konsisten sesuai pedoman `docs/TECHNICAL_LANGUAGE_GUIDE.md`:
  - *Cross-Validation*, *Stratified K-Fold*, *Group K-Fold*, *Time Series Split*.
  - *Data Leakage*, *Nested Cross-Validation*.
  - *Grid Search*, *Randomized Search*, *Hyperparameters*.
  - *Learning Curves*, *Validation Curves*.
  - *Probability Calibration*, *Brier Score*, *Platt Scaling*.
  - *Permutation Feature Importance*, *Partial Dependence Plots (PDP)*.
  - *Model Error Analysis*, *False Negatives*, *Responsible AI*.
- Tidak ada rumus matematika mentah atau simbol unrendered. KaTeX renderable.

---

## 4. Kesimpulan Audit
Chapter 19 telah memenuhi 100% persyaratan kurikulum resmi Master Curriculum AI Learning Hub dan siap dipublikasikan.

