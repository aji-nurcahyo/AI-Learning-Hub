# Katalog Kode Praktik Chapter 14: Supervised Learning — Classification Algorithms

Katalog ini memuat implementasi skrip Python modular untuk eksplorasi, pemodelan probabilistik, kalibrasi ambang batas, dan evaluasi diagnostik algoritma klasifikasi (Supervised Learning) menggunakan ekosistem Scikit-learn, NumPy, dan Pandas.

---

## 1. Lingkungan Eksekusi & Prasyarat

Seluruh skrip diuji dan diverifikasi menggunakan Python 3.12 dengan pustaka standar:
- `numpy >= 1.26.0`
- `pandas >= 2.2.0`
- `scikit-learn >= 1.4.0`

Dataset telemetri kerusakan peralatan industri (`equipment_fault_records.csv`) dibangkitkan secara deterministik menggunakan skrip `generate_dataset.py` dan disimpan di direktori `data/`.

---

## 2. Struktur Berkas & Ringkasan Skrip

| No | Berkas Skrip | Fokus Materi & Metodologi | Status Eksekusi |
|---|---|---|---|
| 00 | `generate_dataset.py` | Pembangkitan dataset kerusakan mesin (350 baris, 6 fitur kontinu, target biner dan 4 kelas multikelas imbalanced). | Terverifikasi (Exit 0) |
| 01 | `binary_logistic_regression.py` | Regresi Logistik biner ($y \in \{0, 1\}$), fungsi Sigmoid, perumusan Log-Odds, dan rasio odds $\exp(w)$. | Terverifikasi (Exit 0) |
| 02 | `multiclass_logistic_regression.py` | Klasifikasi multikelas 4 tipe kerusakan via fungsi Softmax, evaluasi matriks konfusi 4x4, Macro vs Weighted Average. | Terverifikasi (Exit 0) |
| 03 | `gaussian_naive_bayes.py` | Gaussian Naive Bayes pada fitur kontinu, estimasi prior $P(y)$, rata-rata $\mu$, dan varians $\sigma^2$ per kelas. | Terverifikasi (Exit 0) |
| 04 | `multinomial_naive_bayes.py` | Multinomial Naive Bayes pada data diskrit, pembuktian bahaya probabilitas nol dan solusi Laplace Smoothing ($\alpha=1.0$). | Terverifikasi (Exit 0) |
| 05 | `confusion_matrix_metrics.py` | Perhitungan manual vs API untuk TP, FP, TN, FN, Akurasi, Presisi, Recall, Spesifisitas, FPR, F1-Score, dan aturan urutan label. | Terverifikasi (Exit 0) |
| 06 | `classification_threshold.py` | Penyesuaian ambang keputusan ($T \in [0.10, 0.90]$) pada Validation Set, analisis trade-off biaya FP vs FN. | Terverifikasi (Exit 0) |
| 07 | `roc_auc_analysis.py` | Kurva ROC (TPR vs FPR), pembuktian interpretasi probabilistik Wilcoxon-Mann-Whitney, bahaya prediksi biner diskrit. | Terverifikasi (Exit 0) |
| 08 | `precision_recall_analysis.py` | Kurva Precision-Recall dan Average Precision (AP), keunggulan kurva PR pada ketidakseimbangan kelas ekstrem. | Terverifikasi (Exit 0) |
| 09 | `classification_model_comparison.py` | Benchmark 4 model klasifikasi dengan 5-Fold Stratified Cross-Validation dan evaluasi akhir pada Test Set. | Terverifikasi (Exit 0) |

---

## 3. Hasil Pengujian Numerik Terverifikasi

Seluruh nilai di bawah ini dihasilkan secara aktual dari eksekusi skrip:

```text
1. BINARY LOGISTIC REGRESSION (binary_logistic_regression.py):
   - Odds Ratio Terbesar : vibration_rms (w=1.7575, exp(w)=5.7979x kenaikan risiko)
   - Verifikasi Sigmoid  : Manual=0.9672 | API predict_proba=0.9672 (Identik 100%)
   - Test Accuracy       : 100.00%
   - Test ROC-AUC        : 1.0000

2. MULTICLASS DIAGNOSIS 4 KELAS (multiclass_logistic_regression.py):
   - Test Accuracy       : 98.57% (69/70 benar)
   - Macro Avg F1        : 0.9530 (Memberikan bobot setara pada Electrical Surge yang langka)
   - Weighted Avg F1     : 0.9860 (Membobot berdasarkan prevalensi populasi kelas)

3. GAUSSIAN NAIVE BAYES (gaussian_naive_bayes.py):
   - Prior Probability   : Normal = 69.3% | Fault = 30.7%
   - Rata-rata Suhu (mu) : Normal = 55.10 °C | Fault = 74.07 °C
   - Test Accuracy       : 100.00% | Test F1 = 1.0000

4. MULTINOMIAL NB & LAPLACE SMOOTHING (multinomial_naive_bayes.py):
   - Tanpa Smoothing (alpha=1e-9) : Accuracy = 97.14% | F1 = 0.9524
   - Laplace Standar (alpha=1.0)  : Accuracy = 98.57% | F1 = 0.9767 (Kinerja optimal dan aman)

5. CONFUSION MATRIX VERIFICATION (confusion_matrix_metrics.py):
   - True Negatives (TN) : 48  | False Positives (FP) : 0
   - False Negatives (FN): 0   | True Positives (TP)  : 22
   - Verifikasi Rumus    : Accuracy, Precision, Recall, Specificity, F1-Score identik 100%.

6. BENCHMARK AKHIR (classification_model_comparison.py):
   - Dummy Baseline (Majority) : CV F1 = 0.0000 | Test Recall = 0.0000 | Akurasi Semu = 68.57%
   - Logistic Regression (Std) : CV F1 = 1.0000 | Test Recall = 1.0000 | Test ROC-AUC = 1.0000
   - Gaussian Naive Bayes      : CV F1 = 1.0000 | Test Recall = 1.0000 | Test ROC-AUC = 1.0000
```

---

## 4. Cara Menjalankan

Dari direktori root proyek:
```bash
# Mengaktifkan virtual environment
.\.venv\Scripts\Activate.ps1

# Menjalankan seluruh rangkaian skrip
python content/level-2-machine-learning/chapter-14/code/binary_logistic_regression.py
python content/level-2-machine-learning/chapter-14/code/multiclass_logistic_regression.py
python content/level-2-machine-learning/chapter-14/code/classification_model_comparison.py
```

