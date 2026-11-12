# Panduan Praktik Laboratorium — Chapter 14: Supervised Learning — Classification Algorithms

Selamat datang di laboratorium **Chapter 14: Supervised Learning — Classification Algorithms**. Praktikum ini memandu Anda menguasai formulasi matematis, pemodelan probabilistik, kalibrasi ambang batas keputusan, dan evaluasi diagnostik model klasifikasi biner dan multikelas pada data sensor industri.

---

## 1. Tujuan Praktik

Setelah menyelesaikan seluruh rangkaian praktikum laboratorium mandiri ini, Anda diharapkan mampu:
1. Mengimplementasikan Binary Logistic Regression, memahami fungsi Sigmoid, perumusan Log-Odds, dan menginterpretasikan rasio odds $\exp(w)$ dari koefisien model.
2. Membangun model klasifikasi multikelas menggunakan fungsi Softmax dan menganalisis perbedaan antara *Macro Average* dan *Weighted Average*.
3. Menerapkan Gaussian Naive Bayes pada fitur numerik kontinu dan mengekstrak prior probability serta likelihood parameter ($\mu, \sigma^2$).
4. Mengimplementasikan Multinomial Naive Bayes pada data diskrit dan membuktikan bagaimana Laplace Smoothing ($\alpha = 1.0$) mencegah bencana probabilitas nol (*zero-probability trap*).
5. Menghitung dan memvalidasi seluruh elemen Confusion Matrix (TP, FP, TN, FN) serta metrik Akurasi, Presisi, Recall, Spesifisitas, dan F1-Score secara mandiri.
6. Mengoptimasi ambang batas keputusan (*Decision Threshold*) pada Validation Set berdasarkan matriks kerugian biaya bisnis operasional.
7. Menganalisis kurva Receiver Operating Characteristic (ROC) dan membuktikan mengapa evaluasi ROC-AUC memerlukan skor probabilitas kontinu, bukan prediksi biner diskrit.
8. Menganalisis kurva Precision-Recall (PR) dan Average Precision (AP) untuk mengevaluasi data dengan ketidakseimbangan kelas (*class imbalance*).
9. Melakukan benchmark komparatif multi-model menggunakan 5-Fold Stratified Cross-Validation dan evaluasi akhir pada *unseen test data*.

---

## 2. Struktur Direktori Praktikum

Seluruh kode sumber mandiri, dataset telemetri kerusakan mesin, dan notebook interaktif tersimpan dalam struktur berikut:

```text
content/level-2-machine-learning/chapter-14/
├── code/
│   ├── data/
│   │   └── equipment_fault_records.csv# Dataset telemetri kerusakan mesin (350 baris, 6 fitur)
│   ├── generate_dataset.py            # Skrip pembangkit dataset klasifikasi sintetis
│   ├── binary_logistic_regression.py  # Regresi logistik biner, sigmoid & odds ratio
│   ├── multiclass_logistic_regression.py # Klasifikasi multikelas 4 tipe via Softmax
│   ├── gaussian_naive_bayes.py        # Gaussian Naive Bayes pada fitur kontinu
│   ├── multinomial_naive_bayes.py     # Multinomial NB diskrit & Laplace smoothing
│   ├── confusion_matrix_metrics.py    # TP/FP/TN/FN, akurasi, presisi, recall, spesifisitas
│   ├── classification_threshold.py    # Tuning decision threshold pada Validation Set
│   ├── roc_auc_analysis.py            # Kurva ROC, ROC-AUC, dan interpretasi Wilcoxon
│   ├── precision_recall_analysis.py   # Kurva PR & Average Precision pada data imbalanced
│   ├── classification_model_comparison.py # Benchmark 4 model dengan 5-Fold Stratified CV
│   └── README.md                      # Dokumentasi katalog skrip & hasil pengujian aktual
├── notebooks/
│   └── chapter_14_companion.ipynb     # Interactive Jupyter Companion Notebook (Format v4)
├── practice.md                        # Panduan praktikum laboratorium (berkas ini)
├── index.md                           # Modul teori komprehensif (Single Source of Truth)
├── slides/
│   └── deck.json                      # Slide presentasi pengajaran Reveal.js (32 slide)
├── audit.md                           # Laporan audit mutu publikasi bab
└── metadata.json                      # Registri metadata resmi kurikulum
```

---

## 3. Prasyarat Lingkungan & Persiapan

Praktikum ini menggunakan lingkungan virtual Python terpadu (`.venv`):
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

# 3. Pastikan dependensi siap
python -c "import sklearn, pandas, numpy; print('Environment Chapter 14 Siap!')"
```

---

## 4. Rangkaian Praktikum Laboratorium Terpandu

Lakukan praktikum ini langkah demi langkah sesuai urutan skrip modular di bawah ini:

### Lab 1: Binary Logistic Regression, Sigmoid & Odds Ratio
Jalankan skrip regresi logistik biner:
```powershell
python content/level-2-machine-learning/chapter-14/code/binary_logistic_regression.py
```
**Analisis Kunci:**
- Amati koefisien logit dan rasio odds: variabel getaran (`vibration_rms`) memiliki koefisien tertinggi $w = 1.7575$ yang berarti setiap kenaikan 1 satuan getaran melipatgandakan risiko kerusakan mesin sebesar **$5.80\times$**.
- Buktikan perhitungan manual Sigmoid $P(y=1|x) = 1 / (1 + e^{-z})$ menghasilkan nilai probabilitas yang identik 100% dengan fungsi `predict_proba()`.

### Lab 2: Multiclass Diagnosis via Softmax Function
Jalankan skrip diagnosis multi-kelas 4 status:
```powershell
python content/level-2-machine-learning/chapter-14/code/multiclass_logistic_regression.py
```
**Analisis Kunci:**
- Amati matriks konfusi 4x4 yang memisahkan status Normal, Kerusakan Bantalan, Overheating, dan Lonjakan Listrik.
- Pahami perbedaan antara **Macro Average F1 ($0.9530$)** yang memberi bobot setara pada kelas langka Electrical Surge, dan **Weighted Average F1 ($0.9860$)** yang membobot berdasarkan proporsi sampel dominan.

### Lab 3: Gaussian Naive Bayes pada Fitur Kontinu
Jalankan skrip Gaussian Naive Bayes:
```powershell
python content/level-2-machine-learning/chapter-14/code/gaussian_naive_bayes.py
```
**Analisis Kunci:**
- Telusuri parameter yang dipelajari model: prior probability $P(\text{Normal}) = 69.3\%$ dan $P(\text{Fault}) = 30.7\%$, serta rata-rata suhu mesin normal ($55.1^\circ\text{C}$) vs mesin rusak ($74.1^\circ\text{C}$).
- Evaluasi bagaimana asumsi independensi fitur (*naive assumption*) tetap mampu menghasilkan klasifikasi yang sangat akurat dengan komputasi minimal.

### Lab 4: Multinomial Naive Bayes & Laplace Smoothing
Jalankan skrip Multinomial Naive Bayes:
```powershell
python content/level-2-machine-learning/chapter-14/code/multinomial_naive_bayes.py
```
**Analisis Kunci:**
- Amati proses diskritisasi sensor menjadi 5 level alarm bertingkat.
- Bandingkan performa model tanpa smoothing ($\alpha = 10^{-9}$, F1 = $0.9524$) versus dengan Laplace Smoothing ($\alpha = 1.0$, F1 = $0.9767$).
- Pahami peran Laplace Smoothing dalam meniadakan probabilitas nol yang dapat menggagalkan perhitungan posterior.

### Lab 5: Verifikasi Matriks Konfusi & Metrik Klasifikasi
Jalankan skrip verifikasi metrik:
```powershell
python content/level-2-machine-learning/chapter-14/code/confusion_matrix_metrics.py
```
**Analisis Kunci:**
- Hitung elemen True Positive ($22$), False Positive ($0$), True Negative ($48$), dan False Negative ($0$).
- Buktikan rumus manual Akurasi, Presisi, Recall, Spesifisitas, dan F1-Score cocok sempurna dengan Scikit-learn API.
- Waspadai aturan urutan label (*label ordering*) agar tidak salah menafsirkan susunan sel pada matriks konfusi.

### Lab 6: Decision Threshold Selection & Analisis Biaya
Jalankan skrip tuning ambang keputusan:
```powershell
python content/level-2-machine-learning/chapter-14/code/classification_threshold.py
```
**Analisis Kunci:**
- Amati pergeseran nilai Presisi dan Recall saat threshold $T$ diturunkan dari $0.90$ ke $0.10$.
- Diskusikan skenario keselamatan pabrik: jika biaya membiarkan mesin meledak (False Negative) bernilai puluhan ribu dolar, memilih threshold rendah ($T = 0.20$) adalah keputusan teknis yang sangat bijak.

### Lab 7: Analisis Kurva ROC & Skor ROC-AUC
Jalankan skrip analisis ROC-AUC:
```powershell
python content/level-2-machine-learning/chapter-14/code/roc_auc_analysis.py
```
**Analisis Kunci:**
- Pahami pembuktian probabilistik Wilcoxon-Mann-Whitney: nilai ROC-AUC adalah peluang model memberikan skor probabilitas lebih tinggi pada sampel positif acak dibanding sampel negatif acak.
- Buktikan bahaya fatal memasukkan prediksi biner diskrit (0/1) ke dalam `roc_auc_score` yang akan mereduksi kurva ROC menjadi hanya satu titik tunggal.

### Lab 8: Analisis Kurva Precision-Recall pada Data Tidak Seimbang
Jalankan skrip analisis PR curve:
```powershell
python content/level-2-machine-learning/chapter-14/code/precision_recall_analysis.py
```
**Analisis Kunci:**
- Bandingkan baseline acak ROC ($0.5000$) versus baseline acak PR ($0.3143$ yang persis sama dengan prevalensi kelas positif).
- Pahami mengapa kurva PR jauh lebih sensitif dan informatif daripada ROC saat mendeteksi anomali atau kegagalan yang sangat langka.

### Lab 9: Benchmark Komparatif Multi-Model
Jalankan skrip komparasi menyeluruh:
```powershell
python content/level-2-machine-learning/chapter-14/code/classification_model_comparison.py
```
**Analisis Kunci:**
- Amati bagaimana Dummy Baseline (Majority) memperoleh akurasi semu sebesar $68.57\%$, namun memiliki **Recall = 0.0000** (sama sekali tidak mendeteksi kerusakan).
- Evaluasi keunggulan Logistic Regression dan Gaussian Naive Bayes yang mencapai Recall $100\%$ dan ROC-AUC mendekati $1.0000$.

---

## 5. Eksplorasi Mandiri & Tantangan Analisis

Untuk memperdalam keahlian klasifikasi Anda, selesaikan 3 tantangan berikut:

### Tantangan 1: Analisis Kurva Kalibrasi Probabilitas (Reliability Diagram)
Gunakan modul `sklearn.calibration.calibration_curve`:
1. Buat kurva keandalan (*reliability diagram*) untuk Logistic Regression dan Gaussian Naive Bayes.
2. Analisis apakah probabilitas keluaran model mencerminkan frekuensi empiris sebenarnya (apakah sampel dengan skor $0.80$ benar-benar memiliki peluang rusak $80\%$).
3. Uji penerapan `CalibratedClassifierCV` menggunakan metode sigmoid (Platt scaling) atau isotonic regression.

### Tantangan 2: Optimasi Ambang Batas Berbasis Kurva Biaya (Cost Curve)
Pada `classification_threshold.py`:
1. Definisikan matriks biaya riil: Biaya False Negative = Rp 50.000.000, Biaya False Positive = Rp 500.000, Biaya True Positive = Rp 0, Biaya True Negative = Rp 0.
2. Hitung total kerugian finansial pada seluruh rentang threshold $T \in [0.05, 0.95]$.
3. Tentukan titik threshold $T^*$ yang menghasilkan kerugian finansial total paling minimum.

### Tantangan 3: Penanganan Data Campuran Menggunakan Mixed Naive Bayes
Jika data memiliki kombinasi fitur kontinu dan fitur diskrit kategorikal:
1. Bangun pipeline pra-pemrosesan yang memisahkan fitur kontinu untuk `GaussianNB` dan fitur diskrit untuk `CategoricalNB`.
2. Gabungkan prediksi log-likelihood dari kedua model untuk menghasilkan klasifikasi akhir.

---

## 6. Rubrik Penilaian Praktikum Laboratorium (100 Poin)

| Kriteria Penilaian | Bobot | Deskripsi Bukti Capaian |
|---|---|---|
| **Formulasi Logistik & Interpretasi Odds Ratio** | 20 Poin | Mampu mengimplementasikan Binary Logistic Regression, membuktikan rumus Sigmoid, dan menginterpretasikan pengaruh koefisien secara tepat. |
| **Implementasi Multikelas & Pemahaman Averaging** | 20 Poin | Mengimplementasikan fungsi Softmax untuk klasifikasi 4 status, menganalisis matriks konfusi 4x4, serta membedakan Macro dan Weighted average. |
| **Penguasaan Naive Bayes & Laplace Smoothing** | 20 Poin | Menerapkan Gaussian NB dan Multinomial NB, menjelaskan asumsi independensi fitur, serta membuktikan manfaat Laplace Smoothing dalam mencegah zero probability. |
| **Kebenaran Metrik & Diagnostik Threshold** | 20 Poin | Menghitung seluruh elemen Confusion Matrix secara manual dan via API, serta melakukan optimasi ambang batas keputusan (*threshold tuning*) secara metodologis. |
| **Analisis Kurva ROC/PR & Benchmark Pipeline** | 20 Poin | Membedakan kondisi penggunaan ROC-AUC dan PR-AUC, menyajikan benchmark multi-model dengan 5-Fold Stratified CV, dan mengevaluasi bahaya akurasi semu baseline. |

---

## 7. Pelaporan Hasil Praktikum

Dokumentasikan hasil eksperimen mandiri, tabel ringkasan metrik aktual, serta analisis diagnostik ke dalam laporan laboratorium format Markdown:
`content/level-2-machine-learning/chapter-14/lab_report_<NIM>.md`
Pastikan seluruh nilai metrik evaluasi didasarkan pada eksekusi nyata skrip Python.

