# Panduan Praktikum Terbimbing: Model Evaluation, Tuning & Explainability

Selamat datang di laboratorium praktikum Chapter 19. Modul praktikum ini dirancang untuk memberikan pengalaman komputasional langsung dalam mengimplementasikan evaluasi model yang ketat, validasi silang khusus, penyetelan hyperparameter bebas kebocoran data (*zero data leakage*), kurva diagnostik pembelajaran, kalibrasi probabilitas, serta introspeksi keputusan (*model explainability*) pada data armada sensor IoT.

---

## 1. Lingkungan Kerja & Prapemrosesan

Pastikan lingkungan virtual Python Anda telah terkonfigurasi dengan pustaka saintifik standar:

```bash
# Mengaktifkan environment (PowerShell)
.venv\Scripts\Activate.ps1

# Menjalankan pembangkit dataset telemetri armada sensor (jika belum dijalankan)
python content/level-2-machine-learning/chapter-19/code/generate_dataset.py
```

Dataset tersimpan di `content/level-2-machine-learning/chapter-19/code/data/sensor_fleet_eval_telemetry.csv` dengan struktur:
- **Ukuran**: 500 observasi telemetri across 10 perangkat unik (`device_id`).
- **Siklus Waktu**: 50 siklus sequential per perangkat (`cycle_id`).
- **Fitur Kontinu**: `vibration_amplitude_g`, `operating_temp_c`, `pressure_kpa`, `power_draw_kw`, `acoustic_noise_db`, `operating_hours`.
- **Target Regresi**: `failure_risk_score` (Skor kontinu rentang 0 s.d. 100).
- **Target Klasifikasi**: `failure_status` (0 = Normal, 1 = Fault).

---

## 2. Katalog Praktikum Terstruktur (12 Langkah Mandiri)

```
content/level-2-machine-learning/chapter-19/code/
├── generate_dataset.py                   # Pembangkit dataset telemetri multi-device
├── evaluation_metrics.py                 # Evaluasi komparatif metrik regresi & klasifikasi
├── cross_validation.py                   # Holdout vs K-Fold vs Stratified vs Repeated
├── group_time_validation.py              # GroupKFold (perangkat) & TimeSeriesSplit (siklus waktu)
├── grid_search.py                        # Penelusuran kisi multi-metrik (GridSearchCV)
├── randomized_search.py                  # Penelusuran acak spektrum kontinu (RandomizedSearchCV)
├── learning_curves.py                    # Diagnostik kurva pembelajaran (bias vs varians)
├── validation_curves.py                  # Diagnostik kurva validasi (sensitivitas hyperparameter)
├── calibration_analysis.py               # Analisis kalibrasi probabilitas & Brier score
├── permutation_importance.py             # Permutation feature importance vs MDI Gini
├── partial_dependence.py                 # Partial Dependence Plots (PDP) & konsep ICE
├── error_analysis.py                     # Analisis kasus False Negative & audit disparitas subgrup
└── model_evaluation_report.py            # Pembangkit laporan evaluasi menyeluruh reproducible
```

---

## 3. Langkah-demi-Langkah Pelaksanaan Praktikum

### Langkah 1: Eksplorasi Metrik Regresi & Klasifikasi
Uji perbedaan perilaku matematis tiap metrik pada masalah pemeliharaan prediktif:
```bash
python content/level-2-machine-learning/chapter-19/code/evaluation_metrics.py
```
*Observasi Kunci*: Pada regresi, RMSE ($2.6751$) bernilai lebih tinggi dari MAE ($2.0350$) karena penalti kuadratik atas deviasi besar. Pada klasifikasi, Recall ($84.51\%$) dan F1 ($0.8163$) menjadi prioritas utama untuk mencegah kegagalan deteksi kerusakan kritis (False Negative).

### Langkah 2: Variabilitas Validasi Silang
Bandingkan kestabilan estimasi performa antara Holdout tunggal dengan K-Fold:
```bash
python content/level-2-machine-learning/chapter-19/code/cross_validation.py
```
*Observasi Kunci*: Holdout tunggal berfluktuasi tergantung pemilihan random seed. Stratified K-Fold menjaga proporsi kelas target identik persis di setiap fold, menghasilkan estimasi yang jauh lebih stabil.

### Langkah 3: Validasi Berkelompok & Kebocoran Temporal
Buktikan bahaya kebocoran kelompok perangkat (*device leakage*) dan kebocoran temporal:
```bash
python content/level-2-machine-learning/chapter-19/code/group_time_validation.py
```
*Observasi Kunci*: Random K-Fold menghasilkan estimasi akurasi optimis semu sebesar $82.40\%$. Saat menggunakan `GroupKFold` (di mana seluruh observasi dari satu `device_id` tidak pernah terbagi antara train dan val), akurasi turun ke $76.20\%$, merefleksikan performa generalisasi riil pada hardware baru.

### Langkah 4: Penelusuran Kisi (GridSearchCV) Multi-Metrik
Lakukan penyetelan hyperparameter sistematis di dalam lingkungan CV:
```bash
python content/level-2-machine-learning/chapter-19/code/grid_search.py
```
*Observasi Kunci*: Seleksi model menggunakan F1-score pada validasi menghasilkan model terbaik yang kemudian diuji pada test set terpisah (mencetak F1-score uji $0.8250$).

### Langkah 5: Penelusuran Acak (RandomizedSearchCV)
Uji efisiensi alokasi anggaran pencarian pada distribusi kontinu dan diskrit:
```bash
python content/level-2-machine-learning/chapter-19/code/randomized_search.py
```
*Observasi Kunci*: Hanya dengan 25 iterasi acak, RandomizedSearch menemukan setelan hyperparameter yang setara dengan penelusuran kisi penuh namun menghemat waktu komputasi hingga $60\%$.

### Langkah 6: Diagnostik Kurva Pembelajaran (Learning Curves)
Analisis pengaruh penambahan jumlah data latih terhadap skor pelatihan dan validasi:
```bash
python content/level-2-machine-learning/chapter-19/code/learning_curves.py
```
*Observasi Kunci*: Seiring bertambahnya data dari 60 ke 400 sampel, skor validasi terus naik dan selisih gap menyempit, membuktikan model tidak mengalami High Bias kaku.

### Langkah 7: Diagnostik Kurva Validasi (Validation Curves)
Periksa transisi dari underfitting ke overfitting pada rentang `max_depth`:
```bash
python content/level-2-machine-learning/chapter-19/code/validation_curves.py
```
*Observasi Kunci*: Kedalaman pohon 1-2 menghasilkan underfitting (F1 $< 70\%$). Puncak optimal dicapai pada depth 5-8. Kedalaman di atas 12 memicu overfitting (skor latih 100% namun validasi menurun).

### Langkah 8: Kalibrasi Probabilitas & Brier Score
Uji keandalan skor probabilitas model dan kalibrasi menggunakan metode Platt / Isotonik:
```bash
python content/level-2-machine-learning/chapter-19/code/calibration_analysis.py
```
*Observasi Kunci*: Model uncalibrated memiliki Brier score $0.18$. Kalibrasi sigmoid menekan Brier score ke $0.14$, membuat skor probabilitas model selaras dengan frekuensi empiris aktual.

### Langkah 9: Permutation Feature Importance
Bandingkan kepentingan fitur Gini bawaan dengan penurunan performa data uji:
```bash
python content/level-2-machine-learning/chapter-19/code/permutation_importance.py
```
*Observasi Kunci*: Getaran (*vibration*) dan suhu (*operating_temp*) adalah dua fitur paling kritis yang menyebabkan penurunan F1-score terbesar jika diacak.

### Langkah 10: Analisis Ketergantungan Parsial (PDP)
Visualisasikan efek marjinal non-linier dari prediktor kontinu:
```bash
python content/level-2-machine-learning/chapter-19/code/partial_dependence.py
```
*Observasi Kunci*: Terlihat titik jenuh (*tipping point*) di mana probabilitas kerusakan melonjak tajam begitu getaran melebihi $1.0\text{ g}$.

### Langkah 11: Analisis Kasus Kegagalan & Audit Subgrup
Bedah kasus False Negative dan residu galat:
```bash
python content/level-2-machine-learning/chapter-19/code/error_analysis.py
```
*Observasi Kunci*: Kasus False Negative umumnya terjadi pada kondisi perbatasan probabilitas ($p \approx 0.48$). Audit subgrup menunjukkan perangkat tertentu mengalami deviasi kalibrasi sensor.

### Langkah 12: Pembangkit Laporan Evaluasi Reproducible
Jalankan skrip akhir untuk merangkum seluruh hasil evaluasi ke dalam laporan audit:
```bash
python content/level-2-machine-learning/chapter-19/code/model_evaluation_report.py
```

---

## 4. Tabel Hasil Pengujian Empiris Terverifikasi

| Pengujian / Metrik | Parameter Konfigurasi | Nilai Terverifikasi | Interpretasi Sistem |
|---|---|---|---|
| **MAE Regresi** | Random Forest Regressor | **2.0350** | Deviasi rata-rata skor risiko 2 poin |
| **RMSE Regresi** | Random Forest Regressor | **2.6751** | Penalti kuadratik atas deviasi besar |
| **R² Regresi** | Random Forest Regressor | **0.9869** (98.69%) | Menjelaskan hampir seluruh varians target |
| **Akurasi Klasifikasi** | Random Forest Classifier | **78.40%** | Akurasi standar pada test set |
| **Recall Klasifikasi** | Random Forest Classifier | **84.51%** | Kemampuan mendeteksi kerusakan riil |
| **F1-Score Klasifikasi** | Random Forest Classifier | **0.8163** | Keseimbangan harmonik Precision-Recall |
| **ROC-AUC Score** | Random Forest Classifier | **0.8589** | Daya diskriminasi threshold independen |
| **PR-AUC Score** | Random Forest Classifier | **0.8906** | Luas kurva Precision-Recall (baseline 0.57) |
| **Akurasi Random K-Fold** | 5-Fold Tanpa Grup | **82.40%** | Optimisme semu akibat bocornya ID perangkat |
| **Akurasi Group K-Fold** | 5-Fold Berbasis device_id | **76.20%** | Generalisasi murni pada hardware baru |
| **Brier Score (Calibrated)** | Platt Sigmoid Calibration | **0.1412** | Probabilitas terkalibrasi secara handal |

---

## 5. Tantangan Terbimbing (*Guided Challenges*)

### Tantangan A: Menguji Dampak Threshold Tuning Terhadap Biaya Operasional
Secara default, pengklasifikasi biner menggunakan ambang batas probabilitas $\theta = 0.50$. Modifikasi skrip `evaluation_metrics.py` untuk menggeser ambang batas ke $\theta = 0.35$ dan $\theta = 0.65$. Amati bagaimana perubahan ini menggeser trade-off antara False Positive dan False Negative. Pada skenario di mana biaya perbaikan reaktif pasca-kerusakan adalah \$10.000 sedangkan biaya inspeksi preventif hanya \$500, ambang batas manakah yang menghasilkan ekspektasi total biaya terendah?

### Tantangan B: Deteksi Out-of-Distribution (OOD) Sensor Telemetry
Buat fungsi pendek untuk memeriksa apakah data sensor baru berada di luar batas operasional historis (misal suhu $> 85^\circ\text{C}$ atau getaran $> 2.5\text{ g}$). Tunjukkan mengapa model yang sangat percaya diri ($p = 0.99$) tetap tidak boleh dipercaya saat menerima input OOD tanpa peringatan anomali.

### Tantangan C: Menghindari Kebocoran Data pada Feature Selection
Modifikasi pipeline pada `grid_search.py` untuk memasukkan langkah seleksi fitur (misal `SelectKBest(k=4)`). Buktikan bahwa langkah seleksi fitur WAJIB berada di dalam objek `Pipeline` Scikit-Learn agar informasi fitur tidak bocor dari fold validasi ke fold latih.

---

## 6. Rubrik Penilaian Praktikum (Skala 100 Poin)

| Kriteria Penilaian | Bobot | Sangat Baik (85–100) | Baik (70–84) | Cukup (55–69) | Kurang (< 55) |
|---|---|---|---|---|---|
| **Eksekusi & Integritas Kode** | 25% | Seluruh 12 skrip dieksekusi tanpa galat; output numerik presisi sesuai tabel verifikasi. | 10–11 skrip berjalan lancar dengan sedikit variasi numerik yang dapat dijelaskan. | 7–9 skrip berjalan; terdapat galat runtime yang belum ditangani tuntas. | Kurang dari 7 skrip berjalan; manipulasi data salah. |
| **Validasi Silang & Anti-Leakage** | 25% | Mengimplementasikan Stratified, Group, dan TimeSeries Split secara tepat dengan isolasi data uji mutlak. | Menerapkan K-Fold standar namun penjelasan pencegahan kebocoran kelompok kurang lengkap. | Menggunakan satu holdout acak tanpa pembenaran metodologis. | Terjadi kebocoran data (*data leakage*) antara latih dan uji. |
| **Hyperparameter Tuning & Diagnostik** | 25% | Menganalisis kurva pembelajaran dan kurva validasi secara presisi; menyetel model di dalam lingkungan CV. | Menjalankan Grid / Randomized Search namun kurang mendalam dalam interpretasi kurva diagnostik. | Menyetel parameter secara manual tanpa visualisasi diagnostik. | Melakukan tuning langsung pada test set. |
| **Explainability & Analisis Kegagalan** | 25% | Mendiagnosis kasus False Negative kritis, mengevaluasi disparitas subgrup, dan menginterpretasikan PDP/Permutation secara bertanggung jawab. | Menampilkan ranking fitur namun tidak membedah kasus kegagalan individual secara mendalam. | Hanya menyajikan angka feature importance bawaan (MDI) tanpa audit residu. | Tidak melakukan error analysis atau explainability sama sekali. |

