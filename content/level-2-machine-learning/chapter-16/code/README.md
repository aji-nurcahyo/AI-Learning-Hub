# Katalog Praktikum & Eksperimen Kode — Chapter 16: Support Vector Machines & k-Nearest Neighbors

Direktori ini memuat seluruh implementasi kode Python mandiri (*runnable*), skrip pembangkit data deterministik status jaringan sensor nirkabel (*Wireless Sensor Networks* / WSN), eksplorasi geometris batas margin, fungsi kernel (Linear, Polinomial, RBF), metrik jarak KNN (Euclidean, Manhattan, Minkowski), serta analisis komparasi komprehensif SVM vs KNN.

---

## 1. Lingkungan Eksekusi & Kebutuhan Dependensi

Seluruh skrip dikembangkan dan diverifikasi menggunakan Python 3.12 pada lingkungan virtual `.venv`:

```bash
# Aktifkan virtual environment
.\.venv\Scripts\activate

# Dependensi utama (terinstal):
# - numpy >= 1.26.0
# - pandas >= 2.2.0
# - scikit-learn >= 1.4.0
# - matplotlib >= 3.8.0
```

---

## 2. Struktur Berkas & Deskripsi Modul

| Berkas Skrip | Deskripsi Algoritmik & Pembelajaran | Keluaran Utama / Metrik |
|---|---|---|
| `generate_dataset.py` | Pembangkit data sintetis telemetri jaringan sensor nirkabel deterministik (360 baris) | `data/wireless_sensor_network_state.csv` (NORMAL, WARNING, CRITICAL) |
| `linear_svm.py` | Linear SVM dengan standarisasi, inspeksi support vectors, dan batas keputusan linier | Akurasi Uji, proporsi Support Vectors, Confusion Matrix 3x3 |
| `hard_soft_margin_demo.py` | Demonstrasi Hard-Margin vs Soft-Margin dan peran variabel slack ($\xi_i$) | Bukti ketidakfisibelan hard-margin pada data ber-derau |
| `svm_regularization.py` | Eksperimen variasi parameter regularisasi $C \in [0.001, 1000]$ | Analisis lebar margin, jumlah SV, dan akurasi generalisasi |
| `svm_kernel_comparison.py` | Komparasi fungsi kernel: Linear, Polynomial (derajat 3), RBF, dan Sigmoid | 5-Fold Stratified CV, akurasi uji, dan latensi komputasi |
| `rbf_gamma_analysis.py` | Analisis radius pengaruh support vector via parameter $\gamma \in [0.001, 100]$ | Spektrum bias-varians RBF (dari underfitting hingga overfitting) |
| `knn_classifier.py` | Implementasi dasar k-Nearest Neighbors dengan pencarian tetangga terdekat | Laporan klasifikasi presisi, recall, dan matriks konfusi |
| `knn_distance_metrics.py` | Perhitungan manual dan evaluasi metrik jarak: Euclidean ($L_2$), Manhattan ($L_1$), Chebyshev ($L_\infty$), Minkowski ($p=3$) | Analisis geometri ruang metrik dan akurasi model |
| `knn_k_parameter.py` | Eksperimen variasi nilai $k \in [1, 45]$ serta perbandingan Uniform vs Distance Weighting | Bukti hafalan $k=1$ (100% latih) vs kehalusan batas $k$ besar |
| `feature_scaling_effect.py` | Demonstrasi kritis dampak ketiadaan standarisasi fitur pada SVM dan KNN | Bukti dominasi fitur berskala besar (Latensi vs Voltase) |
| `svm_knn_comparison.py` | Benchmark komparatif 6 arsitektur SVM vs KNN pada 5-Fold Stratified CV | Tabel akurasi, Macro F1, waktu latih (*fit*), dan latensi inferensi |

---

## 3. Dataset Telemetri Status Jaringan Sensor Nirkabel

Dataset disimpan pada `data/wireless_sensor_network_state.csv` dengan metadata:
- **Jumlah Baris**: 360 sampel telemetri simpul sensor.
- **Fitur Masukan (6 Prediktor)**:
  1. `signal_strength_dbm`: Kekuatan sinyal radio RSSI (-85.0 s/d -35.0 dBm).
  2. `temperature_c`: Suhu operasional lingkungan simpul (22.0 s/d 58.0 °C).
  3. `humidity_pct`: Kelembaban relatif udara sekitar (35.0% s/d 90.0%).
  4. `packet_loss_pct`: Persentase paket data yang hilang saat transmisi (0.0% s/d 30.0%).
  5. `latency_ms`: Waktu tunda transmisi pulang-pergi (*Round Trip Time*) (10.0 s/d 400.0 ms).
  6. `sensor_voltage_v`: Tegangan catu daya baterai lithium simpul (2.1 s/d 3.8 V).
- **Target Klasifikasi**:
  - `node_status`: Kategori diskret (`NORMAL`: 97 sampel, `WARNING`: 123 sampel, `CRITICAL`: 140 sampel).
  - `status_code`: Representasi numerik terenkode (`0`, `1`, `2`).

---

## 4. Cara Menjalankan Skrip

Jalankan skrip secara individual melalui terminal:

```bash
# Menjalankan pembuktian dampak feature scaling
python content/level-2-machine-learning/chapter-16/code/feature_scaling_effect.py

# Menjalankan perbandingan kernel SVM
python content/level-2-machine-learning/chapter-16/code/svm_kernel_comparison.py

# Menjalankan komparasi komprehensif SVM vs KNN
python content/level-2-machine-learning/chapter-16/code/svm_knn_comparison.py
```

