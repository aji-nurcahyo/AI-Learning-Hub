# Panduan Praktikum Laboratorium — Chapter 16: Support Vector Machines & k-Nearest Neighbors

Panduan laboratorium terpadu ini menyajikan petunjuk langkah demi langkah (*step-by-step*) untuk mengeksplorasi, melatih, mentransformasi kernel, dan mengevaluasi model Support Vector Machines (SVM) serta k-Nearest Neighbors (KNN) menggunakan Python dan Scikit-learn pada data telemetri jaringan sensor nirkabel (*Wireless Sensor Networks* / WSN).

---

## 1. Sasaran Pembelajaran Praktikum

Setelah menyelesaikan sesi laboratorium ini, mahasiswa diharapkan mampu:
1. Memahami secara intuitif konsep **Hiperbidang Pemisah Margin Maksimum** (*Maximum Margin Separating Hyperplane*) dan mengidentifikasi **Support Vectors** pada ruang fitur.
2. Membedakan perilaku **Hard-Margin SVM** pada data terpisah linier murni dengan **Soft-Margin SVM** berbasis variabel slack ($\xi_i$) dan penalti regularisasi $C$.
3. Mengaplikasikan **Kernel Trick** (Linear, Polynomial, dan Gaussian RBF) untuk memisahkan data non-linier tanpa transformasi eksplisit.
4. Menganalisis peran parameter $\gamma$ (*gamma*) pada fungsi basis radial (RBF) dalam mengendalikan radius pengaruh setiap support vector.
5. Mengimplementasikan algoritma **k-Nearest Neighbors (KNN)** berbasis memori (*instance-based lazy learning*) menggunakan berbagai metrik jarak ($L_1, L_2, L_\infty, L_p$).
6. Menganalisis trade-off bias-varians pada variasi nilai $k$ serta skema pembobotan suara (*Uniform* vs *Distance-Weighted*).
7. Membuktikan secara empiris dampak fatal pengabaian **Feature Scaling** pada model berbasis jarak dan geometri.
8. Melakukan benchmark komparatif komprehensif antara SVM dan KNN dalam hal akurasi, Macro F1, waktu pelatihan, dan latensi inferensi.

---

## 2. Persiapan Lingkungan & Struktur Proyek

Pastikan Anda berada di direktori akar repositori dan mengaktifkan virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\activate

# Pindah ke direktori kode Chapter 16
cd content/level-2-machine-learning/chapter-16/code
```

Verifikasi ketersediaan modul utama:
```python
import numpy as np
import pandas as pd
import sklearn
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__} | Scikit-learn: {sklearn.__version__}")
```

---

## 3. Langkah-Langkah Praktikum Terpandu

### Langkah 1: Pembangkitan Data Telemetri Jaringan Sensor (`generate_dataset.py`)
Jalankan skrip pembangkit data untuk membuat 360 sampel telemetri simpul sensor nirkabel:
```bash
python generate_dataset.py
```
- **Target**: `node_status` (`NORMAL`: 97, `WARNING`: 123, `CRITICAL`: 140 sampel).
- **Fitur**: `signal_strength_dbm`, `temperature_c`, `humidity_pct`, `packet_loss_pct`, `latency_ms`, `sensor_voltage_v`.

### Langkah 2: Pembuktian Fatalitas Ketiadaan Standarisasi Fitur (`feature_scaling_effect.py`)
Pahami mengapa model geometris sangat sensitif terhadap skala fitur:
```bash
python feature_scaling_effect.py
```
*Temuan Utama:*
- Fitur Latensi memiliki rentang $75.4$ ms, sedangkan Voltase hanya $1.2$ V.
- Tanpa scaling, kuadrat selisih latensi mendominasi voltase puluhan ribu kali lipat dalam perhitungan jarak Euclidean.
- Akurasi RBF SVM anjlok menjadi **58.89%** tanpa scaling, dan melonjak drastis ke **70.00%** saat StandardScaler diaplikasikan!

### Langkah 3: Pelatihan Linear SVM & Inspeksi Support Vectors (`linear_svm.py`)
Latih Linear SVM dalam Scikit-learn Pipeline bebas kebocoran:
```bash
python linear_svm.py
```
*Hasil Verifikasi:*
- Akurasi Uji Linear SVM: **62.22%**.
- Total Support Vectors: $171$ dari $270$ sampel data latih ($63.3\%$). Sampel-sampel inilah yang secara eksklusif menentukan orientasi hiperbidang pemisah.

### Langkah 4: Demonstrasi Batas Keras vs Batas Lunak (`hard_soft_margin_demo.py`)
Buktikan ketidakmampuan Hard-Margin menangani data dengan sampel tumpang tindih (*overlap*):
```bash
python hard_soft_margin_demo.py
```
*Temuan Konseptual:*
- Hard-Margin ($C \to \infty$) hanya fisibel pada data terpisah linier sempurna.
- Soft-Margin memperkenalkan variabel slack $\xi_i \ge 0$ yang memungkinkan pelanggaran batas terkontrol untuk mempertahankan margin yang lebar dan stabil.

### Langkah 5: Analisis Pengaruh Parameter Regularisasi C (`svm_regularization.py`)
Eksplorasi spektrum penalti pelanggaran margin:
```bash
python svm_regularization.py
```
- Nilai $C = 0.001$: Margin sangat longgar, banyak support vector ($92.6\%$), model underfitting (Akurasi Uji $47.78\%$).
- Nilai $C = 1.0$: Keseimbangan optimal, akurasi uji **62.22%**.
- Nilai $C = 1000.0$: Penalti ketat, margin menyempit, risiko overfitting pada noise lokal.

### Langkah 6: Komparasi Fungsi Kernel SVM (`svm_kernel_comparison.py`)
Bandingkan kemampuan memetakan non-linearitas:
```bash
python svm_kernel_comparison.py
```
*Hasil 5-Fold Stratified Cross-Validation:*
- **RBF Kernel**: Unggul dengan akurasi CV **67.78% ± 6.58%** dan Akurasi Uji **70.00%**.
- Linear Kernel: Akurasi CV $59.26\% \pm 4.83\%$ (Akurasi Uji $62.22\%$).
- Polynomial Kernel ($d=3$): Akurasi CV $53.33\% \pm 4.60\%$ (Akurasi Uji $60.00\%$).

### Langkah 7: Analisis Parameter Gamma pada RBF Kernel (`rbf_gamma_analysis.py`)
Pahami bagaimana radius fungsi lonceng Gaussian mengontrol kompleksitas:
```bash
python rbf_gamma_analysis.py
```
- $\gamma = 0.001$: Lonceng sangat datar, model bertindak seperti linear classifier datar.
- $\gamma = \text{scale} \approx 0.16$: Kontur non-linier halus, akurasi uji optimal $70.00\%$.
- $\gamma = 100.0$: Lonceng sangat runcing, setiap support vector membentuk pulau terisolasi (Akurasi Latih $100\%$, Akurasi Uji anjlok ke $43.33\%$ - Overfitting Parah!).

### Langkah 8: Implementasi Dasar k-Nearest Neighbors (`knn_classifier.py`)
Jalankan model KNN berbasis memori:
```bash
python knn_classifier.py
```
*Hasil Uji ($k=5$):* Akurasi Latih $70.74\%$, Akurasi Uji **57.78%** (Macro F1: 0.5746).

### Langkah 9: Eksplorasi Metrik Jarak & Nilai k (`knn_distance_metrics.py` & `knn_k_parameter.py`)
Bandingkan geometri ruang metrik ($L_1, L_2, L_\infty$) dan spektrum nilai $k$:
```bash
python knn_distance_metrics.py
python knn_k_parameter.py
```
*Hasil Verifikasi:*
- Pada $k=1$, akurasi latih selalu $100\%$ (murni menghafal). Nilai $k$ optimal berada pada rentang $k \in [5, 11]$.
- Skema *Distance-Weighted Voting* ($1/d$) meningkatkan akurasi uji menjadi **62.22%** pada $k=5$.

### Langkah 10: Benchmark Komparatif Menyeluruh (`svm_knn_comparison.py`)
Bandingkan seluruh model di bawah protokol validasi silang 5-fold yang adil:
```bash
python svm_knn_comparison.py
```

---

## 4. Tabel Hasil Pengujian Numerik Terverifikasi

| Arsitektur Model | CV Accuracy (Mean ± Std) | CV Macro F1 | Test Accuracy | Fit Time (ms) | Pred Time (ms) | Karakteristik Utama |
|---|---|---|---|---|---|---|
| **Dummy Baseline** | $38.89\% \pm 0.00\%$ | $18.67\%$ | $38.89\%$ | $0.8$ ms | $0.12$ ms | Baseline naif, hanya menebak modus |
| **Linear SVM ($C=1.0$)** | $59.26\% \pm 4.83\%$ | $59.10\%$ | $62.22\%$ | $7.1$ ms | **$1.63$ ms** | Cepat, efisien memori $O(D)$ |
| **RBF SVM ($C=1.0, \gamma=\text{scale}$)** | **$67.78\% \pm 6.58\%$** | **$68.05\%$** | **$70.00\%$** | $7.0$ ms | $2.01$ ms | **Model Terbaik**: Batas non-linier optimal |
| **Polynomial SVM ($d=3$)** | $53.33\% \pm 4.60\%$ | $53.28\%$ | $60.00\%$ | $5.9$ ms | $1.54$ ms | Menangkap interaksi kubik fitur |
| **KNN Uniform ($k=5$)** | $58.89\% \pm 5.42\%$ | $59.05\%$ | $57.78\%$ | **$3.2$ ms** | $2.47$ ms | Lazy learning, voting sama rata |
| **KNN Weighted ($k=5, 1/d$)** | $59.63\% \pm 6.13\%$ | $59.78\%$ | $62.22\%$ | **$3.6$ ms** | $1.89$ ms | Tetangga lebih dekat berbobot tinggi |

---

## 5. Tugas Eksplorasi Mandiri & Tantangan Analitis

1. **Tantangan 1: Kutukan Dimensi (*Curse of Dimensionality*) pada KNN**  
   Tambahkan 20 kolom fitur derau acak Gaussian ($X_{\text{noise}} \sim \mathcal{N}(0, 1)$). Latih kembali KNN dan RBF SVM. Mengapa performa KNN turun jauh lebih drastis dibandingkan SVM? (Petunjuk: Analisis rasio jarak minimum terhadap jarak maksimum di ruang hiper-dimensi tinggi).
2. **Tantangan 2: Sensitivitas Outlier pada Pemilihan Support Vectors**  
   Ubah nilai $C$ pada Linear SVM menjadi sangat besar ($C = 10000$). Tambahkan satu titik anomali di dekat pusat kelas lawan. Amati bagaimana satu titik tersebut dapat memutar arah hiperbidang pemisah secara ekstrem!
3. **Tantangan 3: Latensi Inferensi Skala Besar**  
   Lakukan simulasi pengujian dengan $10.000$ sampel data uji. Bandingkan waktu eksekusi prediksi antara Linear SVM ($w^T x + b$) versus KNN ($10.000 \times 270$ kalkulasi jarak matriks). Jelaskan mengapa KNN tidak cocok untuk sistem IoT tertanam berdaya rendah (*Edge AI*)!

---

## 6. Rubrik Penilaian Praktikum Laboratorium (Total 100 Poin)

| Kriteria Evaluasi | Bobot | Bukti Lengkap (Full Credit) | Bukti Sebagian (Partial Credit) | Kurang Bukti (Insufficient) |
|---|---|---|---|---|
| **Eksekusi & Lingkungan Pipeline** | **20** | Seluruh 10 skrip dan companion notebook tereksekusi tanpa galat di dalam virtual environment. | Sebagian besar skrip berjalan, terdapat galat minor pada penamaan argumen atau path. | Banyak skrip gagal dieksekusi atau tidak menggunakan Pipeline terisolasi. |
| **Pemahaman Geometri Margin & Slack** | **20** | Mampu menjelaskan peran Support Vectors, lebar margin $2/\|w\|$, dan variabel slack $\xi_i$ secara matematis. | Menjelaskan konsep margin namun kurang memahami peran parameter $C$ dan slack. | Tidak memahami konsep support vectors dan margin. |
| **Analisis Kernel Trick & Hyperparameter** | **20** | Membandingkan Linear, Poly, dan RBF Kernel serta menganalisis efek interaksi $C$ dan $\gamma$ secara mendalam. | Menjalankan kernel RBF tanpa analisis pengaruh parameter gamma. | Tidak melakukan eksplorasi fungsi kernel non-linier. |
| **Evaluasi KNN & Analisis Skala Fitur** | **25** | Membuktikan secara eksperimental dampak feature scaling pada KNN/SVM dan menganalisis spektrum nilai $k$. | Melatih KNN namun mengabaikan pembuktian dampak scaling fitur. | Tidak menyertakan implementasi KNN yang valid. |
| **Dokumentasi & Komparasi Model** | **15** | Menyajikan laporan komparasi sistematis (akurasi, F1, waktu fit vs predict, memori) dan kesimpulan teknis berbobot. | Laporan singkat, pembahasan trade-off latensi inferensi kurang tajam. | Tidak ada laporan pembahasan komparatif. |

