# Chapter 37: Edge AI & Model Optimization

> **Artificial Intelligence Learning Hub — Level 4: Applied & Advanced AI Systems**  
> *Pengembang:* Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera  
> *Identitas Mata Kuliah:* Bab 37 dari 40 Master Curriculum  
> *Prasyarat:* Chapter 23 (Deep Learning with PyTorch), Chapter 24 (CNNs), Chapter 36 (MLOps & Serving)

---

## DAFTAR ISI
1. [Bagian A: Fondasi Edge AI & Keterbatasan Perangkat Keras](#bagian-a-fondasi-edge-ai--keterbatasan-perangkat-keras)
2. [Bagian B: Profiling Model Dasar pada CPU](#bagian-b-profiling-model-dasar-pada-cpu)
3. [Bagian C: Matematika Kuantisasi (FP32, FP16 & INT8)](#bagian-c-matematika-kuantisasi-fp32-fp16--int8)
4. [Bagian D: Post-Training Quantization (Dynamic & Static Calibration)](#bagian-d-post-training-quantization-dynamic--static-calibration)
5. [Bagian E: Pengantar Quantization-Aware Training (QAT)](#bagian-e-pengantar-quantization-aware-training-qat)
6. [Bagian F: Weight Pruning (Unstructured vs Structured Sparsity)](#bagian-f-weight-pruning-unstructured-vs-structured-sparsity)
7. [Bagian G: Knowledge Distillation (Teacher–Student Learning)](#bagian-g-knowledge-distillation-teacherstudent-learning)
8. [Bagian H: Komparasi Komprehensif Strategi Optimasi](#bagian-h-komparasi-komprehensif-strategi-optimasi)
9. [Bagian I: Keterbatasan Mikrokontroler & Embedded Deployment](#bagian-i-keterbatasan-mikrokontroler--embedded-deployment)
10. [Latihan Bertingkat & Pembahasan Lengkap](#latihan-bertingkat--pembahasan-lengkap)
11. [Panduan Mini Project & Rubrik Asesmen 100 Poin](#panduan-mini-project--rubrik-asesmen-100-poin)
12. [Referensi Akademik Terverifikasi](#referensi-akademik-terverifikasi)

---

## BAGIAN A: FONDASI EDGE AI & KETERBATASAN PERANGKAT KERAS

**Edge AI** adalah paradigma menjalankan inferensi model kecerdasan buatan secara langsung pada perangkat keras lokal (*on-device inference*) di dekat sumber pengambilan data—seperti sensor IoT industri, stasiun monitoring jaringan nirkabel, kamera pengawas, ponsel pintar, hingga mikrokontroler pertanian pintar (*smart agriculture*)—alih-alih mengirimkan seluruh data mentah melalui jaringan ke server komputasi awan (*cloud*).

```
+-----------------------------------------------------------------------+
|             ARSITEKTUR CLOUD INFERENCE VS ON-DEVICE EDGE AI           |
+-----------------------------------------------------------------------+
| 1. CLOUD INFERENCE                                                    |
|    [Sensor/Kamera] --(WAN 50-300ms)--> [Cloud GPU] --(Resp)--> [Aksi] |
|    + Komputasi tak terbatas | - Biaya bandwidth, privasi & offline nol|
|                                                                       |
| 2. ON-DEVICE EDGE AI                                                  |
|    [Sensor/Kamera] --> [NPU / MCU Lokal (< 5 ms)] ----------> [Aksi]  |
|    + Real-time, privat & offline | - Terbatas ketat SRAM, Flash & daya|
+-----------------------------------------------------------------------+
```

### Tiga Kategori Perangkat Keras Edge
1. **Mikrokontroler Ultra-Rendah Daya (TinyML / MCU):** Contoh *ARM Cortex-M7*, *ESP32-S3*. Memori SRAM hanya `256 KB – 512 KB`, Flash penyimpanan `1 MB – 4 MB`, tanpa sistem operasi penuh.
2. **Single-Board Computers (SBC):** Contoh *Raspberry Pi 4/5*, *NVIDIA Jetson Nano*. Menjalankan *Embedded Linux*, RAM `2–8 GB`, mampu menjalankan runtime PyTorch/ONNX lengkap.
3. **Mobile SoC & NPU Khusus:** Prosesor ponsel pintar yang memiliki sirkuit pengali matriks `INT8` berdedikasi.

> **Kejujuran Pengujian:** Simulasi model di CPU komputer kerja tidak membuktikan secara fisik bahwa model tersebut dapat di-flash dan berjalan di ESP32 atau mikrokontroler sebelum diuji langsung pada perangkat keras target (*hardware-in-the-loop*).

---

## BAGIAN B: PROFILING MODEL DASAR PADA CPU

Sebelum melakukan kompresi apa pun, insinyur Edge AI wajib melakukan **Model Profiling** atas baseline FP32 (`EdgeTeacherMLP`: `16 -> 256 -> 128 -> 64 -> 4` pada dataset telemetri sensor 4-kelas, `code/model_profiling.py`):
- **Parameter Count:** `45,764` parameter.
- **Raw FP32 Weight RAM:** `178.77 KB` ($45,764 \times 4\text{ byte} / 1024$).
- **Serialized `.pt` Disk Size:** `183.94 KB`.
- **Test Accuracy:** `92.33%` ($N=600$).
- **CPU Inference Latency:** Mean **`0.0779 ms`**, P95 **`0.1025 ms`** (Windows 11, 32 logical CPUs, PyTorch `2.15.0+cpu`, diukur setelah 50 *warm-up*).

---

## BAGIAN C: MATEMATIKA KUANTISASI (FP32, FP16 & INT8)

### 1. Representasi Bilangan & Reduksi Presisi
- **FP32 (32-bit Floating Point):** 4 byte per bobot (1 bit tanda, 8 bit eksponen, 23 bit mantisa).
- **FP16 (16-bit Half Precision):** 2 byte per bobot ($2\times$ penghematan memori).
- **INT8 (8-bit Signed Integer):** Hanya 1 byte per bobot ($4\times$ penghematan memori!), hanya mampu menampung **256 nilai bilangan bulat diskret** dari $-128$ hingga $+127$.

```
+-----------------------------------------------------------------------+
|          PEMETAAN LINEAR QUANTIZATION (FP32 <---> INT8)               |
+-----------------------------------------------------------------------+
|                                                                       |
|  FP32 Riil:  [ x_min ............................... 0 ....... x_max ]|
|                  |                                   |           |    |
|  Skala (s)       v                                   v           v    |
|  INT8 Grid:  [ -128 ................................ z ........ 127 ] |
|                                                                       |
|  Quantize   :  q = clip( round(x / s) + z,  q_min, q_max )            |
|  Dequantize :  x_hat = s * (q - z)                                    |
+-----------------------------------------------------------------------+
```

### 2. Kuantisasi Simetris vs Asimetris (Worked Example Manual)
Diberikan tensor miring $x = [-0.50,\; 0.00,\; 0.25,\; 1.20,\; 2.54,\; 4.00]$ (24 byte FP32):

1. **Symmetric Quantization** (rentang $[-127, 127]$, *zero point* $z = 0$):
   $$s = \frac{\max(|x|)}{127} = \frac{4.00}{127} \approx 0.031496$$
   - Hasil INT8 (6 byte): $q = [-16,\; 0,\; 8,\; 38,\; 81,\; 127]$
   - Dekuantisasi $\hat{x} = s \cdot q = [-0.5039,\; 0.0000,\; 0.2520,\; 1.1969,\; 2.5512,\; 4.0000]$
   - MSE Error = **$6.923 \times 10^{-5}$**.

2. **Asymmetric Affine Quantization** (rentang penuh $[-128, 127]$, $z \neq 0$):
   $$s = \frac{x_{\max} - x_{\min}}{255} = \frac{4.00 - (-0.50)}{255} \approx 0.017647, \quad z = \text{round}\left(-\frac{-0.50}{s}\right) - 128 = -100$$
   - Hasil INT8: $q = [-128,\; -100,\; -86,\; -32,\; 44,\; 127]$
   - Dekuantisasi $\hat{x} = s \cdot (q - z)$ menghasilkan **MSE Error = $1.522 \times 10^{-5}$** ($4.5\times$ lebih akurat pada aktivasi miring!).

> **Penting:** Karena hanya tersedia 256 tingkat diskret pada INT8, tidak semua nilai FP32 dapat direpresentasikan secara tepat; pembulatan pasti menyisakan galat kuantisasi (*quantization error*). Selain itu, keberadaan **outlier ekstrem** memperbesar nilai skala $s$ secara menyeluruh (menjatuhkan SQNR bobot normal), sehingga arsitektur modern menggunakan **Per-Channel Quantization** yang menghitung $(s_c, z_c)$ secara mandiri per baris saluran.

---

## BAGIAN D: POST-TRAINING QUANTIZATION (DYNAMIC & STATIC CALIBRATION)

**Post-Training Quantization (PTQ)** mengonversi model FP32 yang sudah selesai dilatih ke presisi INT8 tanpa pelatihan ulang dari nol. Terdapat dua mode utama yang tersedia di PyTorch CPU (memakai backend aktif `onednn`):

| Mode PTQ | Cara Kerja | Membutuhkan Data Kalibrasi? | Karakteristik Praktis |
| :--- | :--- | :---: | :--- |
| **Dynamic Quantization** | Bobot disimpan `INT8`, tetapi rentang skala aktivasi dihitung **on-the-fly saat runtime**. | **Tidak** | Paling mudah (`quantize_dynamic`), mengecilkan `.pt` ke **`52.63 KB` ($3.49\times$)**, akurasi `92.17%`. |
| **Static Quantization** | Memasang `QuantStub`/`DeQuantStub`, memantau rentang aktivasi sebelumnya via **`prepare()` + data kalibrasi**, lalu mengunci skala di **`convert()`**. | **Ya** (misal 250 sampel acuan) | Menghilangkan observer min/max saat inferensi; `.pt` = **`62.70 KB` ($2.93\times$)**, akurasi **`92.67%`**. |

*Catatan Kompatibilitas API:* Di PyTorch `2.15.0+cpu`, API `torch.ao.quantization` didukung penuh oleh backend `onednn` dengan catatan transisi masa depan menuju `torchao`.

---

## BAGIAN E: PENGANTAR QUANTIZATION-AWARE TRAINING (QAT)

Apabila PTQ menyebabkan degradasi akurasi yang tidak dapat diterima (umumnya pada arsitektur super sempit seperti MobileNet atau presisi sub-8-bit), digunakan **Quantization-Aware Training (QAT)**. QAT menyisipkan node **Fake Quantization** (simulasi pembulatan + *clamp* INT8) di dalam *forward pass* selama pelatihan sementara gradien *backpropagation* tetap diperbarui menggunakan **Straight-Through Estimator (STE)** dalam FP32, memungkinkan bobot belajar mengkompensasi galat pembulatan secara langsung.

---

## BAGIAN F: WEIGHT PRUNING (UNSTRUCTURED VS STRUCTURED SPARSITY)

**Weight Pruning** membuang bobot jaringan yang memiliki magnitudo kecil ($|w| \approx 0$) dan kontribusi minim terhadap prediksi.

```
+-----------------------------------------------------------------------+
|         UNSTRUCTURED PRUNING VS STRUCTURED NEURON PRUNING             |
+-----------------------------------------------------------------------+
| 1. UNSTRUCTURED (Sparsity 70%)       2. STRUCTURED (Hapus 50% Neuron) |
|    [ 0.4   0    0  -0.9 ]               [ 0.4  -0.9 ]                 |
|    [  0    0   0.2   0  ]               [ 0.8   0.2 ] (Dimensi fisik  |
|    [ 0.8   0    0    0  ]               Matriks baru   benar-benar    |
|    Matriks tetap 3x4 penuh angka 0!     menyusut ke 2x2!)             |
+-----------------------------------------------------------------------+
```

### Realitas Penting Penyimpanan & Kecepatan CPU
Banyak pengembang pemula keliru mengira *Unstructured Pruning* otomatis memperkecil ukuran berkas model atau mempercepat CPU. Pengujian aktual pada `code/magnitude_pruning.py` & `code/model_size_comparison.py` membuktikan kenyataan berikut:
- **Unstructured L1 Pruning (70% nol + 5-Epoch Fine-Tuning):** Akurasi bertahan di **`89.17%`**, namun berkas padat `.pt` tetap berukuran **`183.94 KB` ($1.00\times$ rasio kompresi mentah!)** dan latensi CPU hampir tidak berubah (`0.0724 ms`). Angka `0.0` tetap memakan 32 bit di memori kecuali dikonversi ke format jarang (**CSR/CSC**) atau dikompresi *entropy coding* (**`gzip` turun ke `70.27 KB`**).
- **Structured Neuron Pruning (50% neuron tersembunyi dihapus):** Memangkas lapisan fisik dari `16->256->128->64->4` menjadi `16->128->64->32->4` (`45,764 -> 12,644` parameter), langsung menyusutkan berkas `.pt` fisik menjadi **`54.57 KB` ($3.37\times$ lebih kecil)** dengan akurasi **`89.67%`**.

---

## BAGIAN G: KNOWLEDGE DISTILLATION (TEACHER–STUDENT LEARNING)

**Knowledge Distillation** (Hinton et al., 2015) melatih arsitektur baru berukuran sangat ramping (**Student Model**, misal `16 -> 32 -> 16 -> 4` hanya **1.140 parameter**) untuk meniru keluaran probabilitas lembut (*soft targets*) dari **Teacher Model** besar yang dibekukan (`requires_grad = False`).

### Formulasi Temperature Scaling & $T^2$ KL Divergence
Logit guru $z_T$ dan murid $z_S$ dihaluskan dengan parameter suhu $T > 1$:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Fungsi rugi gabungan (*Distillation Loss*) memadukan penyelarasan distribusi lembut (KL Divergence) dan label keras asli (Cross-Entropy):

$$\mathcal{L}_{\text{total}} = \alpha \cdot T^2 \cdot D_{\text{KL}}\left(\sigma\left(\frac{z_T}{T}\right) \;\middle\|\; \sigma\left(\frac{z_S}{T}\right)\right) + (1 - \alpha) \cdot \text{CE}(y, z_S)$$

Faktor pengali **$T^2$** secara matematis wajib disertakan karena gradien terhadap logit yang dibagi $T$ mengecil sebanding dengan $1/T^2$; perkalian $T^2$ menyeimbangkan kembali skala gradien lunak agar setara dengan suku *Cross-Entropy*.

Pada pengujian `code/knowledge_distillation.py` ($T=3.0, \alpha=0.6$), **Distilled Student (`1,140` parameter)** mencapai akurasi uji **`91.33%`** hanya dengan ukuran disk **`8.82 KB`** (**$20.85\times$ lebih kecil** dari Teacher). *(Catatan: Distilasi tidak otomatis selalu mengalahkan kapasitas maksimal guru; kapasitas murid tetap menjadi batas atas akurasi).*

---

## BAGIAN H: KOMPARASI KOMPREHENSIF STRATEGI OPTIMASI

Tabel berikut merangkum eksekusi aktual berulang pada **Windows 11 CPU (32 logical CPUs, PyTorch `2.15.0+cpu`, setelah 50 *warm-up*)** dari `code/optimization_comparison.py` dan `code/model_size_comparison.py`:

| Strategi Optimasi | Parameter | Test Acc (%) | `.pt` Mentah (KB) | Rasio Ukuran | `gzip` (KB) | Mean Lat (ms) | P95 Lat (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. FP32 Baseline** | 45,764 | **92.33%** | 183.94 | 1.00x | 167.99 | 0.0779 | 0.1025 |
| **2. Dynamic INT8** | 45,764 | 92.17% | **52.63** | **3.49x** | 43.99 | 1.0943 | 1.8133 |
| **3. Static INT8 (Calib)** | 45,764 | **92.67%** | 62.70 | **2.93x** | 50.04 | 2.0225 | 3.3462 |
| **4. Unstructured Pruned (70%)** | 45,764 | 89.17% | 183.94 | **1.00x** | 70.27 | 0.0724 | 0.1077 |
| **5. Structured Pruned (50%)** | 12,644 | 89.67% | 54.57 | **3.37x** | 48.25 | 0.0714 | 0.0876 |
| **6. Distilled Student** | **1,140** | 91.33% | **8.82** | **20.85x** | **6.02** | **0.0503** | **0.0620** |

> **Observasi Jujur Benchmark CPU:** Pada tensor MLP kecil (*batch=1*, lebar maks 256) di x86 Windows CPU, kernel kuantisasi `onednn` memiliki *dispatch overhead* lebih tinggi dari operasi FP32 AVX2 bawaan sehingga latensi INT8 (`1.09–2.02 ms`) terukur lebih lambat daripada FP32 (`0.0779 ms`), meskipun ukuran file menyusut $3.5\times$. Percepatan latensi CPU sejati pada model kecil dicapai melalui **Distilled Student (`0.0503 ms`)** dan **Structured Pruning (`0.0714 ms`)**.

---

## BAGIAN I: KETERBATASAN MIKROKONTROLER & EMBEDDED DEPLOYMENT

Menempatkan model ke mikrokontroler nyata mengharuskan konversi format (misalnya TFLite Micro C array / ONNX) serta memperhitungkan anggaran **Tensor Arena SRAM** dan **Flash Firmware**. Berdasarkan simulasi anggaran di `code/edge_hardware_constraints.py`:
- **ARM Cortex-M7 (256 KB SRAM, 1 MB Flash):** `Distilled Student (4.5 KB bobot)` dan `INT8 Teacher (45.9 KB)` masuk aman di anggaran simulasi bersama runtime TFLite Micro (`~120 KB`).
- **Heavy Vision CNN (ResNet18, 44 MB):** Mustahil berjalan di MCU (*RAM & Flash overflow*), memerlukan perangkat SBC kelas Raspberry Pi 4 atau NPU tepi.

---

## LATIHAN BERTINGKAT & PEMBAHASAN LENGKAP

### Tingkat Dasar
1. **Perhitungan Ukuran Bobot:** Berapa byte memori yang dibutuhkan menyimpan matriks bobot `nn.Linear(512, 256, bias=False)` dalam `FP32` vs `INT8`?  
   *Pembahasan:* Total elemen = $512 \times 256 = 131.072$. Dalam `FP32`: $131.072 \times 4 = 524.288\text{ byte} = 512\text{ KB}$. Dalam `INT8`: $131.072 \times 1 = 131.072\text{ byte} = 128\text{ KB}$.
2. **Rentang Kuantisasi:** Mengapa rentang asimetris `[-128, 127]` disarankan pada output aktivasi `ReLU` dibandingkan simetris `[-127, 127]`?  
   *Pembahasan:* Output `ReLU` selalu $\ge 0$. Kuantisasi simetris dengan $z=0$ membuang seluruh 127 tingkat bilangan bulat negatif secara percuma, memangkas resolusi presisi hingga setengahnya.

### Tingkat Menengah
3. **Diagnosis Pruned Storage:** Mengapa file `.pt` model yang 70% bobotnya telah di-nol-kan lewat `prune.l1_unstructured` + `prune.remove` masih berukuran sama persis (`183.94 KB`)?  
   *Pembahasan:* Karena tensor PyTorch standar berjenis *dense float32*, sehingga angka `0.0000` tetap memakan 4 byte penyimpanan kecuali dikonversi menjadi struktur jarang *CSR/CSC* atau dikompresi menggunakan `gzip`.

### Tingkat Mahir
4. **Pemilihan Strategi Edge:** Anda merancang sensor getaran baterai bertenaga koin dengan anggaran Flash tersisa hanya `20 KB` dan latensi wajib `< 0.5 ms`. Manakah kombinasi teknik yang Anda rekomendasikan berdasarkan bukti tabel Bagian H?  
   *Pembahasan:* **Knowledge Distillation** (`8.82 KB` raw `.pt`, `1,140` parameter, `0.0503 ms` latensi, akurasi `91.33%`) adalah satu-satunya opsi yang memenuhi anggaran memori ketat sambil mempertahankan akurasi kompetitif dan latensi tercepat.

---

## PANDUAN MINI PROJECT & RUBRIK ASESMEN 100 POIN

### Judul Proyek
**Optimizing a Neural Network for CPU-Based Edge Inference**

### Rubrik Penilaian (100 Poin)
| Kriteria Asesmen | Bobot | Indikator Penilaian Maksimal |
| :--- | :---: | :--- |
| **1. Problem & Baseline** | **15** | Dataset bebas kebocoran data, profil lengkap parameter, ukuran disk `.pt`, akurasi, dan latensi baseline terukur. |
| **2. Optimization Implementation** | **30** | Kuantisasi (Dynamic + Static kalibrasi), Pruning (Unstructured vs Structured), dan Knowledge Distillation guru beku diimplementasikan benar. |
| **3. Performance Benchmarking** | **25** | Pengukuran latensi berulang setelah warm-up (Mean, P50, P95), rasio ukuran disk mentah vs terkompresi, dan dokumentasi CPU. |
| **4. Trade-Off & Hardware Analysis** | **20** | Interpretasi jujur atas realitas latensi INT8 vs kompresi ruang, analisis perbedaan *structured/unstructured*, dan pemetaan anggaran SRAM/Flash. |
| **5. Documentation & Reproducibility** | **10** | Kode bersih, penetapan random seed konsisten, visualisasi trade-off rapi, serta reproduktif 100% di CPU. |

---

## REFERENSI AKADEMIK TERVERIFIKASI

1. **Hinton, G., Vinyals, O., & Dean, J. (2015).** *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning and Representation Learning Workshop. https://arxiv.org/abs/1503.02531
2. **Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., ... & Kalenichenko, D. (2018).** *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2018). https://doi.org/10.1109/CVPR.2018.00286
3. **Han, S., Mao, H., & Dally, W. J. (2016).** *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* International Conference on Learning Representations (ICLR 2016). https://arxiv.org/abs/1510.00149
4. **Warden, P., & Situnayake, D. (2019).** *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers.* O'Reilly Media.
5. **PyTorch Team. (2024).** *PyTorch Quantization & Pruning Official Documentation.* https://pytorch.org/docs/stable/quantization.html

