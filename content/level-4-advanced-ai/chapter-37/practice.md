# Panduan Praktik Laboratorium: Edge AI & Model Optimization
**Artificial Intelligence Learning Hub — Level 4 (Chapter 37)**  
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## 1. DESKRIPSI DAN PRASYARAT PRAKTIKUM

Praktikum ini membimbing mahasiswa mengimplementasikan, mengukur, dan mengevaluasi secara kuantitatif beragam teknik kompresi model pembelajaran mendalam (**Quantization**, **Weight Pruning**, dan **Knowledge Distillation**) pada CPU lokal tanpa memerlukan perangkat keras embedded fisik maupun akselerator GPU.

### Prasyarat Pengetahuan
- Arsitektur PyTorch dasar (`nn.Module`, `nn.Linear`, `CrossEntropyLoss`) dari Chapter 23.
- Pengukuran metrik evaluasi model dan latensi inferensi dari Chapter 19 & 36.

### Lingkungan Eksekusi
- **Bahasa:** Python 3.10+ (Teruji pada Python 3.12.10).
- **Pustaka Utama:** `torch` (`2.15.0+cpu`, quantized backend `onednn`), `numpy`, `scikit-learn`, `matplotlib`.

---

## 2. PANDUAN LANGKAH DEMI LANGKAH EKSEKUSI

Seluruh 16 script Python mandiri tersimpan di `content/level-4-advanced-ai/chapter-37/code/`:

### Langkah 1: Fondasi Edge AI & Pelatihan Baseline FP32
```bash
python content/level-4-advanced-ai/chapter-37/code/edge_ai_foundations.py
```
- **Tujuan:** Melatih `EdgeTeacherMLP` (`16 -> 256 -> 128 -> 64 -> 4`, `45,764` parameter) pada dataset sensor telemetri 4-kelas hingga akurasi uji `92.33%`.

### Langkah 2: Profiling Model Dasar di CPU
```bash
python content/level-4-advanced-ai/chapter-37/code/model_profiling.py
```
- **Tujuan:** Mengukur jumlah parameter, memori bobot mentah (`178.77 KB`), ukuran berkas `.pt` (`183.94 KB`), latensi dingin/hangat (Mean, P50, P95), dan alokasi heap `tracemalloc`.

### Langkah 3: Matematika Kuantisasi FP32 ke INT8 Manual
```bash
python content/level-4-advanced-ai/chapter-37/code/fp32_int8_quantization_math.py
```
- **Tujuan:** Memverifikasi perhitungan manual skala $s$, *zero point* $z$, nilai `int8`, dekuantisasi, serta kesalahan MSE untuk metode *Symmetric* vs *Asymmetric Affine*.

### Langkah 4: Analisis Galat Kuantisasi & Granularitas Per-Channel
```bash
python content/level-4-advanced-ai/chapter-37/code/quantization_error.py
```
- **Tujuan:** Menguji dampak outlier ekstrem (`15.0`) terhadap penurunan SQNR dan pembuktian bagaimana *Per-Channel Quantization* memulihkan ketepatan saluran berskala kecil.

### Langkah 5: Post-Training Dynamic Quantization
```bash
python content/level-4-advanced-ai/chapter-37/code/dynamic_quantization.py
```
- **Tujuan:** Menerapkan `torch.ao.quantization.quantize_dynamic` (mesin `onednn`), memeriksa penyusutan berkas `.pt` menjadi `52.63 KB` (`3.49x`), dan membandingkan latensi aktual.

### Langkah 6: Post-Training Static Quantization dengan Kalibrasi
```bash
python content/level-4-advanced-ai/chapter-37/code/static_quantization_demo.py
```
- **Tujuan:** Menyisipkan `QuantStub`/`DeQuantStub`, mengumpulkan statistik aktivasi pada 250 sampel kalibrasi melalui `prepare()`, dan mengonversi ke `62.70 KB` (`92.67%` akurasi).

### Langkah 7: Audit Kesetiaan Logit Model Terkuantisasi
```bash
python content/level-4-advanced-ai/chapter-37/code/quantization_evaluation.py
```
- **Tujuan:** Mengukur tingkat kesamaan prediksi Top-1, Mean Absolute Logit Error, dan Cosine Similarity terhadap baseline FP32.

### Langkah 8: Unstructured Magnitude Pruning & Realitas Penyimpanan Padat
```bash
python content/level-4-advanced-ai/chapter-37/code/magnitude_pruning.py
```
- **Tujuan:** Memangkas 70% bobot terendah, memulihkan akurasi lewat *fine-tuning* 5-epoch (`87.17% -> 89.17%`), serta membuktikan ukuran berkas `.pt` padat tidak mengecil (`183.94 KB`).

### Langkah 9: Structured Pruning (Kompresi Fisik Neuron)
```bash
python content/level-4-advanced-ai/chapter-37/code/structured_pruning_demo.py
```
- **Tujuan:** Memangkas 50% neuron tersembunyi berdasarkan norma L1 untuk memangkas jumlah parameter fisik (`45,764 -> 12,644`) dan ukuran `.pt` (`54.57 KB`).

### Langkah 10: Kurva Sensitivitas Sparsity vs Akurasi
```bash
python content/level-4-advanced-ai/chapter-37/code/pruning_sparsity_analysis.py
```
- **Tujuan:** Menyapu rasio kejarangan 0%–95% sebelum dan sesudah fine-tuning lalu menyimpan grafik `experiments/pruning_sparsity_curve.png`.

### Langkah 11: Knowledge Distillation (Teacher–Student Learning)
```bash
python content/level-4-advanced-ai/chapter-37/code/knowledge_distillation.py
```
- **Tujuan:** Memverifikasi bobot Guru membeku (*frozen*), menerapkan *Temperature Scaling* ($T=3.0$) serta *T²-scaled KL Divergence*, dan melatih Student 1.140 parameter (`91.33%`).

### Langkah 12: Komparasi Teacher vs Scratch Student vs Distilled Student
```bash
python content/level-4-advanced-ai/chapter-37/code/teacher_student_comparison.py
```
- **Tujuan:** Menguji perbandingan langsung akurasi, jumlah parameter, ukuran disk, dan latensi antarmodel.

### Langkah 13: Audit Rasio Ukuran Artefak (Mentah vs Gzip)
```bash
python content/level-4-advanced-ai/chapter-37/code/model_size_comparison.py
```
- **Tujuan:** Membandingkan `.pt` mentah versus kompresi `gzip` di seluruh metode optimasi (`8.82 KB` Student = `20.85x` lebih kecil).

### Langkah 14: Benchmark Latensi CPU Berulang
```bash
python content/level-4-advanced-ai/chapter-37/code/inference_latency_benchmark.py
```
- **Tujuan:** Menjalankan 3 trial $\times$ 400 pengulangan setelah 50 *warm-up*, mencatat Mean $\pm$ Std, P50, P95, serta konteks OS/CPU secara objektif.

### Langkah 15: Kartu Skor Komparasi Optimasi Terpadu
```bash
python content/level-4-advanced-ai/chapter-37/code/optimization_comparison.py
```
- **Tujuan:** Menghasilkan tabel komparasi menyeluruh dan grafik 3-panel `experiments/edge_optimization_tradeoffs.png`.

### Langkah 16: Evaluasi Simulasi Anggaran Hardware Embedded
```bash
python content/level-4-advanced-ai/chapter-37/code/edge_hardware_constraints.py
```
- **Tujuan:** Menguji kesesuaian anggaran simulasi SRAM/Flash terhadap profil ARM Cortex-M7, ESP32-S3, dan Raspberry Pi 4 secara konseptual.

---

## 3. LATIHAN BERTINGKAT & PEMBAHASAN

### Tingkat Dasar
1. **Parameter Count & Model Size:** Hitung jumlah bobot + bias pada lapisan `nn.Linear(128, 64)`, lalu hitung berapa KB memori yang dibutuhkan dalam format `FP32` (4 byte) dibanding `INT8` (1 byte)!  
   *Pembahasan:* Jumlah parameter = $(128 \times 64) + 64 = 8.192 + 64 = 8.256$ parameter. Ukuran `FP32` = $8.256 \times 4 = 33.024\text{ byte} \approx 32.25\text{ KB}$. Ukuran `INT8` = $8.256 \times 1 = 8.256\text{ byte} \approx 8.06\text{ KB}$.
2. **Quantization Range:** Sebuah tensor memiliki rentang nilai $[-3.0, +9.0]$. Hitung faktor skala $s$ untuk kuantisasi *Asymmetric INT8* $[-128, 127]$!  
   *Pembahasan:* $s = \frac{9.0 - (-3.0)}{255} = \frac{12.0}{255} \approx 0.047059$.

### Tingkat Menengah
3. **Quantization Error:** Jelaskan mengapa suatu nilai FP32 `x = 1.0` tidak selalu kembali persis menjadi `1.0` setelah mengalami *quantize $\to$ dequantize* ke INT8!  
   *Pembahasan:* Proses pembulatan `round(x / s)` memetakan interval riil kontinu ke hanya 256 bilangan bulat diskret, sehingga menimbulkan galat pembulatan maksimal $\pm s/2$.
4. **Sparsity Ratio:** Jika sebuah model mempunyai $100.000$ parameter dan kita membuang seluruh bobot yang $|w| < 0.01$ yang berjumlah $65.000$ bobot, berapakah rasio kejarangan (*sparsity ratio*) model dan apakah berkas `.pt` otomatis menyusut menjadi $35\%$ ukuran semula?  
   *Pembahasan:* Sparsity = $65\%$. Berkas `.pt` padat **TIDAK** otomatis menyusut karena angka `0.0` tetap disimpan dalam tipe *float* 32-bit kecuali menggunakan format pengkodean matriks jarang (CSR) atau kompresi entropi.

### Tingkat Mahir
5. **Distillation Loss Balance:** Mengapa dalam fungsi rugi distilasi $\mathcal{L}_{KD} = \alpha T^2 \text{KL}(\sigma(z_T/T), \sigma(z_S/T)) + (1-\alpha)\text{CE}(y, z_S)$ suku KL harus dikalikan $T^2$? Apa yang terjadi pada pelatihan Student jika kita memakai $T=5.0$ namun lupa mengalikan $T^2$?  
   *Pembahasan:* Pembagian logit dengan $T$ mengecilkan gradien terhadap $z_S$ dengan faktor $1/T^2 = 1/25$. Tanpa kompensasi perkalian $T^2$, bobot sinyal guru menyusut menjadi hanya $4\%$ kekuatannya sehingga kehilangan kendali atas pembelajaran Student.
6. **Hardware Constraint Trade-off:** Anda ditugaskan memasang pengklasifikasi sensor getaran pada ARM Cortex-M7 (256 KB SRAM, 1 MB Flash, overhead RTOS+TFLite 120 KB SRAM & 200 KB Flash). Evaluasi apakah model `FP32 Teacher (179.5 KB)` aman dipasang dan bandingkan dengan opsi `Distilled Student`!

---

## 4. PANDUAN MINI PROJECT

### Judul Proyek
**Optimizing a Neural Network for CPU-Based Edge Inference**

### Deliverables Wajib
1. Definisi masalah klasifikasi dan dataset (tabular sensor atau citra kompak).
2. Pelatihan model baseline FP32 dengan profiling parameter, ukuran disk `.pt`, dan latensi berulang (Mean/P50/P95).
3. Eksperimen **Quantization** (Dynamic & Static INT8 + analisis kesalahan kuantisasi).
4. Eksperimen **Weight Pruning** (Unstructured vs Structured + pemulihan fine-tuning).
5. Eksperimen **Knowledge Distillation** (Teacher beku melatih Student kompak).
6. Tabel dan visualisasi grafik trade-off Ukuran vs Akurasi vs Latensi, serta analisis kesesuaian anggaran hardware embedded.

---

## 5. RUBRIK PENILAIAN (100 POIN)

| Komponen Penilaian | Bobot | Kriteria Pemenuhan Standar Penuh |
| :--- | :---: | :--- |
| **1. Problem & Baseline Understanding** | **15 Poin** | Masalah terdefinisi jelas, dataset terpisah bebas *leakage*, profil baseline FP32 (parameter, disk size, akurasi, latensi) diukur akurat. |
| **2. Optimization Implementation** | **30 Poin** | Ketiga jalur optimasi (Quantization, Pruning, Knowledge Distillation dengan guru beku + $T^2$ KL) diimplementasikan bersih tanpa bug. |
| **3. Performance Benchmarking** | **25 Poin** | Pengukuran berulang dengan *warm-up* tepat, melaporkan Mean/P50/P95 latensi, rasio kompresi `.pt` nyata, dan konteks CPU. |
| **4. Trade-Off & Hardware Analysis** | **20 Poin** | Interpretasi jujur atas overhead latensi INT8 vs kompresi ruang, perbedaan nyata *Structured* vs *Unstructured*, dan evaluasi anggaran SRAM/Flash. |
| **5. Documentation & Reproducibility** | **10 Poin** | Kode bersih, penetapan random seed terdokumentasi, grafik terbaca rapi berlabel lengkap, serta berjalan 100% di CPU lokal. |

---

## 6. REFERENSI AKADEMIK TERVERIFIKASI

1. **Hinton, G., Vinyals, O., & Dean, J. (2015).** *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning and Representation Learning Workshop. arXiv:1503.02531.
2. **Jacob, B., et al. (2018).** *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018.
3. **Han, S., Mao, H., & Dally, W. J. (2016).** *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR 2016.
4. **Warden, P., & Situnayake, D. (2019).** *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers.* O'Reilly Media.
5. **PyTorch Team. (2024).** *PyTorch Quantization & Pruning Official Documentation.* https://pytorch.org/docs/stable/quantization.html.

