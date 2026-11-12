# Panduan Praktik Laboratorium: Convolutional Neural Networks (CNN)

Selamat datang di modul panduan praktik laboratorium **Convolutional Neural Networks** (Bab 24). Pada laboratorium ini, Anda akan mempraktikkan perancangan, implementasi, dan pengujian arsitektur CNN modern menggunakan **PyTorch** dan pustaka visi komputer **torchvision** pada jalur komputasi CPU: mulai dari mekanisme dasar korelasi silang 2D (*cross-correlation*), perhitungan dimensi spasial output dan parameter bobot, komparasi efisiensi parameter CNN terhadap Multilayer Perceptron (MLP), pemrosesan pipeline citra standar Fashion-MNIST, komparasi performa empiris, hingga analisis kesalahan (*error analysis*) dan visualisasi representasi *feature maps*.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- Python: Versi 3.10+ (Direkomendasikan Python 3.12).
- Pustaka Inti: `torch>=2.0.0` (Jalur CPU terverifikasi pada PyTorch 2.15.0+cpu), `torchvision>=0.15.0` (terverifikasi 0.29.2+cpu), `numpy>=2.0.0`, `matplotlib>=3.8.0`.
- Akselerator Perangkat Keras: Bersifat opsional. Seluruh skrip, pipeline pelatihan 5 epoch, dan evaluasi bab ini dirancang agar dapat dieksekusi secara efisien pada CPU laptop/komputer standar dalam waktu kurang dari 5 menit tanpa memerlukan GPU.

### Struktur Repositori Praktik Bab 24:
```
content/level-3-deep-learning/chapter-24/
├── code/
│   ├── README.md
│   ├── manual_convolution.py
│   ├── conv_shape_calculator.py
│   ├── cnn_architecture.py
│   ├── data_loader_fashion_mnist.py
│   ├── train_cnn_vs_baseline.py
│   └── error_analysis_feature_maps.py
├── experiments/
│   └── results_ch24.json
├── models/
│   └── fashion_cnn_weights.pt
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_24_companion.ipynb
├── slides/
│   ├── build_slides.py
│   └── deck.json
├── index.md
├── practice.md
├── metadata.json
└── audit.md
```

---

## 2. Panduan Langkah-demi-Langkah Praktik Laboratorium

### Langkah 1: Operasi Konvolusi Manual (Cross-Correlation 2D) vs PyTorch
Jalankan skrip pembuktian konvolusi dari scratch untuk memahami mekanisme perkalian dot geser dan korelasi silang:
```bash
python content/level-3-deep-learning/chapter-24/code/manual_convolution.py
```
- **Fokus Pengamatan**:
  - Pahami bahwa operasi "konvolusi" dalam pustaka deep learning (seperti PyTorch dan TensorFlow) sesungguhnya adalah **korelasi silang 2D** (*cross-correlation*) tanpa membalikkan kernel 180 derajat.
  - Perhatikan bagaimana kernel detektor tepi horizontal $3 \times 3$ menggeser patch jendela pada matriks masukan $5 \times 5$ untuk menghasilkan keluaran $3 \times 3$.
  - Amati validasi numerik terhadap `torch.nn.functional.conv2d`: deviasi absolut maksimum harus berada di bawah presisi floating-point ($10^{-6}$), membuktikan kebenaran matematis implementasi manual.

### Langkah 2: Kalkulator Dimensi Spasial & Parameter Bobot
Hitung dan verifikasi bentuk tensor serta jumlah parameter lapisan konvolusional:
```bash
python content/level-3-deep-learning/chapter-24/code/conv_shape_calculator.py
```
- **Fokus Pengamatan**:
  - Uji rumus spasial:
    $$H_{out} = \left\lfloor \frac{H_{in} + 2P - D(K - 1) - 1}{S} + 1 \right\rfloor$$
    pada konfigurasi *Valid Padding* ($P=0$), *Same Padding* ($P=1$), *Strided Convolution* ($S=2$), dan *Dilated Convolution* ($D=2$).
  - Uji rumus parameter bobot:
    $$\text{Params} = (K_h \times K_w \times C_{in} + 1) \times C_{out}$$
  - Buktikan bahwa lapisan pooling (`nn.MaxPool2d`) memiliki **0 parameter**, karena hanya melakukan agregasi nilai maksimum tanpa bobot terpelajari.

### Langkah 3: Komparasi Arsitektur CNN vs Flatten MLP Baseline
Bandingkan desain struktural antara jaringan konvolusional hierarkis dan jaringan multilayer perceptron:
```bash
python content/level-3-deep-learning/chapter-24/code/cnn_architecture.py
```
- **Fokus Pengamatan**:
  - Amati perbedaan parameter lapisan pertama: `Conv2d(1, 16, kernel_size=3)` hanya membutuhkan **160 parameter**, sedangkan `Linear(784, 128)` pada MLP membutuhkan **100.480 parameter** (CNN 99.84% lebih hemat bobot!).
  - Pahami prinsip *Local Receptive Field* dan *Weight Sharing* yang memungkinkan CNN mendeteksi pola yang sama di sembarang koordinat citra tanpa ledakan parameter.
  - Periksa dimensi perataan tensor (*flatten*) sebelum masuk ke lapisan fully connected: $(B, 32 \times 7 \times 7) = (B, 1568)$.

### Langkah 4: Pipeline Aliran Data Fashion-MNIST & Normalisasi
Siapkan pipeline pemuatan data citra standar menggunakan torchvision:
```bash
python content/level-3-deep-learning/chapter-24/code/data_loader_fashion_mnist.py
```
- **Fokus Pengamatan**:
  - Amati transformasi input: konversi `transforms.ToTensor()` yang menskalakan nilai intensitas piksel $[0, 255]$ ke interval $[0.0, 1.0]$.
  - Terapkan normalisasi empiris resmi dataset Fashion-MNIST: `mean = 0.2860` dan `std = 0.3530` untuk menstabilkan distribusi masukan di sekitar nol.
  - Perhatikan partisi data: 60.000 citra latih dibagi menjadi Train (50.000 sampel) dan Validation (10.000 sampel) dengan seed tetap, serta 10.000 citra Test yang diisolasi ketat.

### Langkah 5: Eksperimen Benchmark Komparatif (CNN vs MLP)
Latih model `FashionCNN` dan `FlattenMLPBaseline` di bawah protokol komparasi yang sepenuhnya setara:
```bash
python content/level-3-deep-learning/chapter-24/code/train_cnn_vs_baseline.py
```
- **Fokus Pengamatan**:
  - Kedua model dilatih selama 5 epoch menggunakan optimizer Adam ($lr=0.001$), fungsi rugi `nn.CrossEntropyLoss()`, dan ukuran batch 64 pada CPU.
  - Perhatikan bahwa `FashionCNN` mencapai akurasi uji **86.30%** (loss 0.3664) mengungguli `FlattenMLPBaseline` sebesar **84.15%** (loss 0.4410).
  - Telusuri performa spesifik per kelas: pada kelas kemeja (*Shirt*), CNN mencatat akurasi **75.63%** dibandingkan MLP yang hanya mencapai **54.31%**—menghasilkan keunggulan masif sebesar **+21.32%**!
  - Verifikasi berkas hasil eksperimen terstruktur yang tersimpan di `experiments/results_ch24.json` dan bobot model terbaik di `models/fashion_cnn_weights.pt`.

### Langkah 6: Error Analysis & Inspeksi Feature Maps
Analisis pola kekeliruan prediksi model dan visualisasikan peta aktivasi fitur yang dipelajari:
```bash
python content/level-3-deep-learning/chapter-24/code/error_analysis_feature_maps.py
```
- **Fokus Pengamatan**:
  - Identifikasi pasangan kelas yang paling sering tertukar (*top confused pairs*): pasangan T-shirt vs Shirt (226 kekeliruan), Pullover vs Coat (129 kekeliruan), dan Shirt vs Coat (118 kekeliruan) yang merefleksikan kedekatan siluet visual pada resolusi $28 \times 28$.
  - Ekstraksi aktivasi dari 16 kanal lapisan konvolusi pertama (`features[0:3]`) dan amati filter-filter yang bertindak sebagai detektor tepi vertikal, horizontal, dan kontras tekstur lokal.

---

## 3. Latihan Laboratorium Mandiri (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. **Perhitungan Output Konvolusi**: Suatu citra berwarna berdimensi $(C_{in}, H_{in}, W_{in}) = (3, 64, 64)$ diproses melalui lapisan konvolusi dengan spesifikasi: jumlah filter $C_{out} = 32$, ukuran kernel $K = 5 \times 5$, padding $P = 2$, dan stride $S = 2$.
   - Berapakah dimensi spasial $(H_{out}, W_{out})$ dari tensor keluaran?
   - Berapakah total parameter terbobot (termasuk bias) pada lapisan tersebut?
   - *Jawaban*:
     - Dimensi spasial: $H_{out} = \lfloor \frac{64 + 2(2) - 5}{2} + 1 \rfloor = \lfloor \frac{63}{2} + 1 \rfloor = 31 + 1 = 32$. Dimensi keluaran adalah $(32, 32, 32)$.
     - Total parameter: $(K_h \times K_w \times C_{in} + 1) \times C_{out} = (5 \times 5 \times 3 + 1) \times 32 = (75 + 1) \times 32 = 76 \times 32 = 2.432$ parameter.

2. **Weight Sharing vs Fully Connected**: Jika citra di atas diproses oleh lapisan Fully Connected (`nn.Linear`) yang menghasilkan representasi 32 dimensi ($32 \times 32 \times 32 = 32.768$ nilai keluaran), berapa banyak bobot yang dibutuhkan lapisan linier tersebut? Bandingkan dengan efisiensi parameter CNN!
   - *Jawaban*:
     - Lapisan linier membutuhkan: $(3 \times 64 \times 64) \times (32 \times 32 \times 32) = 12.288 \times 32.768 \approx 402.653.184$ bobot (lebih dari 400 juta parameter!).
     - Lapisan konvolusi hanya membutuhkan $2.432$ parameter. CNN menghemat lebih dari $99.999\%$ parameter berkat prinsip *Weight Sharing* (satu kernel digunakan berulang di seluruh area spasial) dan *Local Receptive Field*.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Galat Shape Dimensi Linear**: Seorang praktikan menyusun arsitektur CNN berikut untuk input citra MNIST $(1, 28, 28)$:
   ```python
   self.features = nn.Sequential(
       nn.Conv2d(1, 16, kernel_size=3, padding=0),  # Output: 26x26
       nn.MaxPool2d(2, 2),                         # Output: 13x13
       nn.Conv2d(16, 32, kernel_size=3, padding=0), # Output: 11x11
       nn.MaxPool2d(2, 2)                          # Output: 5x5
   )
   self.fc = nn.Linear(32 * 7 * 7, 10) # Runtime error di baris ini!
   ```
   Jelaskan mengapa kode di atas memicu galat runtime `mat1 and mat2 shapes cannot be multiplied` dan berikan koreksi perbaikannya!
   - *Jawaban*:
     - Pada `Conv2d` dengan $P=0$, dimensi spasial berkurang pada setiap layer:
       1. Setelah Conv1 ($K=3, P=0$): $\lfloor \frac{28 - 3}{1} + 1 \rfloor = 26$.
       2. Setelah Pool1 ($K=2, S=2$): $\lfloor \frac{26}{2} \rfloor = 13$.
       3. Setelah Conv2 ($K=3, P=0$): $\lfloor \frac{13 - 3}{1} + 1 \rfloor = 11$.
       4. Setelah Pool2 ($K=2, S=2$): $\lfloor \frac{11}{2} \rfloor = 5$.
     - Tensor setelah perataan berdimensi $(B, 32 \times 5 \times 5) = (B, 800)$, sedangkan `Linear` mengasumsikan input sebesar $32 \times 7 \times 7 = 1568$.
     - Perbaikan yang benar: Ubah inisialisasi lapisan linier menjadi `nn.Linear(32 * 5 * 5, 10)` atau gunakan padding $P=1$ pada kedua lapisan konvolusi.

2. **Stride vs Pooling**: Dalam arsitektur modern seperti ResNet dan ConvNeXt, mengapa para perancang jaringan sering menggantikan lapisan `nn.MaxPool2d` dengan *Strided Convolution* (`nn.Conv2d` dengan `stride=2`) untuk mereduksi resolusi spasial?
   - *Jawaban*:
     - Max Pooling menggunakan fungsi agregasi non-linier deterministik tanpa parameter terpelajari, yang dapat membuang informasi gradien halus.
     - Strided Convolution memungkinkan jaringan mempelajari sendiri fungsi downsampling optimal yang meminimalkan rugi informasi melalui pembelajaran backpropagation adaptif.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Effective Receptive Field (ERF)**: Receptive field menunjukkan luas area pada citra masukan yang mempengaruhi satu nilai aktivasi pada lapisan tersembunyi. Rumus rekursif receptive field adalah:
   $$RF_{l} = RF_{l-1} + (K_l - 1) \times J_{l-1}$$
   di mana $J_{l-1} = \prod_{i=1}^{l-1} S_i$ adalah akumulasi langkah pergeseran (*jump* atau *stride* kumulatif).
   Hitung Receptive Field pada keluaran lapisan ke-3 untuk arsitektur berikut:
   - Layer 1: Conv2d ($K_1=3, S_1=1$)
   - Layer 2: MaxPool2d ($K_2=2, S_2=2$)
   - Layer 3: Conv2d ($K_3=3, S_3=1$)
   - *Jawaban*:
     - Input: $RF_0 = 1, J_0 = 1$.
     - Layer 1: $RF_1 = 1 + (3 - 1) \times 1 = 3$. Jump: $J_1 = 1 \times 1 = 1$.
     - Layer 2: $RF_2 = 3 + (2 - 1) \times 1 = 4$. Jump: $J_2 = 1 \times 2 = 2$.
     - Layer 3: $RF_3 = 4 + (3 - 1) \times 2 = 4 + 4 = 8$.
     - Jadi, satu neuron pada keluaran Layer 3 memiliki Receptive Field sebesar $8 \times 8$ piksel pada citra input asli.

---

## 4. Mini Project Chapter 24

### Judul Tugas Proyek:
**"Benchmarking Convolutional Neural Networks on Visual Patterns"**

### Deskripsi Tugas:
Rancang, bangun, dan evaluasi studi komparatif terstruktur antara arsitektur **Convolutional Neural Networks (CNN)** dan arsitektur **Multilayer Perceptron (MLP)** pada dataset pengenalan pola citra **Fashion-MNIST**. Anda diminta untuk:
1. Membangun pipeline data terstandar menggunakan torchvision dengan normalisasi empiris resmi ($mean=0.2860, std=0.3530$), pembagian partisi 3-way (Train, Validation, Test) bebas tumpang tindih, dan isolasi augmentasi data hanya pada data latih.
2. Membangun model `FashionCNN` modular yang menerapkan prinsip *local connectivity*, *weight sharing*, *batch normalization*, dan *spatial pooling*.
3. Membangun model acuan `FlattenMLPBaseline` dengan parameter bobot yang seimbang terhadap CNN (~105k–109k parameter) guna menjamin perbandingan kapasitas komputasional yang adil.
4. Melatih kedua model pada CPU selama 5 epoch di bawah konfigurasi hyperparameter yang sepenuhnya identik (Adam, lr=0.001, batch 64).
5. Menganalisis hasil eksperimen: komparasi metrik uji keseluruhan, konvergensi loss, dan pembuktian empiris keunggulan CNN pada kategori pakaian bertekstur kompleks (*Shirt*, *Pullover*).
6. Mengekstraksi aktivasi *feature maps* lapisan konvolusi pertama dan memetakan matriks kontingensi kekeliruan prediksi (*error analysis*).

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Data, Split & Transforms** | 20 Poin | Pipeline data torchvision bersih, transformasi `ToTensor()` dan normalisasi empiris ($mean=0.2860, std=0.3530$) diterapkan tepat, partisi 3-way (Train 50k, Val 10k, Test 10k) terisolasi tanpa tumpang tindih data leakage, dan mekanisme fallback sintetis tersedia. |
| **2. Architecture & Shape Correctness** | 20 Poin | Desain `FashionCNN` modular berbasis `nn.Module`, kalkulasi dimensi spasial tensor ($H_{out}, W_{out}$) akurat di setiap layer, perataan (*flatten*) ke lapisan linear presisi, dan luaran berupa logits mentah 10-kelas. |
| **3. Experiments & Baseline Comparison** | 25 Poin | Pelatihan komparatif CNN vs FlattenMLP tereksekusi mulus pada CPU, kondisi eksperimen dikontrol ketat (jumlah epoch, optimizer, learning rate, loss function identik), dan hasil benchmark tercatat rapi dalam format JSON. |
| **4. Evaluation & Error Analysis** | 20 Poin | Evaluasi performa uji menyeluruh, pelaporan akurasi per kelas yang membuktikan keunggulan CNN pada pola spasial kompleks (*Shirt*), identifikasi pasangan kelas tertukar (*top confused pairs*), dan visualisasi/ekstraksi aktivasi *feature maps* lapisan pertama. |
| **5. Reproducibility & Communication** | 15 Poin | Repositori kode bersih tanpa duplikasi, dokumentasi keterbatasan resolusi spasial $28 \times 28$, ketiadaan ketergantungan GPU khusus, dan penyajian kesimpulan ilmiah yang lugas dan berbasis data empiris. |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 6 skrip Python di direktori `code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Operasi konvolusi manual terbukti menghasilkan nilai identik terhadap `F.conv2d`.
- [ ] Kalkulator dimensi spasial dan jumlah parameter terverifikasi cocok dengan output PyTorch.
- [ ] Dataset Fashion-MNIST termuat dengan normalisasi `mean=0.2860` dan `std=0.3530`.
- [ ] Model `FashionCNN` dan `FlattenMLPBaseline` dilatih dan dievaluasi di bawah protokol eksperimen setara.
- [ ] Keunggulan CNN pada kelas pakaian bertekstur kompleks (*Shirt*) terbukti secara kuantitatif.
- [ ] Analisis kesalahan memetakan 3 pasangan kelas yang paling rentan tertukar.
- [ ] Fitur peta aktivasi konvolusi pertama terekstraksi dan terdokumentasi.
- [ ] Berkas hasil eksperimen `experiments/results_ch24.json` dan bobot model `models/fashion_cnn_weights.pt` tersimpan.
- [ ] Notebook pendamping `chapter_24_companion.ipynb` tervalidasi bersih dalam format v4.

