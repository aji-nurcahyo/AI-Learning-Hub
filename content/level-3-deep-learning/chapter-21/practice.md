# Panduan Praktik Laboratorium: Neural Network Fundamentals

Selamat datang di modul panduan praktik **Neural Network Fundamentals** (Bab 21). Modul ini membuka perjalanan Anda di **Level 3: Deep Learning & Generative AI**. Di laboratorium ini, Anda akan membedah dan mengimplementasikan mekanisme internal jaringan saraf tiruan lapis banyak (*Multilayer Perceptron*) dari dasar matematika aljabar linier murni menggunakan **NumPy** tanpa bergantung pada pustaka deep learning tingkat tinggi.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- Python 3.10+ (Direkomendasikan Python 3.12).
- Pustaka: `numpy` (wajib), `matplotlib` (opsional untuk visualisasi).
- Tidak memerlukan GPU, CUDA, atau framework PyTorch/TensorFlow pada bab ini.

### Struktur Repositori Praktik:
```
content/level-3-deep-learning/chapter-21/
├── code/
│   ├── README.md
│   ├── artificial_neuron.py
│   ├── perceptron_logic_gates.py
│   ├── xor_limitation.py
│   ├── activation_functions.py
│   ├── stable_softmax.py
│   ├── mlp_architecture.py
│   ├── forward_propagation.py
│   ├── matrix_shape_demo.py
│   ├── loss_functions.py
│   └── numpy_neural_network_forward.py
└── notebooks/
    └── chapter_21_companion.ipynb
```

---

## 2. Panduan Langkah-demi-Langkah Praktik Laboratorium

### Langkah 1: Model Neuron Buatan Tunggal
Pahami formulasi kombinasi linier terbobot $z = \mathbf{w}^T \mathbf{x} + b$ dan transformasi aktivasi biner:
```bash
python content/level-3-deep-learning/chapter-21/code/artificial_neuron.py
```
- **Fokus Pengamatan**: Perhatikan bagaimana skalar bias $b = -3.0$ mengendalikan ambang keputusan. Tanpa bias, garis keputusan selalu dipaksa melewati titik $(0,0)$.

### Langkah 2: Gerbang Logika & Keterpisahan Linier (*Linear Separability*)
Verifikasi bagaimana Perceptron tunggal dapat memisahkan fungsi logika boolean AND, OR, NAND, dan NOR:
```bash
python content/level-3-deep-learning/chapter-21/code/perceptron_logic_gates.py
```
- **Analisis Hyperplane**: Buktikan bahwa setiap gerbang memiliki konfigurasi parameter bobot dan bias yang memotong ruang input 2D menjadi dua belahan (*half-spaces*).

### Langkah 3: Investigasi Keterbatasan XOR (Minsky & Papert, 1969)
Eksekusi bukti kontradiksi matematis mengapa Perceptron tunggal gagal pada XOR, serta bagaimana jaringan 2-layer mengatasinya:
```bash
python content/level-3-deep-learning/chapter-21/code/xor_limitation.py
```
- **Konsep Representasi**: Perhatikan bagaimana lapisan tersembunyi (*hidden layer*) mentransformasikan masukan non-linier ke ruang baru $[h_1, h_2]$ yang *linearly separable*.

### Langkah 4: Karakteristik Fungsi Aktivasi Non-Linier
Uji perilaku Sigmoid, Tanh, ReLU, dan Leaky ReLU pada domain luas $[-10, 10]$:
```bash
python content/level-3-deep-learning/chapter-21/code/activation_functions.py
```
- **Investigasi Dead ReLU**: Perhatikan bahwa saat masukan bernilai negatif, nilai luaran dan gradien ReLU menjadi tepat nol. Diskusikan solusi Leaky ReLU.

### Langkah 5: Stabilitas Numerik Softmax (Subtract-Max Trick)
Bandingkan formulasi naif $\exp(z_i) / \sum \exp(z_j)$ dengan formulasi stabil numerik:
```bash
python content/level-3-deep-learning/chapter-21/code/stable_softmax.py
```
- **Uji Stres Overflow**: Berikan logits ekstrem seperti $[1000.0, 1001.0, 1002.0]$. Amati bagaimana trik subtraksi nilai maksimum mencegah floating-point `inf` dan `NaN`.

### Langkah 6: Arsitektur MLP & Penghitungan Parameter
Pelajari cara memetakan dimensi matriks bobot dan bias untuk berbagai topologi jaringan:
```bash
python content/level-3-deep-learning/chapter-21/code/mlp_architecture.py
```
- **Aturan Dimensi**: Bobot dari layer berukuran $n_{l-1}$ ke layer berukuran $n_l$ memiliki bentuk matriks $(n_{l-1}, n_l)$, dan bias berbentuk $(1, n_l)$.

### Langkah 7: Worked Example Propagasi Maju (2 -> 3 -> 1)
Lakukan verifikasi perhitungan tangan manual terhadap hasil komputasi matriks NumPy:
```bash
python content/level-3-deep-learning/chapter-21/code/forward_propagation.py
```
- **Kalkulasi Skalar vs Matriks**: Buktikan bahwa nilai $A^{[2]} = 0.5915$ cocok persis dengan penurunan rumus manual langkah-demi-langkah.

### Langkah 8: Validasi Dimensi Batch Tensor
Simulasikan aliran tensor pada ukuran batch $N = 1, 16, 64, 128$:
```bash
python content/level-3-deep-learning/chapter-21/code/matrix_shape_demo.py
```
- **Mekanisme Broadcasting**: Amati bagaimana vektor bias $(1, D_h)$ secara otomatis ditambahkan ke setiap baris dari matriks hasil perkalian $(N, D_h)$.

### Langkah 9: Evaluasi Fungsi Rugi (Loss Functions)
Bandingkan penalti Mean Squared Error (MSE), Binary Cross-Entropy (BCE), dan Categorical Cross-Entropy (CCE):
```bash
python content/level-3-deep-learning/chapter-21/code/loss_functions.py
```
- **Penalti Kesalahan Percaya Diri**: Amati lonjakan nilai loss saat model membuat prediksi salah dengan tingkat keyakinan tinggi ($\hat{y} \to 1$ padahal $y = 0$).

### Langkah 10: Integrasi Forward Pass Lengkap OOP
Jalankan pengujian kelas `NeuralNetworkForward` modular berbasis NumPy:
```bash
python content/level-3-deep-learning/chapter-21/code/numpy_neural_network_forward.py
```

---

## 3. Latihan Laboratorium Mandiri (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. Diberikan neuron buatan dengan masukan $\mathbf{x} = [0.8, -0.5, 1.2]$, bobot $\mathbf{w} = [0.5, 1.0, -0.4]$, dan bias $b = 0.2$. Hitunglah kombinasi linier terbobot $z$ dan luaran aktivasi jika menggunakan fungsi tangga Heaviside ($z \ge 0 \to 1$, $z < 0 \to 0$)!
   - *Jawaban Singkat*: $z = 0.8(0.5) + (-0.5)(1.0) + 1.2(-0.4) + 0.2 = 0.40 - 0.50 - 0.48 + 0.20 = -0.38$. Karena $z < 0$, maka luaran $a = 0$.

### Tingkat 2: Analisis Matematis (Menengah)
1. Mengapa fungsi aktivasi Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ tidak direkomendasikan untuk hidden layers pada jaringan yang sangat dalam (*deep networks*)?
   - *Jawaban Singkat*: Turunan maksimum Sigmoid adalah $\sigma'(0) = 0.25$. Pada jaringan dalam, perkalian berulang faktor $\le 0.25$ saat backpropagation menyebabkan gradien menyusut menuju nol secara eksponensial (*vanishing gradient*), menghentikan proses pembelajaran pada lapisan-lapisan awal.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. Tuliskan fungsi Python NumPy murni `softmax_with_temperature(logits, temperature=1.0)` yang mendukung parameter temperatur $T > 0$. Analisis apa yang terjadi pada distribusi probabilitas ketika $T \to 0$ dan ketika $T \to \infty$!
   - *Jawaban Singkat*: Skala logits menjadi $z_i / T$. Ketika $T \to 0$, distribusi menjadi sangat tajam menyerupai one-hot argmax. Ketika $T \to \infty$, distribusi menjadi seragam (*uniform distribution* $\frac{1}{K}$).

---

## 4. Mini Project Chapter 21

### Judul Proyek:
**"Building a Neural Network Forward Pass from Scratch Using NumPy"**

### Deskripsi Tugas:
Rancang dan implementasikan engine propagasi maju jaringan saraf tiruan lapis banyak (*Multilayer Perceptron*) dari dasar menggunakan NumPy murni tanpa pustaka deep learning eksternal. Model harus mampu menerima konfigurasi arsitektur lapis bebas, inisialisasi bobot adaptif, aktivasi heterogen (ReLU pada hidden layer, Sigmoid/Softmax pada output), dan menghasilkan komputasi loss yang stabil.

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Architecture Understanding** | 20% | Mampu memetakan dimensi layer, bobot matriks, vektor bias, dan parameter count secara tepat. |
| **2. Mathematical Correctness** | 25% | Penurunan matematis forward pass, verifikasi perhitungan manual terhadap kode, dan penanganan stabilitas numerik (Softmax/BCE clipping). |
| **3. Python Implementation** | 25% | Kode NumPy modular, terstruktur secara OOP/fungsi bersih, efisien tanpa loop tak perlu pada pemrosesan batch. |
| **4. Testing and Interpretation** | 20% | Dilengkapi pengujian unit otomatis, pembuktian pada data sintetis biner/multikelas, dan analisis mendalam. |
| **5. Documentation & Reproducibility** | 10% | Laporan rapi, dokumentasi docstring lengkap, dan deterministik (`seed=42`). |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 10 skrip Python dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Perhitungan manual pada arsitektur 2-3-1 terverifikasi identik dengan luaran program.
- [ ] Implementasi Softmax dan Binary Cross-Entropy terbukti tahan terhadap overflow numerik.
- [ ] Notebook pendamping `chapter_21_companion.ipynb` telah dieksekusi secara penuh.
- [ ] Menyertakan laporan Mini Project beserta ringkasan dimensi matriks.

