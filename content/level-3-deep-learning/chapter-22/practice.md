# Panduan Praktik Laboratorium: Backpropagation & Optimization

Selamat datang di modul panduan praktik **Backpropagation & Optimization** (Bab 22). Di laboratorium ini, Anda akan membedah algoritma inti yang menggerakkan seluruh revolusi deep learning modern: kalkulus propagasi balik (*backpropagation* via aturan rantai), verifikasi gradien numerik (*gradient checking*), dinamika pembaruan parameter (*stochastic*, *mini-batch*, *momentum*, *RMSprop*, dan *Adam*), teknik inisialisasi bobot (*Xavier* dan *He*), serta stabilisasi pelatihan menggunakan *gradient clipping*. Seluruh modul diimplementasikan dari dasar matematika murni menggunakan **NumPy** tanpa framework tingkat tinggi.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- Python 3.10+ (Direkomendasikan Python 3.12).
- Pustaka: `numpy` (wajib), `matplotlib` (opsional untuk plotting visualisasi).
- Tidak memerlukan GPU, akselerator CUDA, atau framework PyTorch/TensorFlow untuk menjalankan seluruh kode inti bab ini.

### Struktur Repositori Praktik:
```
content/level-3-deep-learning/chapter-22/
├── code/
│   ├── README.md
│   ├── chain_rule_demo.py
│   ├── computational_graph.py
│   ├── numerical_derivatives.py
│   ├── manual_backpropagation.py
│   ├── gradient_checking.py
│   ├── batch_gradient_descent.py
│   ├── stochastic_gradient_descent.py
│   ├── mini_batch_gradient_descent.py
│   ├── momentum_optimizer.py
│   ├── rmsprop_optimizer.py
│   ├── adam_optimizer.py
│   ├── learning_rate_comparison.py
│   ├── weight_initialization.py
│   ├── gradient_clipping.py
│   └── numpy_neural_network_training.py
├── notebooks/
│   └── chapter_22_companion.ipynb
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

### Langkah 1: Aturan Rantai Kalkulus (Chain Rule)
Pelajari dekomposisi fungsi komposit bertingkat $y = f(g(h(x)))$ dan akumulasi turunan parsial:
```bash
python content/level-3-deep-learning/chapter-22/code/chain_rule_demo.py
```
- **Fokus Pengamatan**: Perhatikan bagaimana turunan total diperoleh dari perkalian turunan lokal di setiap tahap kalkulasi, dan bagaimana hasil analitik ini cocok hingga presisi $10^{-11}$ dengan metode numerik beda terpusat.

### Langkah 2: Computational Graph & Autograd Sederhana
Bedah representasi graf komputasi asiklik berarah (*Directed Acyclic Graph* - DAG) dan algoritma penelusuran balik (*topological sort*):
```bash
python content/level-3-deep-learning/chapter-22/code/computational_graph.py
```
- **Fokus Pengamatan**: Amati bagaimana operator `+` dan `*` membungkus fungsi `_backward()`. Evaluasi pemanggilan `L.backward()` yang otomatis mendistribusikan gradien ke setiap variabel input secara rekursif.

### Langkah 3: Perbandingan Diferensiasi Numerik (Finite Difference)
Evaluasi metode beda maju (*forward difference*), beda mundur (*backward difference*), dan beda terpusat (*central difference*):
```bash
python content/level-3-deep-learning/chapter-22/code/numerical_derivatives.py
```
- **Fokus Pengamatan**: Buktikan secara matematis dan empiris bahwa beda terpusat memiliki kesalahan pemotongan orde $O(\epsilon^2)$, jauh lebih akurat daripada metode satu sisi $O(\epsilon)$. Amati trade-off antara galat pemotongan (*truncation error*) dan pembulatan numerik (*roundoff error*) saat $\epsilon < 10^{-12}$.

### Langkah 4: Worked Example Propagasi Balik Manual (2-2-1)
Jalankan verifikasi perhitungan manual langkah-demi-langkah pada jaringan neural dua layer:
```bash
python content/level-3-deep-learning/chapter-22/code/manual_backpropagation.py
```
- **Fokus Pengamatan**: Cocokkan nilai skalar $\delta^{[2]}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}}$, $\delta^{[1]}$, dan $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}}$ dengan penurunan matematika di modul teoritis. Buktikan nilai loss berkurang dari $0.0280$ menjadi $0.0245$ setelah tepat satu kali pembaruan gradien.

### Langkah 5: Gradient Checking (Verifikasi Presisi Gradien)
Validasi implementasi backpropagation analitik terhadap aproksimasi numerik dua sisi:
```bash
python content/level-3-deep-learning/chapter-22/code/gradient_checking.py
```
- **Kriteria Keberhasilan**: Error relatif $\frac{\|\mathbf{g}_{\text{analitik}} - \mathbf{g}_{\text{numerik}}\|_2}{\|\mathbf{g}_{\text{analitik}}\|_2 + \|\mathbf{g}_{\text{numerik}}\|_2}$ harus bernilai lebih kecil dari $10^{-7}$. Bila error $\ge 10^{-4}$, dipastikan terdapat cacat logika (*bug*) pada kode backprop.

### Langkah 6: Batch Gradient Descent (BGD)
Amati dinamika pelatihan menggunakan 100% sampel data untuk setiap iterasi pembaruan parameter:
```bash
python content/level-3-deep-learning/chapter-22/code/batch_gradient_descent.py
```
- **Karakteristik**: Kurva loss turun secara mulus (*monotonik*), arah gradien merupakan estimasi eksak populasi, namun biaya komputasi per langkah sangat mahal $O(N)$ jika dataset berukuran raksasa.

### Langkah 7: Stochastic Gradient Descent (SGD)
Simulasikan pembaruan parameter berbasis 1 sampel tunggal ($B = 1$):
```bash
python content/level-3-deep-learning/chapter-22/code/stochastic_gradient_descent.py
```
- **Karakteristik**: Waktu per iterasi sangat instan, fluktuasi stokastik tinggi memungkinkan model meloloskan diri dari minimum lokal dangkal (*saddle points*), namun kurva loss sangat bising dan tidak pernah benar-benar diam di titik optimal.

### Langkah 8: Mini-Batch Gradient Descent (MBGD)
Pelajari kompromi terbaik antara stabilitas BGD dan kecepatan SGD dengan ukuran batch moderat ($B = 32$):
```bash
python content/level-3-deep-learning/chapter-22/code/mini_batch_gradient_descent.py
```
- **Pemanfaatan SIMD/Hardware**: Menjelaskan konsep epoch vs iterasi, dan mengapa ukuran batch berbentuk kelipatan 2 ($16, 32, 64, 128$) memaksimalkan paralelisasi memori cache prosesor.

### Langkah 9: Momentum Optimizer
Uji akumulasi vektor kecepatan eksponensial untuk meredam osilasi liar pada lanskap fungsi jurang (*ravine*):
```bash
python content/level-3-deep-learning/chapter-22/code/momentum_optimizer.py
```
- **Formulasi Fisika**: $v_t = \beta v_{t-1} + (1-\beta) g_t$. Parameter momentum $\beta = 0.9$ secara efektif mengakumulasi momentum ke arah lereng utama dan membatalkan getaran transversal bolak-balik.

### Langkah 10: RMSprop Optimizer (Root Mean Square Propagation)
Pelajari adaptasi learning rate secara independen untuk tiap dimensi parameter:
```bash
python content/level-3-deep-learning/chapter-22/code/rmsprop_optimizer.py
```
- **Skala Adaptif**: $s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2$. Bobot dengan gradien besar diperlambat langkahnya, sementara bobot dengan gradien landai dipercepat langkahnya.

### Langkah 11: Adam Optimizer (Adaptive Moment Estimation)
Bandingkan gabungan momen pertama (arah kecepatan) dan momen kedua (skala variansi) dengan koreksi bias awal (*bias correction*):
```bash
python content/level-3-deep-learning/chapter-22/code/adam_optimizer.py
```
- **Analisis Koreksi**: Mengapa pembagi $1 - \beta^t$ sangat krusial pada iterasi awal $t=1, 2$ saat vektor $m_0$ dan $v_0$ masih diinisialisasi dengan angka nol.

### Langkah 12: Eksperimen Learning Rate & Batas Stabilitas
Eksplorasi respons pelatihan terhadap spektrum $\eta = 0.001$ (sangat lambat), $\eta = 0.1$ (optimal), hingga $\eta = 0.45$ (instabilitas Lipschitz):
```bash
python content/level-3-deep-learning/chapter-22/code/learning_rate_comparison.py
```
- **Analisis Ledakan**: Buktikan kondisi $\eta > \frac{2}{L}$ di mana $L$ adalah konstanta Lipschitz gradien memicu divergensi numerik menuju tak hingga.

### Langkah 13: Inisialisasi Bobot: Zero, Normal, Xavier/Glorot, dan He/Kaiming
Visualisasikan propagasi variansi sinyal forward dan backward melintasi 10 lapisan neural bertingkat:
```bash
python content/level-3-deep-learning/chapter-22/code/weight_initialization.py
```
- **Temuan Kritis**: Inisialisasi nol memicu *symmetry problem*; inisialisasi normal standar memicu ledakan sinyal; Xavier mempertahankan variansi sinyal aktivasi Tanh; dan He mempertahankan variansi sinyal aktivasi ReLU.

### Langkah 14: Gradient Clipping (Norm vs Value)
Simulasikan pencegahan fenomena *exploding gradients*:
```bash
python content/level-3-deep-learning/chapter-22/code/gradient_clipping.py
```
- **Preservasi Arah**: Buktikan secara matematis bahwa *clipping by global norm* menjaga keselarasan kosinus arah gradien 100% identik ($\cos \theta = 1.0$), sedangkan *clipping by value* mengubah vektor arah gradien secara distorsif.

### Langkah 15: Integrasi Penuh Pelatihan MLP (Scratch NumPy)
Latih arsitektur neural network lengkap dari nol pada dataset lingkaran konsentris non-linier:
```bash
python content/level-3-deep-learning/chapter-22/code/numpy_neural_network_training.py
```
- **Evaluasi Akhir**: Perhatikan bagaimana kombinasi forward pass, backprop eksak, inisialisasi He, dan Adam optimizer mampu mencapai akurasi validasi $>95\%$ tanpa modul eksternal.

---

## 3. Latihan Laboratorium Mandiri (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. Diberikan fungsi komposit $f(x) = \ln(x^2 + 4)$. Turunkan turunan parsial analitik $\frac{df}{dx}$ menggunakan aturan rantai kalkulus, dan hitung nilainya pada titik $x = 2$!
   - *Jawaban Singkat*: Misalkan $u = x^2 + 4$, maka $\frac{du}{dx} = 2x$ dan $f(u) = \ln(u)$ dengan $\frac{df}{du} = \frac{1}{u}$. Berdasarkan aturan rantai: $\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx} = \frac{2x}{x^2 + 4}$. Pada $x = 2$, diperoleh $\frac{df}{dx} = \frac{2(2)}{2^2 + 4} = \frac{4}{8} = 0.5$.

### Tingkat 2: Analisis Matematis (Menengah)
1. Pada algoritma Adam optimizer, jelaskan secara matematis mengapa pembagian koreksi bias $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ diperlukan, terutama ketika nilai iterasi $t$ masih kecil!
   - *Jawaban Singkat*: Karena vektor momen diinisialisasi dengan nol ($m_0 = \mathbf{0}$), estimasi rata-rata bergerak eksponensial $m_t = \sum_{i=1}^t (1-\beta_1)\beta_1^{t-i} g_i$ memiliki nilai ekspektasi $\mathbb{E}[m_t] = \mathbb{E}[g_t](1 - \beta_1^t)$. Pada awal pelatihan ($t=1$), jika $\beta_1 = 0.9$, maka $m_1 = 0.1 g_1$ (sangat bias ke arah nol). Pembagian dengan $1 - \beta_1^t = 1 - 0.9^1 = 0.1$ menormalkan kembali skala momen sehingga $\hat{m}_1 = g_1$.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. Tuliskan fungsi Python NumPy murni yang mengimplementasikan *Learning Rate Warmup with Cosine Annealing Decay*. Turunkan formula matematis laju pembelajaran pada setiap langkah pelatihan $t$ dari total langkah $T$ dengan durasi pemanasan $T_{\text{warmup}}$!
   - *Jawaban Singkat*:
     $$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, & \text{jika } t \le T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}}\pi\right)\right), & \text{jika } t > T_{\text{warmup}} \end{cases}$$
     Fungsi ini mencegah guncangan parameter di awal ketika estimasi gradien masih acak, kemudian menurunkan laju secara mulus mendekati optimum lokal.

---

## 4. Mini Project Chapter 22

### Judul Proyek:
**"Training a Multilayer Perceptron from Scratch Using NumPy"**

### Deskripsi Tugas:
Rancang dan bangun sistem pelatihan jaringan saraf tiruan lapis banyak (*Multilayer Perceptron*) lengkap dari awal (*scratch*) menggunakan pustaka **NumPy murni**. Model wajib mengimplementasikan:
1. Inisialisasi bobot adaptif (*He initialization* untuk ReLU, *Xavier initialization* untuk Sigmoid/Tanh).
2. Propagasi maju tervektorisasi (*batch forward propagation*) dengan aktivasi non-linier terisolasi.
3. Propagasi balik analitik (*exact backpropagation*) untuk seluruh layer komputasi.
4. Modul *Gradient Checking* untuk memvalidasi kebenaran turunan analitik dengan batas toleransi error relatif $< 10^{-7}$.
5. Pilihan algoritma optimasi: Stochastic Gradient Descent (SGD), SGD dengan Momentum, dan Adam Optimizer.
6. Mekanisme pelatihan mini-batch dengan *shuffle* data di setiap epoch.
7. Pengujian empiris pada dataset klasifikasi biner non-linier (misalnya: *two concentric circles* atau *moons dataset*), dengan evaluasi metrik akurasi dan visualisasi kurva loss konvergensi.

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Architecture & Graph Design** | 20% | Representasi modular lapisan, alokasi memori bobot/bias yang efisien, dan penerapan inisialisasi bobot yang tepat (He/Xavier). |
| **2. Mathematical Correctness & Gradient Checking** | 25% | Penurunan analitik propagasi balik 100% tepat, terbukti lolos uji *gradient checking* dengan relative error $< 10^{-7}$, dan penanganan stabilitas numerik pada cross-entropy. |
| **3. Optimizer Implementation** | 25% | Implementasi kode mandiri untuk SGD, Momentum, dan Adam Optimizer dengan koreksi bias yang benar dan opsi *gradient clipping*. |
| **4. Experimental Validation & Results** | 20% | Model mampu menyelesaikan klasifikasi non-linier dengan akurasi validasi $\ge 90\%$, kurva loss konvergen secara stabil, dan komparasi optimizer disajikan secara objektif. |
| **5. Code Quality, Reproducibility & Documentation** | 10% | Kode terstruktur rapi berorientasi objek atau fungsi modular bersih, deterministik (`seed=42`), tanpa dependensi deep learning eksternal, dan dilengkapi dokumentasi penjelas yang komprehensif. |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 15 skrip Python di direktori `code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Turunan rantai pada arsitektur 2-2-1 terverifikasi analitik dan numerik.
- [ ] *Gradient checking* membuktikan error relatif $< 10^{-7}$ pada seluruh parameter jaringan.
- [ ] Komparasi konvergensi SGD, Momentum, RMSprop, dan Adam terdokumentasi dengan jelas.
- [ ] Efektivitas inisialisasi He/Xavier dan *gradient clipping* terbukti melalui pengujian empiris.
- [ ] Notebook interaktif `chapter_22_companion.ipynb` berhasil dijalankan secara penuh.
- [ ] Laporan Mini Project mencakup grafik konvergensi loss dan analisis batas keputusan model.

