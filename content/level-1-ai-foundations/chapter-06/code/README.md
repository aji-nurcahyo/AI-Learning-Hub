# Katalog Skrip Python — Chapter 06: Probability, Statistics & Linear Algebra for AI

Direktori ini berisi implementasi kode Python untuk **Chapter 06: Probability, Statistics & Linear Algebra for AI**. Seluruh implementasi dibuat dari awal (*from scratch*) tanpa ketergantungan pustaka luar rumit guna memperkuat pemahaman matematis fondasional mahasiswa.

---

## 1. Daftar Skrip & Deskripsi Fungsional

| Nama File | Topik Utama | Fungsi & Cakupan Algoritma |
|:---|:---|:---|
| [`descriptive_statistics.py`](./descriptive_statistics.py) | Statistik Deskriptif | Perhitungan mean, median, modus, rentang, varians populasi ($\sigma^2$) vs sampel ($s^2$ dengan koreksi Bessel $N-1$), standar deviasi, kuartil ($Q_1, Q_2, Q_3$), IQR, dan deteksi pencilan (outlier) berbasis ambang batas Tukey $1.5 \times \text{IQR}$. |
| [`probability_basics.py`](./probability_basics.py) | Probabilitas Dasar | Ruang sampel, kejadian, aturan komplemen, gabungan, irisan, probabilitas bersyarat $P(A\|B) = \frac{P(A \cap B)}{P(B)}$, uji independensi stokastik, nilai harapan $E[X]$, dan distribusi binomial $P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$. |
| [`bayes_theorem.py`](./bayes_theorem.py) | Teorema Bayes | Formulasi Teorema Bayes lengkap dengan Hukum Probabilitas Total, analisis sensitivitas prior vs posterior, serta demonstrasi pedagogis fenomena *Base Rate Fallacy* pada sensor alarm pintar. |
| [`vector_operations.py`](./vector_operations.py) | Aljabar Vektor | Penjumlahan vektor, perkalian skalar, dot product $\mathbf{u} \cdot \mathbf{v}$, norma vektor ($L_1$, $L_2$ Euclidean, $L_\infty$), normalisasi vektor satuan, kemiripan sudut (Cosine Similarity), dan pengujian ortogonalitas. |
| [`matrix_operations.py`](./matrix_operations.py) | Aljabar Matriks | Representasi dan validasi dimensi matriks, transposisi $\mathbf{A}^T$, perkalian matriks-vektor $\mathbf{A}\mathbf{x}$, perkalian matriks-matriks $\mathbf{A}\mathbf{B}$, pembedaan perkalian Hadamard $\mathbf{A} \odot \mathbf{B}$ vs perkalian formal, dan transformasi geometris 2D. |
| [`derivative_example.py`](./derivative_example.py) | Kalkulus Diferensial | Kemiringan garis singgung, perbandingan turunan analitis vs aproksimasi numerik (Central Difference), pencarian titik stasioner (minimum), turunan parsial multivariat $\frac{\partial g}{\partial x}$ dan $\frac{\partial g}{\partial y}$, vektor gradien $\nabla g$, dan pengantar Chain Rule. |
| [`gradient_descent_demo.py`](./gradient_descent_demo.py) | Optimasi & Loss Function | Algoritma optimasi Gradient Descent univariat $f(x) = (x-3)^2 + 5$, komparasi dinamika laju pembelajaran ($\alpha = 0.01$ lambat, $\alpha = 0.1$ ideal, $\alpha = 1.05$ divergen), serta ilustrasi matematis fungsi loss Mean Squared Error (MSE) dan pembaruan bobot $w$. |

---

## 2. Cara Menjalankan Skrip

Seluruh skrip kompatibel dengan Python 3.10+ (diuji pada Python 3.12.10).

```bash
# Jalankan masing-masing skrip langsung dari terminal
python content/level-1-ai-foundations/chapter-06/code/descriptive_statistics.py
python content/level-1-ai-foundations/chapter-06/code/probability_basics.py
python content/level-1-ai-foundations/chapter-06/code/bayes_theorem.py
python content/level-1-ai-foundations/chapter-06/code/vector_operations.py
python content/level-1-ai-foundations/chapter-06/code/matrix_operations.py
python content/level-1-ai-foundations/chapter-06/code/derivative_example.py
python content/level-1-ai-foundations/chapter-06/code/gradient_descent_demo.py
```

---

## 3. Ringkasan Verifikasi Numerik

| Skrip | Input / Domain | Output Numerik Kunci | Status Validasi |
|:---|:---|:---|:---:|
| `descriptive_statistics.py` | 13 data suhu sensor | Mean: 24.2692 °C, Median: 23.1000 °C, Outlier: [35.0 °C] | **VERIFIED (0 Error)** |
| `probability_basics.py` | 100 paket uji | $P(A \cap B) = 0.03$, $P(A\|B) = 0.10$, $E[X] = 17.50$ ms | **VERIFIED (0 Error)** |
| `bayes_theorem.py` | Prior 1%, TPR 95%, FPR 5% | Evidensi $P(\text{Alarm}) = 5.9\%$, Posterior $P(\text{Fire}\|\text{Alarm}) = 16.10\%$ | **VERIFIED (0 Error)** |
| `vector_operations.py` | Vektor $\mathbf{u}=[3,4,0]$, $\mathbf{v}=[6,8,0]$ | $\|\mathbf{u}\|_2 = 5.0$, $\mathbf{u} \cdot \mathbf{v} = 50.0$, Cosine Sim = 1.0000 | **VERIFIED (0 Error)** |
| `matrix_operations.py` | $\mathbf{X}(3 \times 2) \times \mathbf{W}(2 \times 3)$ | Matriks Output $\mathbf{Y}(3 \times 3)$, Hadamard $\mathbf{P} \odot \mathbf{Q} \neq \mathbf{P}\mathbf{Q}$ | **VERIFIED (0 Error)** |
| `derivative_example.py` | $f(x)=x^2-4x+7$, $g(x,y)$ | $f'(2.0) = 0.0$, Min $f(2)=3.0$, Galat Analitis vs Numerik $< 10^{-10}$ | **VERIFIED (0 Error)** |
| `gradient_descent_demo.py` | $f(x)=(x-3)^2+5$, $\alpha=0.1$ | Konvergen pada langkah 20 ke $x = 2.9424 \approx 3.0$, Bobot MSE $w \to 2.0000$ | **VERIFIED (0 Error)** |

