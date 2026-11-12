# Bab 22: Backpropagation & Optimization

## Algoritma Propagasi Balik, Kalkulus Graf Komputasi, dan Optimasi Jaringan Saraf Tiruan

---

## 1. Hakikat dan Kebutuhan Komputasi Gradien

Pada Bab 21, kita telah memahami bagaimana sinyal informasi mengalir ke depan (*forward propagation*) melintasi lapisan-lapisan neuron tiruan hingga menghasilkan prediksi $\hat{y}$ dan mengukur deviasinya melalui fungsi rugi (*loss function*) $\mathcal{L}(\hat{y}, y)$. Namun, sebuah jaringan neural tiruan tidak akan dapat belajar tanpa mekanisme penyesuaian parameter secara sistematis.

Pertanyaan mendasar dalam pelatihan jaringan saraf tiruan adalah:
> *"Bagaimana cara mengubah setiap bobot $w_{ij}^{[l]}$ dan bias $b_i^{[l]}$ di seluruh lapisan jaringan sedemikian rupa sehingga nilai fungsi rugi $\mathcal{L}$ berkurang secara konsisten?"*

Dalam paradigma optimasi matematis, fungsi rugi mendefinisikan suatu permukaan skalar multivariat berdimensi tinggi di atas ruang parameter $\boldsymbol{\theta} = \{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \dots, \mathbf{W}^{[L]}, \mathbf{b}^{[L]}\}$. Vektor gradien $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ menunjukkan arah peningkatan nilai fungsi yang paling curam. Oleh karena itu, untuk meminimalkan fungsi rugi, parameter jaringan harus digeser ke arah yang berlawanan dengan arah gradien:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

Di sinilah **Backpropagation** (Propagasi Balik) memegang peranan krusial: algoritma ini menyediakan metode deterministik yang sangat efisien untuk menghitung seluruh turunan parsial $\frac{\partial \mathcal{L}}{\partial w}$ dan $\frac{\partial \mathcal{L}}{\partial b}$ secara simultan dengan memanfaatkan **Aturan Rantai Kalkulus (*Chain Rule of Calculus*)** dan struktur penyimpanan cache dari propagasi maju.

```
+---------------------------------------------------------------------------------------------------------+
|                               SIKLUS TERTUTUP PELATIHAN JARINGAN SARAF TIRUAN                          |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       [ Input Data X ] ──────────────────────────────────────────┐                                      |
|                                                                  ▼                                      |
|                            +-------------------------------------------+                                |
|                            |         FORWARD PROPAGATION               |                                |
|                            |   Z^[l] = W^[l] A^[l-1] + b^[l]           |                                |
|                            |   A^[l] = f(Z^[l])                        |                                |
|                            +-------------------------------------------+                                |
|                                                  │                                                      |
|                                                  ▼                                                      |
|                                          [ Prediksi y_hat ]                                             |
|                                                  │                                                      |
|       [ Label Target y ] ────────────────────────┼──────────────────────────────────┐                   |
|                                                  ▼                                  │                   |
|                            +-------------------------------------------+            │                   |
|                            |         EVALUASI FUNGSI RUGI              |            │                   |
|                            |            L(y_hat, y)                    |            │                   |
|                            +-------------------------------------------+            │                   |
|                                                  │                                  │                   |
|                                                  ▼                                  │                   |
|                            +-------------------------------------------+            │                   |
|                            |         BACKWARD PROPAGATION              |            │                   |
|                            |   Aturan Rantai & Perhitungan Gradien     |            │                   |
|                            |   dL/dW^[l], dL/db^[l]                    |            │                   |
|                            +-------------------------------------------+            │                   |
|                                                  │                                  │                   |
|                                                  ▼                                  │                   |
|                            +-------------------------------------------+            │                   |
|                            |         PARAMETER OPTIMIZATION            |            │                   |
|                            |   W^[l] <- W^[l] - eta * dL/dW^[l]        |            │                   |
|                            |   b^[l] <- b^[l] - eta * dL/db^[l]        |            │                   |
|                            +-------------------------------------------+            │                   |
|                                                  │                                  │                   |
|                                                  └──────────────────────────────────┘                   |
|                                                  (Ulangi hingga Konvergensi)                            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Aturan Rantai Kalkulus Multivariat & Computational Graph

### Fondasi Aturan Rantai (Chain Rule)
Secara kalkulus dasar satu dimensi, jika sebuah variabel $y$ bergantung pada $u$, dan $u$ bergantung pada $x$, maka turunan total $y$ terhadap $x$ merupakan perkalian turunan lokal:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Pada kasus multivariat di mana sebuah fungsi bergantung pada beberapa variabel antara $\{u_1, u_2, \dots, u_k\}$, aturan rantai diperluas menjadi penjumlahan atas seluruh lintasan kontribusi (*multivariate chain rule*):

$$\frac{\partial y}{\partial x} = \sum_{j=1}^{k} \frac{\partial y}{\partial u_j} \frac{\partial u_j}{\partial x}$$

```
+---------------------------------------------------------------------------------------------------------+
|                                ATURAN RANTAI MULTIVARIAT (POHON DEPENDENSI)                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                                                ( y )                                                    |
|                                               /     \                                                   |
|                          dy/du_1             /       \            dy/du_2                               |
|                                             ▼         ▼                                                 |
|                                          ( u_1 )   ( u_2 )                                              |
|                                             \         /                                                 |
|                          du_1/dx             \       /            du_2/dx                               |
|                                               ▼     ▼                                                   |
|                                                ( x )                                                    |
|                                                                                                         |
|                   Formula Akumulasi: dy/dx = (dy/du_1 * du_1/dx) + (dy/du_2 * du_2/dx)                  |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Konsep Computational Graph (Graf Komputasi)
Setiap ekspresi matematika dalam jaringan saraf tiruan dapat direpresentasikan sebagai **Computational Graph**, yaitu graf berarah tanpa siklus (*Directed Acyclic Graph* - DAG):
- **Node (Simpul)**: Merepresentasikan variabel masukan, bobot, atau hasil operasi perantara.
- **Edge (Sisi Berarah)**: Merepresentasikan aliran data atau fungsi transformasi.

```
+---------------------------------------------------------------------------------------------------------+
|                               KOMPUTASI GRAF UNTUK EKSPRESI: L = (w1*x1 + w2*x2 + b)^2                  |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|     [ w1 ] ───┐                                                                                         |
|               ├─ (*) ───> [ p1 = w1*x1 ] ──┐                                                            |
|     [ x1 ] ───┘                            │                                                            |
|                                            ├─ (+) ───> [ s = p1 + p2 ] ──┐                              |
|     [ w2 ] ───┐                            │                             │                              |
|               ├─ (*) ───> [ p2 = w2*x2 ] ──┘                             ├─ (+) ──> [ z ] ── (sq) ──> L |
|     [ x2 ] ───┘                                                          │                              |
|                                                                          │                              |
|     [  b ] ──────────────────────────────────────────────────────────────┘                              |
|                                                                                                         |
|     ALIRAN FORWARD  ─────────────────────────────────────────────────────────────────────────────>      |
|     ALIRAN BACKWARD <─────────────────────────────────────────────────────────────────────────────      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

Dalam penelusuran balik (*backward pass*), algoritma menggunakan urutan topologis terbalik (*reverse topological sort*). Setiap node menerima gradien dari simpul anak di kanannya (*incoming gradient*), mengalikannya dengan turunan lokal operasi simpul tersebut, lalu meneruskan gradien ke simpul induknya (*outgoing gradient*):

$$\text{Gradien Input} = \text{Gradien Luaran} \times \text{Turunan Lokal}$$

```
+---------------------------------------------------------------------------------------------------------+
|                               MEKANISME PENELUSURAN NODE PADA BACKPROPAGATION                           |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|                         [ Node Komputasi: z = g(x, y) ]                                                 |
|                                                                                                         |
|         x ────> [ x ] ──────────┐                 ┌──────────> z                                        |
|                                 │                 │                                                     |
|                                 ▼                 │                                                     |
|                               ( g ) ──────────────┘                                                     |
|                                 ▲                                                                       |
|                                 │                                                                       |
|         y ────> [ y ] ──────────┘                                                                       |
|                                                                                                         |
|         ==================== ALIRAN BALIK (BACKWARD PASS) ====================                          |
|                                                                                                         |
|         dL/dx = dL/dz * dz/dx <─── [ dz/dx ] <─── [ dL/dz ] (Gradien Masuk dari Lapisan Depan)          |
|                                                                                                         |
|         dL/dy = dL/dz * dz/dy <─── [ dz/dy ] ───┘                                                       |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 3. Diferensiasi Numerik vs Simbolik vs Otomatis

Terdapat tiga paradigma utama dalam mengevaluasi turunan matematis pada sistem komputer:

```
+---------------------------------------------------------------------------------------------------------+
|                               PERBANDINGAN 3 PARADIGMA DIFERENSIASI KOMPUTASI                           |
+---------------------------------------------------------------------------------------------------------+
|  Paradigma             Prinsip Kerja                   Kelebihan                 Kekurangan             |
|  -----------------------------------------------------------------------------------------------------  |
|  1. Numerik            Hampiran selisih hingga         Sangat mudah dibuat;      Lambat O(N); rentan    |
|     (Finite Diff)      f'(x) ~ (f(x+e)-f(x-e))/(2e)   tidak butuh graf simbolik roundoff/truncation err |
|                                                                                                         |
|  2. Simbolik           Manipulasi ekspresi aljabar     Menghasilkan formula      Expression swell;      |
|     (Symbolic Diff)    seperti SymPy/Mathematica       turunan analitik eksak    ledakan memori graf    |
|                                                                                                         |
|  3. Otomatis           Dekomposisi operasi elementer   Eksak hingga batas float; Memerlukan perancangan |
|     (Automatic Diff)   pada graf komputasi (Autograd)  efisien O(1) per node     struktur data graf DAG |
+---------------------------------------------------------------------------------------------------------+
```

### Analisis Galat Metode Beda Hingga (Finite Difference)
Pada metode beda hingga dua sisi (*central difference*):

$$\frac{df}{dx} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

Berdasarkan ekspansi deret Taylor:
- $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{\epsilon^2}{2} f''(x) + \frac{\epsilon^3}{6} f'''(x) + \mathcal{O}(\epsilon^4)$
- $f(x - \epsilon) = f(x) - \epsilon f'(x) + \frac{\epsilon^2}{2} f''(x) - \frac{\epsilon^3}{6} f'''(x) + \mathcal{O}(\epsilon^4)$

Dengan mengurangkan kedua persamaan di atas:
$$f(x + \epsilon) - f(x - \epsilon) = 2\epsilon f'(x) + \frac{\epsilon^3}{3} f'''(x) + \mathcal{O}(\epsilon^5)$$

$$\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} = f'(x) + \mathcal{O}(\epsilon^2)$$

Terbukti bahwa *central difference* memiliki kesalahan pemotongan orde $\mathcal{O}(\epsilon^2)$, sedangkan metode satu sisi (*forward difference*) hanya berorde $\mathcal{O}(\epsilon)$. Namun, bila $\epsilon$ dipilih terlalu kecil ($\epsilon < 10^{-12}$), terjadi pembatalan katastrofik (*catastrophic cancellation*) akibat batas presisi mesin floating-point IEEE-754.

```
+---------------------------------------------------------------------------------------------------------+
|                               TRADE-OFF GALAT NUMERIK FINITE DIFFERENCE                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|    Total Galat |                                                                                        |
|                |\                                                 / Truncation Error O(e^2)             |
|                | \                                               /                                      |
|                |  \                                             /                                       |
|                |   \ Roundoff Error (Presisi Mesin)            /                                        |
|                |    \                                         /                                         |
|                |     \                                       /                                          |
|                |      \_________________                    /                                           |
|                |                        \                  /                                            |
|                |                         \________________/ <── Titik Optimal: e ~ 1e-7                 |
|                +--------------------------------------------------------> Nilai Epsilon (e)             |
|                  1e-15        1e-11        1e-7         1e-3                                            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Worked Numerical Example: Propagasi Balik Manual Langkah-demi-Langkah

Mari kita bedah perhitungan numerik manual secara mendalam pada arsitektur jaringan saraf sederhana: **2 Masukan $\to$ 2 Hidden Neuron $\to$ 1 Luaran Tunggal**.

```
+---------------------------------------------------------------------------------------------------------+
|                               ARSITEKTUR LENGKAP WORKED NUMERICAL EXAMPLE                               |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|      Lapisan Masukan           Lapisan Tersembunyi [1]              Lapisan Luaran [2]                  |
|                                                                                                         |
|         ( x_1 = 0.5 ) ───────┬────────> ( h_1 ) ────────┐                                               |
|                              │\       /                 │\                                              |
|                              │ \     /                  │ \                                             |
|                              │  \   /                   │  \                                            |
|                              │   \ /                    │   \                                           |
|                              │    X                     │    ───> ( o_1 ) ───> y_pred                   |
|                              │   / \                    │   /                                           |
|                              │  /   \                   │  /                                            |
|                              │ /     \                  │ /                                             |
|         ( x_2 = 0.8 ) ───────┴────────> ( h_2 ) ────────┘                                               |
|                                                                                                         |
|      Konfigurasi Awal:                                                                                  |
|      W^[1] = [[0.15, 0.25],    b^[1] = [[0.35, 0.35]]   Target Sejati: y = 1.0                          |
|               [0.20, 0.30]]                                                                             |
|      W^[2] = [[0.40],          b^[2] = [[0.60]]         Aktivasi: Sigmoid pada seluruh layer            |
|               [0.50]]                                   Fungsi Rugi: MSE = 0.5 * (y_pred - y)^2         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Tahap 1: Propagasi Maju (Forward Pass)
1. **Hidden Layer**:
   $$z_1^{[1]} = w_{11}^{[1]} x_1 + w_{21}^{[1]} x_2 + b_1^{[1]} = (0.15 \times 0.5) + (0.20 \times 0.8) + 0.35 = 0.075 + 0.160 + 0.35 = 0.5850$$
   $$z_2^{[1]} = w_{12}^{[1]} x_1 + w_{22}^{[1]} x_2 + b_2^{[1]} = (0.25 \times 0.5) + (0.30 \times 0.8) + 0.35 = 0.125 + 0.240 + 0.35 = 0.7150$$
   $$a_1^{[1]} = \sigma(z_1^{[1]}) = \frac{1}{1 + e^{-0.5850}} \approx 0.642211$$
   $$a_2^{[1]} = \sigma(z_2^{[1]}) = \frac{1}{1 + e^{-0.7150}} \approx 0.671515$$

2. **Output Layer**:
   $$z_1^{[2]} = w_{11}^{[2]} a_1^{[1]} + w_{21}^{[2]} a_2^{[1]} + b_1^{[2]} = (0.40 \times 0.642211) + (0.50 \times 0.671515) + 0.60 = 0.256884 + 0.335758 + 0.60 = 1.192642$$
   $$\hat{y} = a_1^{[2]} = \sigma(z_1^{[2]}) = \frac{1}{1 + e^{-1.192642}} \approx 0.767226$$

3. **Nilai Rugi Awal**:
   $$\mathcal{L} = \frac{1}{2} (\hat{y} - y)^2 = \frac{1}{2} (0.767226 - 1.0)^2 = \frac{1}{2} (-0.232774)^2 \approx 0.027092$$

### Tahap 2: Propagasi Balik pada Output Layer (Layer 2)
Definisikan suku galat (*error term*) $\delta_1^{[2]}$:
$$\delta_1^{[2]} = \frac{\partial \mathcal{L}}{\partial z_1^{[2]}} = \frac{\partial \mathcal{L}}{\partial a_1^{[2]}} \cdot \frac{\partial a_1^{[2]}}{\partial z_1^{[2]}} = (\hat{y} - y) \cdot [\hat{y}(1 - \hat{y})]$$
$$\delta_1^{[2]} = (0.767226 - 1.0) \times [0.767226 \times (1 - 0.767226)] = -0.232774 \times 0.178598 \approx -0.041573$$

Maka gradien terhadap bobot dan bias lapisan output adalah:
$$\frac{\partial \mathcal{L}}{\partial w_{11}^{[2]}} = \delta_1^{[2]} \cdot a_1^{[1]} = -0.041573 \times 0.642211 \approx -0.026699$$
$$\frac{\partial \mathcal{L}}{\partial w_{21}^{[2]}} = \delta_1^{[2]} \cdot a_2^{[1]} = -0.041573 \times 0.671515 \approx -0.027917$$
$$\frac{\partial \mathcal{L}}{\partial b_1^{[2]}} = \delta_1^{[2]} = -0.041573$$

### Tahap 3: Propagasi Balik pada Hidden Layer (Layer 1)
Galat diproyeksikan kembali ke lapisan tersembunyi melalui transpos matriks bobot:
$$\delta_1^{[1]} = (\delta_1^{[2]} \cdot w_{11}^{[2]}) \cdot [a_1^{[1]}(1 - a_1^{[1]})] = (-0.041573 \times 0.40) \times [0.642211 \times 0.357789] = -0.016629 \times 0.229776 \approx -0.003821$$
$$\delta_2^{[1]} = (\delta_1^{[2]} \cdot w_{21}^{[2]}) \cdot [a_2^{[1]}(1 - a_2^{[1]})] = (-0.041573 \times 0.50) \times [0.671515 \times 0.328485] = -0.020787 \times 0.220582 \approx -0.004585$$

Gradien terhadap bobot dan bias hidden layer:
$$\frac{\partial \mathcal{L}}{\partial w_{11}^{[1]}} = \delta_1^{[1]} \cdot x_1 = -0.003821 \times 0.5 = -0.001911$$
$$\frac{\partial \mathcal{L}}{\partial w_{21}^{[1]}} = \delta_1^{[1]} \cdot x_2 = -0.003821 \times 0.8 = -0.003057$$
$$\frac{\partial \mathcal{L}}{\partial w_{12}^{[1]}} = \delta_2^{[1]} \cdot x_1 = -0.004585 \times 0.5 = -0.002293$$
$$\frac{\partial \mathcal{L}}{\partial w_{22}^{[1]}} = \delta_2^{[1]} \cdot x_2 = -0.004585 \times 0.8 = -0.003668$$
$$\frac{\partial \mathcal{L}}{\partial b_1^{[1]}} = \delta_1^{[1]} = -0.003821, \quad \frac{\partial \mathcal{L}}{\partial b_2^{[1]}} = \delta_2^{[1]} = -0.004585$$

### Tahap 4: Pembaruan Bobot (Weight Update)
Menggunakan learning rate $\eta = 0.5$:
$$w_{11}^{[2](\text{baru})} = 0.40 - 0.5 \times (-0.026699) = 0.413350$$
$$w_{21}^{[2](\text{baru})} = 0.50 - 0.5 \times (-0.027917) = 0.513959$$
$$b_1^{[2](\text{baru})} = 0.60 - 0.5 \times (-0.041573) = 0.620787$$

Jika kita menghitung kembali forward pass dengan parameter baru ini, luaran prediksi meningkat menjadi $\hat{y}_{\text{baru}} \approx 0.7812$, dan fungsi rugi berkurang dari **$0.027092$** menjadi **$0.023938$**. Terbukti secara analitik dan empiris bahwa satu langkah gradien menurunkan nilai galat secara efektif!

---

## 5. Gradient Checking (Verifikasi Gradien Numerik)

Salah satu sumber kesalahan paling berbahaya dalam implementasi deep learning dari awal (*from scratch*) adalah **silent gradient bugs** — kode dapat dieksekusi tanpa galat sintaksis, bahkan fungsi rugi tampak menurun perlahan, namun gradien analitik yang dihitung salah sehingga model tidak pernah mencapai performa optimal.

Protokol ilmiah standar industri untuk menjamin keabsahan kalkulus backpropagation adalah **Gradient Checking**.

```
+---------------------------------------------------------------------------------------------------------+
|                               PROTOKOL VERIFIKASI GRADIENT CHECKING                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  1. Flatten seluruh parameter matriks W^[l] dan b^[l] menjadi satu vektor panjang theta in R^D          |
|  2. Hitung vektor gradien analitik g_analitik = [dL/dW, dL/db] melalui backpropagation                  |
|  3. Untuk setiap parameter theta_i:                                                                     |
|     a. Tambahkan perturbasi: theta_plus = theta; theta_plus[i] += epsilon                               |
|     b. Kurangkan perturbasi: theta_minus = theta; theta_minus[i] -= epsilon                             |
|     c. Evaluasi loss: L_plus = Loss(theta_plus), L_minus = Loss(theta_minus)                            |
|     d. Hitung hampiran numerik: g_numerik[i] = (L_plus - L_minus) / (2 * epsilon)                       |
|  4. Hitung kesalahan relatif (Relative Error):                                                          |
|                                                                                                         |
|                          || g_analitik - g_numerik ||_2                                                 |
|        Relative Error = ─────────────────────────────────                                               |
|                          || g_analitik ||_2 + || g_numerik ||_2                                         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Panduan Interpretasi Kriteria Relative Error:
- **$\text{Relative Error} \le 10^{-7}$**: **Sempurna (PASS)**. Turunan analitik 100% tepat dan konsisten dengan kalkulus batas presisi floating-point.
- **$10^{-7} < \text{Relative Error} \le 10^{-5}$**: **Dapat Diterima**. Kemungkinan disebabkan oleh diskontinuitas turunan pada titik tertentu (misalnya titik $z=0$ pada ReLU).
- **$10^{-5} < \text{Relative Error} \le 10^{-3}$**: **Perlu Diinvestigasi**. Periksa apakah pemilihan $\epsilon$ tepat ($10^{-7}$ direkomendasikan) atau terdapat galat formulasi fungsi aktivasi.
- **$\text{Relative Error} > 10^{-3}$**: **GAGAL TOTAL (FAIL)**. Dipastikan terdapat kesalahan matematis atau transpos matriks terbalik pada kode backpropagation.

> [!WARNING]
> Jangan pernah mengaktifkan *gradient checking* selama proses pelatihan model yang sebenarnya! Karena perhitungan beda numerik mengevaluasi forward pass sebanyak $2 \times D$ kali (di mana $D$ adalah total parameter jaringan), waktu komputasinya berorde $\mathcal{O}(D)$ dan akan memperlambat pelatihan ribuan kali lipat. Gradient checking hanya digunakan saat proses debugging dan pengujian unit (*unit testing*).

---

## 6. Taksonomi Varian Gradient Descent

Dalam melatih jaringan dengan kumpulan data berukuran $N$ sampel, terdapat tiga pendekatan utama untuk memperbarui parameter:

```
+---------------------------------------------------------------------------------------------------------+
|                               KOMPARASI 3 VARIAN ALGORITMA GRADIENT DESCENT                             |
+---------------------------------------------------------------------------------------------------------+
|  Dimensi Pembanding     Batch GD (BGD)              Stochastic GD (SGD)         Mini-Batch GD (MBGD)    |
|  -----------------------------------------------------------------------------------------------------  |
|  Ukuran Batch (B)       B = N (Seluruh dataset)     B = 1 (1 sampel acak)       1 < B < N (misal 32-128)|
|  Frekuensi Update       1x per epoch                Nx per epoch                (N/B)x per epoch        |
|  Karakteristik Kurva    Mulus & monotonik turun     Sangat fluktuatif & bising  Fluktuasi moderat       |
|  Waktu per Iterasi      Lambat untuk data besar     Sangat cepat                Cepat & optimal SIMD    |
|  Kemampuan Lolos Minima Sulit (terjebak sadel)      Mudah loncat karena bising  Bagus & terkontrol      |
|  Efisiensi Memori       Tinggi (OOM jika data besar)Rendah & ringan             Ideal untuk GPU/Cache   |
+---------------------------------------------------------------------------------------------------------+
```

```
+---------------------------------------------------------------------------------------------------------+
|                               TRAJEKTORI OPTIMASI PADA KONTUR FUNGSI RUGI                               |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|       [ Batch Gradient Descent ]                 [ Stochastic Gradient Descent ]                        |
|       (Lintasan Halus Langsung)                  (Lintasan Zig-Zag Liar & Bising)                       |
|                                                                                                         |
|          \      |      /                            \      |      /                                     |
|           \  ───────  /                              \  ~^~/\~~^  /                                     |
|            \ (     ) /                                \ ( ~~~ )  /                                      |
|             \   *   /  <── Minimum Global              \   *   /  <── Melompat di sekitar min           |
|              ───────                                    ───────                                         |
|                                                                                                         |
|                                [ Mini-Batch Gradient Descent ]                                          |
|                                (Lintasan Terarah dengan Sedikit Noise)                                  |
|                                                                                                         |
|                                         \      |      /                                                 |
|                                          \  ───/\───  /                                                 |
|                                           \ (  \/ )  /                                                  |
|                                            \   *   /                                                    |
|                                             ───────                                                     |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Konsep Epoch, Iterasi, dan Ukuran Batch:
- **Epoch**: Satu siklus penelusuran penuh di mana setiap sampel dalam seluruh dataset telah dipelajari tepat satu kali oleh model.
- **Batch Size ($B$)**: Jumlah sampel data yang diproses secara simultan sebelum parameter bobot dan bias diperbarui.
- **Iteration (Langkah Pembaruan)**: Jumlah total pembaruan parameter yang dilakukan. Dalam 1 epoch dengan $N$ sampel data dan ukuran batch $B$, jumlah iterasi per epoch adalah:
  $$\text{Iterations per Epoch} = \left\lceil \frac{N}{B} \right\rceil$$

---

## 7. Algoritma Optimasi Tingkat Lanjut

Meskipun Stochastic Gradient Descent sederhana, algoritma ini menderita kelemahan parah ketika berhadapan dengan fungsi jurang (*anisotropic ravine*), di mana kelengkungan permukaan di satu arah jauh lebih terjal daripada arah lainnya. SGD standar akan berosilasi hebat melintasi lereng yang terjal tanpa membuat kemajuan berarti di sepanjang dasar lembah.

```
+---------------------------------------------------------------------------------------------------------+
|                               EVOLUSI ALGORITMA OPTIMASI DEEP LEARNING                                  |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|        [ Vanilla SGD ] ───> Menambahkan Inersia Fisika ───────────> [ SGD with Momentum ]               |
|              │                                                              │                           |
|              │ Adaptasi Per-Parameter                                      │ Penggabungan               |
|              ▼                                                              ▼                           |
|        [ RMSprop ]     ───────────────────────────────────────────> [ ADAM OPTIMIZER ]                  |
|        (Skala Kuadrat Gradien)                                      (Momen 1 + Momen 2)                 |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 1. Momentum Optimizer
Terinspirasi dari hukum fisika gerak benda dengan massa inersia. Algoritma mengakumulasi vektor kecepatan (*velocity*) $v_t$ dari gradien masa lalu:

$$v_t = \beta v_{t-1} + (1 - \beta) g_t$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta v_t$$

Di mana $\beta \in [0, 1)$ (biasanya $\beta = 0.9$) adalah koefisien friksi/momentum. Jika gradien menunjuk ke arah yang sama secara konsisten, kecepatan akan terus bertambah; sebaliknya, jika gradien berosilasi bolak-balik, komponen yang berlawanan akan saling menghilangkan.

### 2. RMSprop Optimizer (Hinton, 2012)
RMSprop mengatasi masalah penurunan learning rate yang terlalu cepat pada Adagrad dengan menggunakan rata-rata bergerak eksponensial (*exponential moving average*) dari kuadrat gradien:

$$s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \odot g_t$$

Di mana $\beta_2 \approx 0.99$ dan $\epsilon \approx 10^{-8}$ mencegah pembagian dengan nol. Parameter yang memiliki gradien historis besar akan dibagi dengan angka besar (langkah diperkecil), sedangkan parameter yang jarang bergerak akan dipercepat.

### 3. Adam Optimizer (Kingma & Ba, 2014)
Adam (*Adaptive Moment Estimation*) mengombinasikan keunggulan Momentum (momen pertama / rata-rata gradien) dan RMSprop (momen kedua / variansi gradien tak terpusat):

1. **Momen Pertama (Arah)**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. **Momen Kedua (Skala)**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3. **Koreksi Bias (Bias Correction)**:
   Karena $m_0 = \mathbf{0}$ dan $v_0 = \mathbf{0}$, estimasi momen di awal pelatihan sangat bias mendekati nol. Koreksi bias menormalkan kembali skala:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. **Pembaruan Parameter**:
   $$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t$$

Nilai hiperparameter standar yang terbukti tangguh secara empiris: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

---

## 8. Dinamika Pelatihan & Tantangan Gradien

### Masalah Vanishing dan Exploding Gradients
Pada jaringan neural yang memiliki banyak lapisan ($L \gg 1$), turunan parsial terhadap lapisan awal $l=1$ melibatkan perkalian matriks bobot dan turunan aktivasi berulang kali:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} \propto \prod_{k=2}^{L} \left( \mathbf{W}^{[k]T} \odot f'(z^{[k]}) \right)$$

- **Vanishing Gradients**: Jika bobot terlalu kecil atau fungsi aktivasi memiliki turunan $< 1$ (seperti Sigmoid dengan $\max f' = 0.25$), besaran gradien akan menyusut secara eksponensial menuju nol. Akibatnya, lapisan-lapisan pertama berhenti belajar sama sekali.
- **Exploding Gradients**: Jika bobot terlalu besar ($> 1$), besaran gradien akan berlipat ganda secara eksponensial hingga menghasilkan nilai `inf` atau `NaN` (Floating-Point Overflow).

```
+---------------------------------------------------------------------------------------------------------+
|                               SPEKTRUM MASALAH PENYALURAN GRADIEN PADA DEEP NETWORKS                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Lapisan 10       Lapisan 9        Lapisan 8                   Lapisan 2        Lapisan 1              |
|   (Dekat Loss)                                                                   (Dekat Input)          |
|                                                                                                         |
|   [ VANISHING GRADIENTS ]                                                                               |
|   ||g|| = 1.0 ────> ||g|| = 0.25 ───> ||g|| = 0.0625 ─────────> ||g|| = 0.000001 ──> ||g|| = 0.0000000 |
|   (Lapisan awal tidak mendapatkan pembaruan sinyal sama sekali)                                         |
|                                                                                                         |
|   [ EXPLODING GRADIENTS ]                                                                               |
|   ||g|| = 1.0 ────> ||g|| = 4.0 ────> ||g|| = 16.0 ───────────> ||g|| = 10^5 ──────> NaN / Overflow     |
|   (Langkah optimasi meledak dan menghancurkan parameter model)                                          |
|                                                                                                         |
|   [ ALIRAN SEIMBANG (XAVIER / HE INITIALIZATION) ]                                                      |
|   ||g|| = 1.0 ────> ||g|| = 0.98 ───> ||g|| = 1.02 ───────────> ||g|| = 0.99 ──────> ||g|| = 1.01     |
|   (Variansi sinyal terjaga stabil dari lapisan luaran hingga lapisan masukan)                           |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Inisialisasi Bobot: Menjaga Variansi Sinyal
Kunci utama menjaga kestabilan aliran forward dan backward adalah memilih distribusi inisialisasi bobot sedemikian rupa sehingga variansi aktivasi tetap konstan di setiap layer: $\text{Var}(a^{[l]}) \approx \text{Var}(a^{[l-1]})$.

```
+---------------------------------------------------------------------------------------------------------+
|                               TAKSONOMI METODE INISIALISASI BOBOT PARAMETER                             |
+---------------------------------------------------------------------------------------------------------+
|  Metode                Formulasi Distribusi Bobot W                Target Aktivasi   Dampak Buruk       |
|  -----------------------------------------------------------------------------------------------------  |
|  Zero Init             W = 0                                       Tidak ada         Symmetry problem;  |
|                                                                                      semua neuron sama  |
|  Standard Normal       W ~ N(0, 1) atau N(0, 0.01)                 Tidak stabil      Vanishing /        |
|                                                                                      Exploding signal   |
|  Xavier / Glorot       W ~ N(0, 2 / (n_in + n_out))                Tanh / Sigmoid    Menjaga variansi   |
|  (Glorot & Bengio, 10)                                                               aktivasi simetris  |
|  He / Kaiming          W ~ N(0, 2 / n_in)                          ReLU / Leaky ReLU Mengimbangi 50%    |
|  (He et al., 2015)                                                                   neuron yang mati   |
+---------------------------------------------------------------------------------------------------------+
```

### Gradient Clipping
Ketika melatih model dengan arsitektur dalam atau jaringan berulang (*recurrent*), ledakan gradien dapat dimitigasi secara efektif dengan membatasi besaran gradien sebelum langkah pembaruan bobot:

```
+---------------------------------------------------------------------------------------------------------+
|                               KOMPARASI CLIPPING BY VALUE VS CLIPPING BY GLOBAL NORM                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  1. CLIPPING BY VALUE:                                                                                  |
|     g_i <- min(max(g_i, -c), c)                                                                         |
|     Karakteristik: Memotong setiap elemen secara independen.                                            |
|     Bahaya: MENGUBAH ARAH VEKTOR GRADIEN! (Distorsi sudut kemiringan lereng)                             |
|                                                                                                         |
|  2. CLIPPING BY GLOBAL NORM:                                                                            |
|     Jika ||g||_2 > c:                                                                                   |
|         g <- g * (c / ||g||_2)                                                                          |
|     Karakteristik: Menskalakan seluruh vektor dengan faktor pembagi yang sama.                          |
|     Keunggulan: MEMPERTAHANKAN ARAH VEKTOR GRADIEN 100% UTUH! (Cos theta = 1.0)                        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 9. Rangkuman & Latihan Soal Mandiri

### Intisari Bab:
1. **Backpropagation** adalah aplikasi efisien dari aturan rantai kalkulus multivariat yang memanfaatkan nilai perantara (*cache*) dari propagasi maju untuk menghitung gradien analitik.
2. **Computational Graph** menstrukturkan komputasi sebagai graf terarah asiklik (DAG), memungkinkan autograd mengevaluasi turunan secara lokal pada setiap node.
3. **Gradient Checking** membuktikan keabsahan implementasi propagasi balik dengan toleransi kesalahan relatif $< 10^{-7}$.
4. **Mini-Batch GD** mengombinasikan efisiensi paralelisasi matriks perangkat keras dengan regularisasi implisit dari kebisingan stokastik.
5. **Adam Optimizer** memadukan inersia gerak momentum dengan penyesuaian laju individual per dimensi dan koreksi bias awal.
6. **Inisialisasi He** adalah standar wajib untuk jaringan berbasis aktivasi ReLU guna mempertahankan variansi propagasi sinyal.

---

### Latihan Soal 3-Tingkat Mandiri

#### Tingkat 1: Pemahaman Konseptual (Dasar)
1. Jelaskan mengapa menginisialisasi seluruh matriks bobot jaringan saraf tiruan dengan nilai angka nol ($W = \mathbf{0}$) menyebabkan kegagalan pembelajaran yang fatal (*symmetry problem*)!
   - **Solusi**: Jika seluruh bobot diinisialisasi dengan angka nol, maka untuk setiap masukan $\mathbf{x}$, nilai kombinasi linier $z_j = \mathbf{w}_j^T \mathbf{x} + b$ akan identik untuk semua neuron di lapisan tersembunyi. Akibatnya, semua neuron menghasilkan luaran aktivasi yang sama dan menerima gradien yang persis sama saat propagasi balik. Seluruh neuron akan terus diperbarui secara simetris dan berperilaku sebagai satu neuron tunggal, menghilangkan kapasitas representasi jaringan lapis banyak.

#### Tingkat 2: Analisis Matematis (Menengah)
1. Diberikan fungsi luaran $a = \sigma(z) = \frac{1}{1 + e^{-z}}$. Buktikan secara analitik bahwa turunan pertamanya memenuhi identitas:
   $$\frac{da}{dz} = a(1 - a)$$
   - **Solusi**:
     $$\frac{da}{dz} = \frac{d}{dz} (1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$
     Tuliskan kembali suku pembilang:
     $$\frac{da}{dz} = \left(\frac{1}{1 + e^{-z}}\right) \cdot \left(\frac{e^{-z}}{1 + e^{-z}}\right) = \left(\frac{1}{1 + e^{-z}}\right) \cdot \left(\frac{(1 + e^{-z}) - 1}{1 + e^{-z}}\right) = a \cdot (1 - a)$$
     Terbukti! Identitas ini sangat berharga secara komputasi karena turunan dapat dihitung langsung dari luaran aktivasi $a$ yang telah tersimpan di memori forward cache tanpa perlu menghitung ulang fungsi eksponensial $e^{-z}$.

#### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. Pada optimasi fungsi kuadratik $f(w) = \frac{1}{2} L w^2$, buktikan batas maksimum learning rate $\eta$ agar algoritma Gradient Descent konvergen ke titik minimum $w^* = 0$!
   - **Solusi**:
     Gradien fungsi adalah $f'(w) = Lw$. Aturan pembaruan parameter:
     $$w_{t+1} = w_t - \eta (L w_t) = (1 - \eta L) w_t$$
     Dengan rekursi: $w_t = (1 - \eta L)^t w_0$.
     Agar barisan $w_t \to 0$ saat $t \to \infty$, faktor pengali harus berada di dalam cakram satuan:
     $$|1 - \eta L| < 1 \iff -1 < 1 - \eta L < 1 \iff 0 < \eta L < 2 \iff 0 < \eta < \frac{2}{L}$$
     Jika $\eta \ge \frac{2}{L}$, nilai parameter akan berosilasi dengan amplitudo membesar dan meledak (*diverge*). Nilai konvergensi tercepat tercapai ketika $1 - \eta L = 0$, yaitu $\eta = \frac{1}{L}$.

---

### Mini Project Chapter 22

#### Judul Tugas Mini Project:
**"Training a Multilayer Perceptron from Scratch Using NumPy"**

#### Deskripsi Tugas:
Rancang dan bangun sistem pelatihan jaringan saraf tiruan lapis banyak (*Multilayer Perceptron*) lengkap dari awal (*scratch*) menggunakan pustaka **NumPy murni**. Model wajib mengimplementasikan:
1. Inisialisasi bobot adaptif (*He initialization* untuk ReLU, *Xavier initialization* untuk Sigmoid/Tanh).
2. Propagasi maju tervektorisasi (*batch forward propagation*) dengan aktivasi non-linier terisolasi.
3. Propagasi balik analitik (*exact backpropagation*) untuk seluruh layer komputasi.
4. Modul *Gradient Checking* untuk memvalidasi kebenaran turunan analitik dengan batas toleransi error relatif $< 10^{-7}$.
5. Pilihan algoritma optimasi: Stochastic Gradient Descent (SGD), SGD dengan Momentum, dan Adam Optimizer.
6. Mekanisme pelatihan mini-batch dengan *shuffle* data di setiap epoch.
7. Pengujian empiris pada dataset klasifikasi biner non-linier lingkaran konsentris (*concentric circles*), dengan evaluasi akurasi dan visualisasi kurva loss konvergensi.

#### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Architecture & Graph Design** | 20% | Representasi modular lapisan, alokasi memori bobot/bias yang efisien, dan penerapan inisialisasi bobot yang tepat (He/Xavier). |
| **2. Mathematical Correctness & Gradient Checking** | 25% | Penurunan analitik propagasi balik 100% tepat, terbukti lolos uji *gradient checking* dengan relative error $< 10^{-7}$, dan penanganan stabilitas numerik pada cross-entropy. |
| **3. Optimizer Implementation** | 25% | Implementasi kode mandiri untuk SGD, Momentum, dan Adam Optimizer dengan koreksi bias yang benar dan opsi *gradient clipping*. |
| **4. Experimental Validation & Results** | 20% | Model mampu menyelesaikan klasifikasi non-linier dengan akurasi validasi $\ge 90\%$, kurva loss konvergen secara stabil, dan komparasi optimizer disajikan secara objektif. |
| **5. Code Quality, Reproducibility & Documentation** | 10% | Kode terstruktur rapi berorientasi objek atau fungsi modular bersih, deterministik (`seed=42`), tanpa dependensi deep learning eksternal, dan dilengkapi dokumentasi penjelas yang komprehensif. |

---

## 10. Referensi Ilmiah Terverifikasi

1. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986)**. *Learning representations by back-propagating errors*. Nature, 323(6088), 533-536.
2. **Kingma, D. P., & Ba, J. (2014)**. *Adam: A Method for Stochastic Optimization*. arXiv preprint arXiv:1412.6980.
3. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)**. *Deep Learning* (Chapter 6: Deep Feedforward Networks & Chapter 8: Optimization for Training Deep Models). MIT Press.
4. **He, K., Zhang, X., Ren, S., & Sun, J. (2015)**. *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 1026-1034.
5. **Glorot, X., & Bengio, Y. (2010)**. *Understanding the difficulty of training deep feedforward neural networks*. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS), JMLR W&CP 9:249-256.
