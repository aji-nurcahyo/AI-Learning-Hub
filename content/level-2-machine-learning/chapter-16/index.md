# Chapter 16: Support Vector Machines & k-Nearest Neighbors

**Artificial Intelligence Learning Hub** — Level 2: Machine Learning  
*Created & Developed by Aji Pamungkas Tri Nurcahyo (Institut Teknologi Sumatera)*

---

## Deskripsi Modul & Tujuan Pembelajaran

Modul ini menyajikan eksplorasi mendalam mengenai dua paradigma pembelajaran terawasi (*Supervised Learning*) berbasis geometri dan ruang metrik: **Support Vector Machines (SVM)** dan **k-Nearest Neighbors (KNN)**. Pembahasan mencakup prinsip optimasi batas pemisah margin maksimum (*Maximum-Margin Hyperplane*), formulasi *Hard-Margin* dan *Soft-Margin* menggunakan variabel kelonggaran (*slack variables* $\xi_i$), parameter penalti regularisasi $C$, serta trik kernel (*Kernel Trick*) untuk memetakan data non-linier ke ruang berdimensi tinggi (Linear, Polinomial, dan Radial Basis Function / Gaussian RBF). Selanjutnya, modul mengkaji algoritma berbasis memori (*Instance-Based Lazy Learning*) KNN, formulasi metrik jarak ($L_1, L_2, L_\infty, L_p$), skema voting mayoritas versus pembobotan jarak, bahaya kutukan dimensi (*Curse of Dimensionality*), dan pembuktian empiris kebutuhan mutlak penskalaan fitur (*Feature Scaling*). Seluruh materi dilengkapi dengan penurunan matematis, diagram alur ASCII, verifikasi kode Scikit-learn, latihan 3 tingkat, dan mini project.

### Prasyarat Pembelajaran
Sebelum mempelajari bab ini, mahasiswa diwajibkan telah menguasai:
1. **Chapter 06**: Aljabar Linier (Vektor, Perkalian Titik / *Dot Product*, Norma Vektor $L_1$ dan $L_2$, Ortogonalitas, Hiperbidang) dan Geometri Analitik.
2. **Chapter 12**: Alur Kerja Machine Learning, Prapemrosesan Data, dan Penskalaan Fitur Bebas Kebocoran.
3. **Chapter 14**: Fondasi Algoritma Klasifikasi, Batas Keputusan (*Decision Boundary*), dan Metrik Evaluasi (Akurasi, Presisi, Recall, F1-Score, Confusion Matrix).
4. **Chapter 15**: Model Pohon Keputusan dan Ansambel (*Decision Trees & Ensemble Learning*).

### Capaian Pembelajaran Kognitif (Bloom's Taxonomy)
Setelah menyelesaikan bab ini, mahasiswa diharapkan mampu:
1. **Menjelaskan** konsep Support Vector Machines dan prinsip *Maximum-Margin Classification*.
2. **Mengidentifikasi** hiperbidang pemisah, margin geometris, dan sampel vektor pendukung (*support vectors*).
3. **Membedakan** formulasi Hard-Margin dan Soft-Margin SVM pada data yang tidak terpisah secara linier.
4. **Menjelaskan** peran parameter regularisasi $C$ dalam mengendalikan trade-off bias-varians dan penalti slack.
5. **Memahami** mekanisme aljabar Kernel Trick tanpa transformasi koordinat berdimensi tinggi eksplisit.
6. **Menggunakan** Linear, Polynomial, dan Gaussian Radial Basis Function (RBF) Kernels secara tepat.
7. **Menjelaskan** konsep k-Nearest Neighbors sebagai algoritma non-parametrik berbasis memori (*lazy learning*).
8. **Menghitung** jarak antar-sampel menggunakan metrik jarak Euclidean ($L_2$) dan Manhattan ($L_1$).
9. **Menganalisis** pengaruh pemilihan parameter $k$ dan metode agregasi suara (*majority vs distance-weighted voting*).
10. **Menjelaskan** kebutuhan mutlak penskalaan fitur (*feature scaling*) serta fenomena kutukan dimensi (*curse of dimensionality*).
11. **Membandingkan** SVM dan KNN berdasarkan akurasi, kapasitas generalisasi, latensi inferensi, dan konsumsi memori.
12. **Mengimplementasikan** dan mengevaluasi kedua model secara komprehensif menggunakan Scikit-learn Pipeline.

---

## Part A: Fondasi Support Vector Machines (SVM)

### 1. Prinsip Geometris Hiperbidang Pemisah Margin Maksimum
Diberikan himpunan data latih terawasi biner $(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_N, y_N)$ di mana $\mathbf{x}_i \in \mathbb{R}^D$ dan $y_i \in \{-1, +1\}$. Hiperbidang pemisah linier didefinisikan oleh persamaan vektor:
$$\mathbf{w}^T \mathbf{x} + b = 0$$
di mana $\mathbf{w} \in \mathbb{R}^D$ adalah vektor normal yang tegak lurus terhadap hiperbidang dan $b \in \mathbb{R}$ adalah bias skalar.

```text
+-------------------------------------------------------------------------+
|                  GEOMETRI HIPERBIDANG & MARGIN SVM                      |
+-------------------------------------------------------------------------+
                      w^T x + b = +1  (Batas Margin Positif)
                       /
                      /     [+] Support Vector Positif
                     /
                    /       [+]
                   /  <----------- Koridor Margin (Lebar = 2 / ||w||)
                  /
                 /  w^T x + b = 0  (Hiperbidang Pemisah Utama)
                /
               /     [-] Support Vector Negatif
              /
             /       [-]
            /
           /  w^T x + b = -1 (Batas Margin Negatif)
```

### 2. Formulasi Margin Geometris & Derivasi Lebar Margin
Jarak tegak lurus dari sembarang titik $\mathbf{x}$ ke hiperbidang $\mathbf{w}^T \mathbf{x} + b = 0$ dirumuskan sebagai:
$$\gamma = \frac{|\mathbf{w}^T \mathbf{x} + b|}{\|\mathbf{w}\|_2}$$
Untuk memastikan klasifikasi yang benar dengan skala kanonik $y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1$, jarak titik terdekat dari masing-masing kelas ke hiperbidang adalah $\frac{1}{\|\mathbf{w}\|_2}$. Dengan demikian, **lebar margin total** antara kelas positif dan negatif adalah:
$$\text{Margin Total} = \frac{1}{\|\mathbf{w}\|_2} + \frac{1}{\|\mathbf{w}\|_2} = \frac{2}{\|\mathbf{w}\|_2}$$
Memaksimalkan margin $\frac{2}{\|\mathbf{w}\|_2}$ setara secara matematis dengan **meminimalkan norma kuadrat bobot** $\frac{1}{2}\|\mathbf{w}\|_2^2$.

### 3. Hard-Margin SVM (Data Terpisah Linier Murni)
Pada kondisi data terpisah linier sempurna tanpa derau (*noise*), masalah optimasi konveks kuadratik Hard-Margin dirumuskan sebagai:
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{dengan kendala} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad \forall i = 1, \dots, N$$
**Keterbatasan Fatal Hard-Margin**: Jika terdapat satu saja sampel kelas $+1$ yang menyusup ke wilayah kelas $-1$ (misalnya akibat kesalahan pembacaan sensor atau outlier), masalah optimasi di atas menjadi *infeasible* (tidak memiliki solusi matematis).

---

## Part B: Soft-Margin SVM & Parameter Regularisasi C

### 1. Variabel Kelonggaran (*Slack Variables* $\xi_i$)
Untuk mengatasi data dunia nyata yang saling tumpang tindih (*linearly non-separable*), Cortes & Vapnik (1995) memperkenalkan variabel kelonggaran $\xi_i \ge 0$ untuk setiap sampel ke-$i$:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

```text
+-------------------------------------------------------------------------+
|                KLASIFIKASI KONDISI SAMPEL BERBASIS SLACK                |
+-------------------------------------------------------------------------+
  xi_i = 0         : Sampel berada tepat pada atau di luar batas margin yang benar.
  0 < xi_i <= 1    : Sampel berada di dalam koridor margin, namun sisi kelasnya BENAR.
  xi_i > 1         : Sampel melanggar margin dan SALAH SISI (Salah Klasifikasi).
```

### 2. Fungsi Objektif Soft-Margin
Fungsi tujuan meminimalkan kombinasi norma bobot (untuk memperlebar margin) dan jumlah total penalti pelanggaran:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{dengan kendala} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
di mana $C > 0$ adalah hiperparameter penalti regularisasi:
- **Nilai $C$ Kecil ($C \to 0$)**: Penalti pelanggaran longgar. Model memprioritaskan koridor margin yang selebar mungkin dengan mentoleransi banyak pelanggaran slack. Model memiliki bias lebih tinggi, varians rendah (*tahan outlier*).
- **Nilai $C$ Besar ($C \to \infty$)**: Penalti pelanggaran sangat berat. Model berusaha sekuat tenaga mengklasifikasikan setiap sampel latihan dengan benar, menghasilkan margin sempit. Model memiliki bias rendah, varians sangat tinggi (*rentan overfitting*).

---

## Part C: Metode Kernel (*Kernel Methods*)

### 1. Prinsip Pemetaan Non-Linier & Teorema Cover
Jika data tidak dapat dipisahkan secara linier di ruang fitur asli $\mathbb{R}^D$, data dipetakan ke ruang fitur berdimensi lebih tinggi $\mathcal{H}$ menggunakan transformasi non-linier $\phi(\mathbf{x}): \mathbb{R}^D \to \mathcal{H}$. Berdasarkan Teorema Cover (1965), data yang tidak terpisah linier di ruang dimensi rendah memiliki probabilitas sangat tinggi untuk terpisah linier setelah diproyeksikan ke ruang berdimensi tinggi yang memadai.

### 2. Formulasi Dual Lagrange & Trik Kernel (*Kernel Trick*)
Dalam formulasi dual Lagrange, vektor bobot $\mathbf{w}$ dinyatakan sebagai kombinasi linier dari data latih:
$$\mathbf{w} = \sum_{i=1}^{N} \alpha_i y_i \phi(\mathbf{x}_i)$$
Fungsi keputusan untuk sampel baru $\mathbf{x}$ hanya bergantung pada hasil perkalian titik (*inner product*) di ruang fitur tinggi:
$$f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b = \sum_{i=1}^{N} \alpha_i y_i \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}) \rangle + b$$
**Trik Kernel**: Kita tidak perlu menghitung koordinat transformasi eksplisit $\phi(\mathbf{x})$ yang sangat mahal atau tak berhingga! Cukup definisikan **fungsi kernel** $K(\mathbf{x}_i, \mathbf{x}_j)$ yang menghitung perkalian titik tersebut secara langsung menggunakan koordinat ruang asal:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$$

```text
+-------------------------------------------------------------------------+
|                  ALUR KERJA TRIK KERNEL (KERNEL TRICK)                  |
+-------------------------------------------------------------------------+
  Ruang Asli 2D (x1, x2)             Ruang Hilbert Tinggi (z1, z2, z3)
  [Titik Tidak Terpisah]             [Titik Terpisah oleh Bidang Datar]
         (X)                                        /
      (O)   (O)                                    /   (O) (O)
         (X)                                      / 
                                             (X) / (X)
             \                                  /
              \--- Transformasi Implisit via Kernel K(x, x') ---/
```

### 3. Tiga Fungsi Kernel Standar
1. **Linear Kernel**:
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$$
   Digunakan ketika jumlah fitur sudah sangat besar ($D \gg N$), misalnya klasifikasi teks atau bioinformatika.
2. **Polynomial Kernel**:
   $$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$$
   Memodelkan interaksi kombinatorial fitur hingga derajat $d$.
3. **Radial Basis Function (RBF / Gaussian) Kernel**:
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2 \right), \quad \gamma = \frac{1}{2\sigma^2} > 0$$
   Kernel universal yang secara implisit memetakan data ke ruang Hilbert berdimensi tak terhingga. Parameter $\gamma$ mengontrol kelengkungan dan radius pengaruh lonceng Gaussian.

---

## Part D: Fondasi k-Nearest Neighbors (KNN)

### 1. Paradigma Instance-Based Learning (*Lazy Learning*)
k-Nearest Neighbors (Cover & Hart, 1967) adalah algoritma pembelajaran terawasi non-parametrik berbasis instans (*instance-based*). Berbeda dengan SVM yang merupakan model *eager* (menghabiskan waktu komputasi saat pelatihan untuk mencari persamaan hiperbidang global), KNN adalah model *lazy*:
- **Fase Pelatihan ($O(1)$)**: Hanya menyimpan data latih terstandarisasi ke dalam memori tanpa pembentukan model eksplisit.
- **Fase Prediksi ($O(N \cdot D)$)**: Menghitung jarak dari sampel uji baru ke seluruh $N$ data latih, memilih $k$ sampel dengan jarak terpendek, dan mengagregasi labelnya.

### 2. Aturan Agregasi Suara (*Voting Schemes*)
1. **Uniform Majority Voting**: Setiap tetangga dari $k$ titik terdekat memiliki bobot suara setara ($1$ suara):
   $$\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbb{I}(y_i = c)$$
2. **Distance-Weighted Voting**: Tetangga yang posisinya sangat dekat dengan titik uji diberikan bobot suara lebih besar:
   $$w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon} \implies \hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \cdot \mathbb{I}(y_i = c)$$

---

## Part E: Metrik Jarak, Penskalaan Fitur & Kutukan Dimensi

### 1. Formulasi Matematis Metrik Jarak
Untuk dua titik data $\mathbf{p}, \mathbf{q} \in \mathbb{R}^D$:
- **Jarak Euclidean ($L_2$ Norm)**:
  $$d_{\text{euc}}(\mathbf{p}, \mathbf{q}) = \|\mathbf{p} - \mathbf{q}\|_2 = \sqrt{\sum_{j=1}^{D} (p_j - q_j)^2}$$
- **Jarak Manhattan ($L_1$ Norm / City Block)**:
  $$d_{\text{man}}(\mathbf{p}, \mathbf{q}) = \|\mathbf{p} - \mathbf{q}\|_1 = \sum_{j=1}^{D} |p_j - q_j|$$
- **Jarak Chebyshev ($L_\infty$ Norm)**:
  $$d_{\text{cheb}}(\mathbf{p}, \mathbf{q}) = \max_{j=1,\dots,D} |p_j - q_j|$$

### 2. Pembuktian Kebutuhan Mutlak Feature Scaling
Misalkan terdapat fitur Latensi Jaringan ($X_1 \in [10, 400]$ ms) dan Tegangan Baterai ($X_2 \in [2.1, 3.8]$ V). Dalam rumus jarak Euclidean:
$$d^2 = (X_{1a} - X_{1b})^2 + (X_{2a} - X_{2b})^2$$
Perbedaan latensi sebesar $100$ ms menghasilkan kontribusi kuadrat $100^2 = 10.000$. Sedangkan perbedaan tegangan maksimum $1.5$ V hanya menghasilkan $1.5^2 = 2.25$. Kontribusi latensi mendominasi tegangan sebesar $\frac{10.000}{2.25} \approx 4.444$ kali lipat! Tanpa `StandardScaler()`, model memperlakukan tegangan baterai sebagai derau yang diabaikan.

```text
+-------------------------------------------------------------------------+
|                  DISTORSI JARAK AKIBAT KETIADAAN SCALING                |
+-------------------------------------------------------------------------+
  Ruang Mentah (Tanpa Scaling)               Ruang Terstandarisasi (Z-Score)
  Latensi (ms)                               Z_Latensi
     ^                                          ^
 400 |       * (Jarak Sangat Jauh)           +2 |       *
     |                                          |     /   \
 200 |   *                                    0 |    *     *  (Jarak Simetris Adil)
     |                                          |     \   /
   0 +---+---+---> Tegangan (V)              -2 +---+---+---> Z_Tegangan
     2.0 3.0 4.0                               -2   0  +2
```

### 3. Kutukan Dimensi (*Curse of Dimensionality*)
Ketika jumlah dimensi $D \to \infty$, volume ruang tumbuh secara eksponensial sehingga kepadatan data mendekati nol. Akibatnya, rasio selisih jarak ke tetangga terjauh terhadap tetangga terdekat mendekati nol:
$$\lim_{D \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$$
Semua titik menjadi sama-sama jauh, menyebabkan konsep 'tetangga terdekat' pada KNN kehilangan makna pembeda statistik.

---

## Part F: Matriks Komparasi Komprehensif SVM vs KNN

| Parameter Karakteristik | Support Vector Machines (SVM) | k-Nearest Neighbors (KNN) |
|---|---|---|
| **Prinsip Dasar** | Optimasi Margin Maksimum Global (*Model-Based*) | Pengelompokan Kedekatan Lokal (*Instance-Based*) |
| **Batas Keputusan** | Hiperbidang Linier atau Kontur Kernel Halus | Batas Non-Linier Mosaik Voronoi Lokal |
| **Kebutuhan Scaling** | **MUTLAK (StandardScaler)** | **MUTLAK (StandardScaler)** |
| **Kompleksitas Latih (*Fit*)** | Tinggi: $O(N^2 D)$ hingga $O(N^3 D)$ | **Nol / Instan: $O(1)$** |
| **Kompleksitas Prediksi** | **Sangat Cepat: $O(D \cdot N_{\text{SV}})$** | Lambat: $O(N \cdot D)$ untuk setiap query uji |
| **Konsumsi Memori** | Hemat: Hanya menyimpan Support Vectors | Boros: Wajib menyimpan seluruh dataset latih |
| **Ketahanan Outlier** | Tinggi (dikontrol parameter $C$) | Rendah pada $k=1$, membaik pada $k$ sedang |
| **Ketahanan Dimensi Tinggi**| Sangat Baik (terlindungi oleh margin) | Sangat Buruk (*Curse of Dimensionality*) |
| **Hasil Benchmark WSN** | **Akurasi Uji: 70.00% (RBF Kernel)** | Akurasi Uji: 62.22% (Weighted $k=5$) |

---

## Part G: Latihan Soal Terstruktur & Pembahasan

### Soal 1 (Tingkat Dasar): Perhitungan Jarak Euclidean & Manhattan
Diberikan simpul sensor baru $X_{\text{baru}} = [-65.0 \text{ dBm}, 45.0^\circ\text{C}]$ dan dua data historis:
- Simpul $A$ (NORMAL): $[-60.0 \text{ dBm}, 35.0^\circ\text{C}]$
- Simpul $B$ (CRITICAL): $[-80.0 \text{ dBm}, 55.0^\circ\text{C}]$
Hitung jarak Euclidean dan Manhattan dari $X_{\text{baru}}$ ke simpul $A$ dan $B$. Berdasarkan 1-Nearest Neighbor ($k=1$), tentukan status simpul baru tersebut!

**Kunci Jawaban Soal 1**:
1. Selisih koordinat:
   - Ke $A$: $\Delta \text{Sinyal} = -65.0 - (-60.0) = -5.0$, $\Delta \text{Suhu} = 45.0 - 35.0 = 10.0$.
   - Ke $B$: $\Delta \text{Sinyal} = -65.0 - (-80.0) = +15.0$, $\Delta \text{Suhu} = 45.0 - 55.0 = -10.0$.
2. Jarak Euclidean ($L_2$):
   $$d_{\text{euc}}(X, A) = \sqrt{(-5.0)^2 + 10.0^2} = \sqrt{25 + 100} = \sqrt{125} \approx 11.18$$
   $$d_{\text{euc}}(X, B) = \sqrt{15.0^2 + (-10.0)^2} = \sqrt{225 + 100} = \sqrt{325} \approx 18.03$$
3. Jarak Manhattan ($L_1$):
   $$d_{\text{man}}(X, A) = |-5.0| + |10.0| = 5.0 + 10.0 = 15.0$$
   $$d_{\text{man}}(X, B) = |15.0| + |-10.0| = 15.0 + 10.0 = 25.0$$
4. Kesimpulan: Karena $d(X, A) < d(X, B)$ pada kedua metrik jarak, maka pada $k=1$ tetangga terdekat adalah simpul $A$. **Prediksi Status: NORMAL**.

### Soal 2 (Tingkat Menengah): Menghitung Lebar Margin dan Menentukan Support Vectors
Sebuah model Linear SVM biner memiliki vektor bobot $\mathbf{w} = [0.6, -0.8]^T$ dan bias $b = -0.5$.
1. Hitung norma bobot $\|\mathbf{w}\|_2$ dan lebar margin total $\frac{2}{\|\mathbf{w}\|_2}$!
2. Jika terdapat sampel data latih $\mathbf{x}_1 = [2.0, 1.0]^T$ dengan label $y_1 = +1$, tentukan apakah $\mathbf{x}_1$ merupakan Support Vector, sampel di luar margin, atau sampel yang melanggar margin!

**Kunci Jawaban Soal 2**:
1. Norma bobot:
   $$\|\mathbf{w}\|_2 = \sqrt{0.6^2 + (-0.8)^2} = \sqrt{0.36 + 0.64} = \sqrt{1.0} = 1.0$$
   Lebar margin total:
   $$\text{Margin} = \frac{2}{\|\mathbf{w}\|_2} = \frac{2}{1.0} = 2.0$$
2. Evaluasi fungsi margin untuk $\mathbf{x}_1$:
   $$\mathbf{w}^T \mathbf{x}_1 + b = (0.6 \times 2.0) + (-0.8 \times 1.0) - 0.5 = 1.2 - 0.8 - 0.5 = -0.1$$
   Nilai margin berskala:
   $$y_1 (\mathbf{w}^T \mathbf{x}_1 + b) = (+1) \times (-0.1) = -0.1$$
   Karena $y_1(\mathbf{w}^T \mathbf{x}_1 + b) < 1$, sampel $\mathbf{x}_1$ melanggar margin. Bahkan karena bernilai negatif ($-0.1 < 0$), sampel tersebut berada di sisi yang salah (salah klasifikasi) dengan variabel slack $\xi_1 = 1 - (-0.1) = 1.1$. Sampel ini adalah **Support Vector yang melanggar batas margin**.

### Soal 3 (Tingkat Mahir): Analisis Latensi Komputasi Inferensi Skala Besar
Sebuah sistem pemantauan gempa bumi berbasis IoT memproses $M = 100.000$ sinyal sensor per detik. Dataset historis yang tersedia memiliki $N = 50.000$ sampel dengan $D = 20$ fitur. Model Linear SVM menghasilkan $N_{\text{SV}} = 1.200$ support vectors.
Bandingkan jumlah operasi perkalian-penjumlahan (*floating-point operations*) yang dibutuhkan per detik antara Linear SVM versus k-Nearest Neighbors!

**Kunci Jawaban Soal 3**:
1. **Kompleksitas Linear SVM**:
   Linear SVM menyederhanakan seluruh support vector menjadi satu vektor bobot tunggal $\mathbf{w} \in \mathbb{R}^D$ saat fase pelatihan:
   $$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$
   Jumlah operasi per sampel = $D$ perkalian $+ (D-1)$ penjumlahan $+ 1$ bias $\approx 2D$ operasi.
   Untuk $M = 100.000$ sampel per detik:
   $$\text{Operasi}_{\text{SVM}} = 100.000 \times (2 \times 20) = 4.000.000 \text{ FLOP/s (4 MFLOP/s)}$$
   Dapat dijalankan dengan mudah pada prosesor mikrokontroler murah.
2. **Kompleksitas k-Nearest Neighbors (KNN)**:
   KNN wajib menghitung jarak Euclidean ke seluruh $N = 50.000$ titik data latih untuk setiap sampel uji:
   Perhitungan jarak ke 1 titik = $D$ selisih kuadrat $+ (D-1)$ penjumlahan $\approx 3D$ operasi.
   Total operasi per sampel uji = $50.000 \times (3 \times 20) = 3.000.000$ operasi.
   Untuk $M = 100.000$ sampel per detik:
   $$\text{Operasi}_{\text{KNN}} = 100.000 \times 3.000.000 = 300.000.000.000 \text{ FLOP/s (300 GFLOP/s)}$$
   Membutuhkan server GPU berdaya tinggi dan menimbulkan latensi jaringan yang tidak dapat diterima. Linear SVM unggul **$75.000$ kali lebih efisien** dalam fase inferensi waktu-nyata!

---

## Part H: Mini Project — Klasifikasi Status Jaringan Sensor Nirkabel

### 1. Deskripsi Masalah & Konteks Industri
Jaringan Sensor Nirkabel (*Wireless Sensor Networks* / WSN) pada perkebunan kelapa sawit cerdas beroperasi di lingkungan luar ruangan yang rentan terhadap cuaca ekstrem. Mahasiswa diminta membangun sistem diagnosis kegagalan simpul sensor secara otomatis ke dalam kategori `NORMAL`, `WARNING`, dan `CRITICAL` menggunakan telemetri 6 prediktor fisik (`signal_strength_dbm`, `temperature_c`, `humidity_pct`, `packet_loss_pct`, `latency_ms`, `sensor_voltage_v`).

### 2. Panduan Pengerjaan & Deliverables
1. **Inspeksi Data**: Muat 360 baris data dari `data/wireless_sensor_network_state.csv` dan periksa ketidakseimbangan kelas.
2. **Preprocessing Pipeline**: Bangun `Pipeline` Scikit-learn yang mengintegrasikan `StandardScaler()` untuk mencegah kebocoran data (*data leakage*).
3. **Eksperimen SVM**: Latih Linear SVM, Polynomial SVM, dan RBF SVM. Lakukan penyetelan hyperparameter $C \in [0.1, 10]$ dan $\gamma \in [0.01, 1]$.
4. **Eksperimen KNN**: Latih KNN dengan variasi $k \in [1, 25]$ dan bandingkan *Uniform Voting* vs *Distance-Weighted Voting*.
5. **Pembuktian Feature Scaling**: Tunjukkan tabel komparasi akurasi model sebelum dan sesudah penskalaan fitur.
6. **Laporan Komparatif**: Sajikan Confusion Matrix, F1-Score makro, analisis support vector, serta rekomendasi arsitektur inferensi edge.

### 3. Rubrik Penilaian Mini Project (Skala 100 Poin)

| Komponen Penilaian | Bobot | Kriteria Penuh (Full Credit) | Kriteria Sebagian (Partial Credit) | Kurang Bukti (Insufficient) |
|---|---|---|---|---|
| **Pemahaman Masalah & Data** | **15 Poin** | Menganalisis sebaran fitur, distribusi target multikelas, dan membagi train/test berstrata secara bersih. | Membagi dataset tanpa stratifikasi target. | Tidak memeriksa karakteristik dataset. |
| **Implementasi SVM & Kernel** | **25 Poin** | Mengimplementasikan Linear, Poly, dan RBF Kernel via Pipeline, inspeksi support vectors, dan analisis parameter C/Gamma. | Menerapkan hanya Linear SVM tanpa eksplorasi kernel. | Kode SVM gagal dijalankan atau bocor skala. |
| **Implementasi KNN & Geometri Jarak**| **25 Poin** | Menerapkan KNN dengan tuning nilai k, komparasi metrik jarak ($L_1$ vs $L_2$), dan pembuktian empiris dampak penskalaan fitur. | Menerapkan KNN standar tanpa tuning parameter k. | Tidak menyertakan implementasi KNN. |
| **Evaluasi Objektif & Analisis Galat**| **25 Poin** | Validasi silang 5-fold berstrata yang adil, tabel perbandingan akurasi/F1/waktu latensi, dan analisis confusion matrix. | Evaluasi hanya menggunakan metrik tunggal akurasi latih. | Tidak ada analisis evaluasi performa. |
| **Dokumentasi & Referensi Ilmiah** | **10 Poin** | Format laporan terstruktur rapi, kode modular (PEP 8), dan menyertakan minimal 5 referensi akademik sah. | Laporan minim penjelasan, analisis teknis kurang tajam. | Tidak ada dokumentasi atau referensi. |

---

## Part I: Referensi Akademik Primer

1. **Cortes, C., & Vapnik, V.** (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273–297.  
   DOI / URL: [https://doi.org/10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
2. **Cover, T., & Hart, P.** (1967). *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory, 13(1), 21–27.  
   DOI / URL: [https://doi.org/10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)
3. **Vapnik, V. N.** (1998). *Statistical Learning Theory*. John Wiley & Sons, Inc.  
   URL: [https://www.wiley.com/en-us/Statistical+Learning+Theory-p-9780471030034](https://www.wiley.com/en-us/Statistical+Learning+Theory-p-9780471030034)
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer New York.  
   URL: [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/)
5. **ACM / IEEE-CS / AAAI**. (2024). *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*.  
   URL: [https://csed.acm.org/](https://csed.acm.org/)

