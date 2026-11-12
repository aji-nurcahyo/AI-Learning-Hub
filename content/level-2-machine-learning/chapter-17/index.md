# Chapter 17: Unsupervised Learning & Clustering Algorithms

Selamat datang pada Chapter 17 dari kurikulum resmi kecerdasan buatan AI Learning Hub. Bab ini merupakan gerbang utama menuju paradigma *Unsupervised Learning* (Pembelajaran Tanpa Supervisi), yang berfokus pada algoritma pengelompokan data (*clustering*). Di dunia nyata, sebagian besar data yang terkumpul tidak memiliki anotasi label. Melalui bab ini, Anda akan mempelajari bagaimana agen cerdas mengungkap struktur laten, mempartisi ruang fitur berdimensi jamak, dan mengidentifikasi anomali tanpa bantuan pengawas eksternal.

---

## 1. Fondasi Pembelajaran Tanpa Supervisi (*Unsupervised Learning*)

### 1.1 Paradigma Supervised vs Unsupervised Learning
Dalam pembelajaran terawasi (*supervised learning*), algoritma diberikan pasangan masukan-keluaran $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ dengan tujuan mengestimasi fungsi pemetaan $f(\mathbf{x}) \approx y$. Sebaliknya, dalam pembelajaran tanpa supervisi (*unsupervised learning*), dataset hanya terdiri atas vektor fitur tanpa label:
$$\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N, \quad \mathbf{x}_i \in \mathbb{R}^D$$

Tujuan utama pembelajaran ini adalah menemukan distribusi probabilitas probabilitas dasar $p(\mathbf{x})$, mengidentifikasi subpopulasi alami, memetakan topologi kepadatan, atau merepresentasikan data dalam ruang berdimensi lebih rendah.

```
+-----------------------------------------------------------------------+
|                    PARADIGMA PEMBELAJARAN MESIN                       |
+-----------------------------------------------------------------------+
|  Supervised Learning:                                                 |
|  [Fitur X] + [Label Target y] ---> [Model Optimasi] ---> [Prediksi y] |
|                                                                       |
|  Unsupervised Learning (Clustering):                                  |
|  [Fitur X] (Tanpa Label)      ---> [Penemuan Pola]  ---> [Partisi C_k]|
+-----------------------------------------------------------------------+
```

### 1.2 Taksonomi Masalah Unsupervised Learning
Unsupervised learning mencakup beberapa cabang spesifik:
1. **Clustering (Pengelompokan)**: Mempartisi observasi ke dalam kelompok-kelompok homogen di mana keserupaan intra-kelompok tinggi dan keserupaan antar-kelompok rendah.
2. **Dimensionality Reduction (Reduksi Dimensi)**: Memproyeksikan fitur berdimensi tinggi ke ruang berdimensi rendah dengan mempertahankan varians atau struktur geometri lokal (dibahas tuntas pada Chapter 18).
3. **Density Estimation & Anomaly Detection**: Memodelkan kepadatan peluang untuk mendeteksi observasi langka atau pencilan (*outliers*).

---

## 2. Metrik Geometri & Kebutuhan Mutlak Penskalaan Fitur

### 2.1 Metrik Jarak dalam Ruang Fitur
Algoritma clustering berbasis partisi dan hierarki bergantung secara eksklusif pada perhitungan jarak antar-vektor sampel:

#### Jarak Euclidean ($L_2$ Norm)
Metrik jarak standar dalam ruang vektor berdimensi $D$:
$$d_2(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2 = \sqrt{\sum_{d=1}^D (x_{id} - x_{jd})^2}$$

#### Jarak Manhattan ($L_1$ Norm / City Block)
Jumlah selisih absolut proyeksi sumbu:
$$d_1(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_1 = \sum_{d=1}^D |x_{id} - x_{jd}|$$

#### Jarak Minkowski ($L_p$ Norm)
Bentuk umum metrik jarak parametrik dengan parameter $p \ge 1$:
$$d_p(\mathbf{x}_i, \mathbf{x}_j) = \left( \sum_{d=1}^D |x_{id} - x_{jd}|^p \right)^{\frac{1}{p}}$$

```
+-----------------------------------------------------------------------+
|                       GEOMETRI METRIK JARAK                           |
+-----------------------------------------------------------------------+
|           Euclidean (L2):                Manhattan (L1):              |
|               (x2, y2)                       (x2, y2)                 |
|                 /|                              |                     |
|                / |                          +---+                     |
|               /  |                          |                         |
|              /   |                          |                         |
|      (x1, y1)----+                  (x1, y1)+                         |
|         Garis Lurus Diagonal               Jalur Sumbu Kisi           |
+-----------------------------------------------------------------------+
```

### 2.2 Kebutuhan Mutlak Penskalaan Fitur (*Feature Scaling*)
Pertimbangkan dataset dengan dua variabel:
- Fitur $A$: Konsumsi Energi (rentang: $100$ hingga $500\text{ mAh}$).
- Fitur $B$: Tegangan Operasional (rentang: $3.0$ hingga $3.6\text{ V}$).

Jika dihitung jarak Euclidean mentah tanpa penskalaan:
$$(\Delta A)^2 \approx (400)^2 = 160.000, \quad (\Delta B)^2 \approx (0.6)^2 = 0.36$$

Perubahan tegangan sebesar $0.6\text{ V}$ (yang mungkin menandakan kegagalan baterai kritis) sama sekali tidak memiliki pengaruh numerik terhadap perhitungan jarak karena teredam oleh skala arus listrik. Oleh karena itu, standarisasi $Z$-score wajib dilakukan sebelum proses clustering:
$$z_{id} = \frac{x_{id} - \mu_d}{\sigma_d}$$
di mana $\mu_d$ adalah rata-rata fitur dan $\sigma_d$ adalah deviasi standar fitur ke-$d$.

---

## 3. Algoritma K-Means: Partisi Berbasis Sentroid

### 3.1 Formulasi Matematis & Fungsi Biaya WCSS
K-Means membagi $N$ observasi ke dalam sejumlah $K$ klaster yang telah ditentukan sebelumnya, dinotasikan sebagai $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$, dengan syarat:
$$\bigcup_{k=1}^K C_k = \mathcal{D}, \quad C_a \cap C_b = \emptyset \quad (\forall a \neq b)$$

Setiap klaster direpresentasikan oleh vektor pusat rata-rata (*centroid*) $\boldsymbol{\mu}_k$:
$$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$$

Fungsi objektif K-Means adalah meminimalkan *Within-Cluster Sum of Squares* (WCSS) yang juga dikenal sebagai inersia (*inertia*):
$$J(C, \boldsymbol{\mu}) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

```
+-----------------------------------------------------------------------+
|                    ALUR ITERASI ALGORITMA LLOYD                       |
+-----------------------------------------------------------------------+
|  [Mulai]                                                              |
|     │                                                                 |
|     ▼                                                                 |
|  [Inisialisasi K Sentroid Awal] (Random atau K-Means++)               |
|     │                                                                 |
|     ▼◄─────────────────────────────────────────────┐                  |
|  [Langkah Penugasan (Assignment)]:                 │                  |
|  Tugaskan tiap sampel ke sentroid terdekat:       │ (Belum Konvergen)|
|  c_i = argmin_k ||x_i - mu_k||^2                   │                  |
|     │                                              │                  |
|     ▼                                              │                  |
|  [Langkah Pembaruan (Update)]:                     │                  |
|  Hitung ulang posisi sentroid:                     │                  |
|  mu_k = (1 / |C_k|) * sum(x_i)                     │                  |
|     │                                              │                  |
|     ▼                                              │                  |
|  [Cek Konvergensi]: ──> Posisi sentroid berubah? ──┘                  |
|     │ Tidak                                                           |
|     ▼                                                                 |
|  [Selesai: Klaster Final Terbentuk]                                   |
+-----------------------------------------------------------------------+
```

### 3.2 Algoritma Lloyd
Algoritma Lloyd bekerja melalui optimasi koordinat bergantian:
1. **Langkah Penugasan (*Assignment Step*)**: Mempertahankan sentroid $\boldsymbol{\mu}_k$ konstan, lalu memperbarui label klaster untuk setiap sampel:
   $$c_i^{(t)} = \arg\min_{k \in \{1, \dots, K\}} \|\mathbf{x}_i - \boldsymbol{\mu}_k^{(t)}\|^2$$
2. **Langkah Pembaruan (*Update Step*)**: Mempertahankan penugasan klaster konstan, lalu menghitung ulang sentroid baru:
   $$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{i: c_i^{(t)} = k} \mathbf{x}_i$$

Iterasi dihentikan apabila perpindahan sentroid lebih kecil dari ambang batas toleransi ($\|\boldsymbol{\mu}_k^{(t+1)} - \boldsymbol{\mu}_k^{(t)}\| \le \epsilon$) atau jumlah maksimum iterasi telah tercapai.

### 3.3 Jebakan Minimum Lokal & Inisialisasi K-Means++
Fungsi biaya WCSS bersifat non-konveks dan memiliki banyak minimum lokal. Inisialisasi acak murni (*random initialization*) dapat menempatkan sentroid awal terlalu berdekatan, menghasilkan partisi sub-optimal.

David Arthur dan Sergei Vassilvitskii (2007) merumuskan algoritma inisialisasi **K-Means++** yang secara teoritis menjamin rasio aproksimasi $\mathcal{O}(\log K)$:
1. Pilih sentroid pertama $\boldsymbol{\mu}_1$ secara acak seragam dari himpunan data $\mathcal{D}$.
2. Untuk setiap sampel $\mathbf{x}_i$, hitung jarak terpendek ke sentroid terdekat yang telah terpilih:
   $$D(\mathbf{x}_i) = \min_{j \in \{1, \dots, m\}} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|$$
3. Pilih sentroid berikutnya $\boldsymbol{\mu}_{m+1}$ dari $\mathcal{D}$ dengan distribusi probabilitas proporsional terhadap kuadrat jarak:
   $$P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_{\mathbf{x}_j \in \mathcal{D}} D(\mathbf{x}_j)^2}$$
4. Ulangi langkah 2 dan 3 sampai tepat $K$ sentroid terpilih.

```
+-----------------------------------------------------------------------+
|                    PERBANDINGAN INISIALISASI                          |
+-----------------------------------------------------------------------+
|  Random Initialization:           K-Means++ Initialization:           |
|  *  *  (x)(x)  *                  *         *         *               |
|  *  *          *                  (x)       (x)       (x)             |
|  *             *                  *         *         *               |
|  (Dua sentroid menumpuk           (Sentroid tersebar merata           |
|   di satu kelompok data)           secara probabilistik proporsional) |
+-----------------------------------------------------------------------+
```

---

## 4. Menentukan Jumlah Klaster Optimal ($K$)

Karena clustering adalah pembelajaran tanpa supervisi, nilai $K$ tidak diberikan secara langsung. Diperlukan metode diagnostik objektif untuk memilih nilai $K$ yang merefleksikan struktur alami data.

### 4.1 Metode Siku (*The Elbow Method*)
Metode Siku memplot nilai inersia $J(K)$ terhadap berbagai nilai $K \in [1, K_{\max}]$:
- Ketika $K=1$, inersia bernilai maksimal (seluruh varians ditanggung satu sentroid).
- Ketika $K=N$, inersia bernilai tepat $0$ (setiap titik menjadi sentroid bagi dirinya sendiri).
- Nilai $K$ optimal diidentifikasi pada **titik siku (*elbow point*)**, yaitu titik di mana penambahan nilai $K$ berikutnya tidak lagi memberikan reduksi inersia yang signifikan secara marjinal.

```
Inersia J
  ^
  |  * (k=1)
  |   \
  |    \
  |     * (k=2)
  |      \
  |       * (k=3) <--- TITIK SIKU (ELBOW POINT)
  |        \________
  |                 *-------*-------* (k=4, 5, 6)
  +---------------------------------------------> Nilai k
```

### 4.2 Analisis Siluet (*Silhouette Analysis*)
Peter Rousseeuw (1987) memperkenalkan koefisien siluet (*silhouette coefficient*) yang mengukur seberapa dekat suatu titik dengan klaster tempat ia bernaung dibandingkan dengan klaster tetangga terdekat.

Untuk setiap sampel data ke-$i$:
1. **Kohesi Intra-Klaster $a(i)$**: Rata-rata jarak sampel $i$ ke semua titik lain dalam klaster yang sama $C_A$:
   $$a(i) = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \neq i} \|\mathbf{x}_i - \mathbf{x}_j\|$$
2. **Separasi Inter-Klaster $b(i)$**: Rata-rata jarak minimum sampel $i$ ke titik-titik pada klaster tetangga terdekat $C_B$:
   $$b(i) = \min_{C_B \neq C_A} \frac{1}{|C_B|} \sum_{j \in C_B} \|\mathbf{x}_i - \mathbf{x}_j\|$$
3. **Koefisien Siluet $s(i)$**:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad -1 \le s(i) \le +1$$

```
+-----------------------------------------------------------------------+
|                    INTERPRETASI NILAI SILUET s(i)                     |
+-----------------------------------------------------------------------+
|  s(i) mendekati +1.0 : Sampel sangat rapat dalam klasternya dan jauh  |
|                        dari klaster tetangga (Klaster Ideal).         |
|  s(i) mendekati  0.0 : Sampel berada tepat di perbatasan antara dua   |
|                        klaster (Ambiguitas Batas).                    |
|  s(i) mendekati -1.0 : Sampel lebih dekat ke klaster lain daripada   |
|                        ke klasternya sendiri (Salah Klaster).         |
+-----------------------------------------------------------------------+
```

---

## 5. Batasan & Kelemahan Fundamental K-Means

Meskipun K-Means sangat cepat secara komputasi ($\mathcal{O}(I \cdot K \cdot N \cdot D)$), algoritma ini mengandalkan asumsi geometris ketat:
1. **Asumsi Klaster Cembung/Sferis**: K-Means mengasumsikan klaster berbentuk bola hiperbola dengan jari-jari identik. K-Means dipastikan gagal jika data memiliki pola melengkung, spiral, atau cincin konsentris.
2. **Sensitif Terhadap Pencilan (*Outliers*)**: Karena sentroid dihitung berdasarkan rata-rata aritmatika kuadratik, satu titik pencilan ekstrem dapat menarik sentroid menjauh dari massa utama klaster.
3. **Tidak Mampu Menangani Kepadatan Variatif**: Klaster dengan varians kepadatan yang sangat kontras akan dipartisi secara tidak seimbang.

```
+-----------------------------------------------------------------------+
|                  KEGAGALAN GEOMETRIS K-MEANS                          |
+-----------------------------------------------------------------------+
|  Bentuk Non-Konveks (Dua Cincin):       K-Means Memaksa Partisi Garis:|
|          . . . . . . . .                      . . . | . . . . .       |
|        .                 .                  .       |         .       |
|       .      * * * *      .                .    * * | * *      .      |
|       .     *       *     .        ===>    .   *    |    *     .      |
|       .      * * * *      .                .    * * | * *      .      |
|        .                 .                  .       |         .       |
|          . . . . . . . .                      . . . | . . . . .       |
|   (Struktur Asli Cincin)               (Hasil K-Means Rusak Total)    |
+-----------------------------------------------------------------------+
```

---

## 6. Algoritma DBSCAN: Pengelompokan Berbasis Densitas

Untuk mengatasi kelemahan K-Means pada geometri non-konveks dan pencilan, Martin Ester dkk. (1996) merancang **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*).

### 6.1 Filosofi Dasar & Hyperparameter
DBSCAN mendefinisikan klaster sebagai wilayah berdensitas titik tinggi yang dipisahkan oleh wilayah berdensitas rendah. Algoritma ini dikendalikan oleh dua parameter:
1. **$\epsilon$ (*Epsilon*)**: Radius ketetanggaan spasial di sekitar sampel $\mathbf{x}$.
   $$N_\epsilon(\mathbf{x}) = \{\mathbf{x}' \in \mathcal{D} \mid \|\mathbf{x} - \mathbf{x}'\| \le \epsilon\}$$
2. **$\text{MinPts}$**: Jumlah observasi minimum yang harus berada dalam $N_\epsilon(\mathbf{x})$ agar wilayah tersebut dikategorikan padat.

### 6.2 Klasifikasi Titik: Core, Border, dan Noise
DBSCAN membagi seluruh titik dalam dataset menjadi 3 kategori:
- **Core Point (Titik Inti)**: Titik $\mathbf{x}$ di mana $|N_\epsilon(\mathbf{x})| \ge \text{MinPts}$.
- **Border Point (Titik Perbatasan)**: Titik $\mathbf{x}$ yang bukan *Core Point*, namun berada dalam lingkungan $\epsilon$ dari setidaknya satu *Core Point*.
- **Noise / Outlier (Titik Derau)**: Titik yang bukan *Core Point* dan bukan *Border Point* (diberi label $-1$).

```
+-----------------------------------------------------------------------+
|                    KLASIFIKASI TITIK PADA DBSCAN                      |
+-----------------------------------------------------------------------+
|                        ( Titik Inti / Core )                          |
|                             *     *                                   |
|                          *     o     *    |<- Radius Epsilon          |
|                             *     *                                   |
|                                                                       |
|       o (Core Point): Memiliki >= MinPts titik tetangga               |
|       + (Border Point): Tetangga < MinPts, namun dekat Core Point     |
|       x (Noise Point): Terisolasi di luar radius kepadatan            |
|                                                                       |
|         o --- o --- o --- +                     x                     |
|          \   /     /                         (Noise: -1)              |
|           \ /     /                                                   |
|            o --- o                                                    |
|     (Komponen Terkoneksi)                                             |
+-----------------------------------------------------------------------+
```

### 6.3 Mekanisme Perambatan Kepadatan
DBSCAN membangun klaster melalui penelusuran graf:
- **Directly Density-Reachable**: Titik $p$ dapat dijangkau langsung dari $q$ jika $q$ adalah *Core Point* dan $p \in N_\epsilon(q)$.
- **Density-Reachable**: Titik $p$ dapat dijangkau secara kepadatan dari $q$ jika terdapat rantai titik $p_1, p_2, \dots, p_n$ dengan $p_1 = q, p_n = p$ di mana setiap $p_{i+1}$ dapat dijangkau langsung dari $p_i$.
- **Density-Connected**: Titik $p$ dan $q$ saling terhubung jika terdapat suatu *Core Point* $o$ sedemikian rupa sehingga $p$ dan $q$ keduanya *density-reachable* dari $o$.

**Definisi Klaster DBSCAN**: Himpunan non-kosong $C \subseteq \mathcal{D}$ yang memenuhi sifat *Maximality* (semua titik yang *density-reachable* masuk ke dalam klaster) dan *Connectivity* (setiap pasang titik dalam klaster saling *density-connected*).

### 6.4 Penentuan Parameter Melalui Grafik Jarak-k (*k-Distance Graph*)
Untuk menentukan nilai $\epsilon$ secara ilmiah:
1. Tetapkan nilai $k = \text{MinPts}$ (heuristik umum: $\text{MinPts} = 2 \times D$).
2. Hitung jarak dari setiap sampel ke tetangga ke-$k$ terdekatnya menggunakan algoritma $k$-Nearest Neighbors.
3. Urutkan jarak-jarak tersebut dari terkecil hingga terbesar dan buat grafiknya.
4. Nilai $\epsilon$ optimal terletak tepat pada **lutut kurva (*knee of the curve*)**, yaitu titik di mana jarak melonjak drastis menandakan transisi dari wilayah padat ke wilayah derau.

---

## 7. Hierarchical Agglomerative Clustering & Analisis Dendrogram

*Hierarchical Clustering* membangun struktur pengelompokan bersarang (*nested clusters*) berbentuk pohon taksonomi. Terdapat dua paradigma:
1. **Agglomerative (Bottom-Up)**: Dimulai dengan $N$ klaster (setiap sampel membentuk klasternya sendiri), lalu menggabungkan pasangan klaster paling mirip secara rekursif hingga terbentuk satu klaster tunggal.
2. **Divisive (Top-Down)**: Dimulai dengan satu klaster induk raksasa, lalu membaginya secara rekursif ke dalam sub-klaster hingga setiap sampel terisolasi.

```
+-----------------------------------------------------------------------+
|                    HIERARCHICAL AGGLOMERATIVE                         |
+-----------------------------------------------------------------------+
|  Langkah 4: [============== Klaster Tunggal ==============]           |
|                                 ▲                                     |
|  Langkah 3:          [=== C1 ===]       [=== C2 ===]                  |
|                           ▲                  ▲                        |
|  Langkah 2:       [x1, x2]    x3          x4    x5                    |
|                      ▲                                                |
|  Langkah 1:         x1  x2    x3          x4    x5                    |
|                    (Masing-masing titik awalnya klaster sendiri)      |
+-----------------------------------------------------------------------+
```

### 7.1 Kriteria Keterkaitan (*Linkage Criteria*)
Jarak antara dua klaster majemuk $A$ dan $B$ dihitung menggunakan aturan keterkaitan:

#### Single Linkage (Jarak Terpendek)
$$d_{\text{single}}(A, B) = \min_{\mathbf{a} \in A, \mathbf{b} \in B} \|\mathbf{a} - \mathbf{b}\|$$
*Karakteristik*: Rentan terhadap fenomena perantaian (*chaining effect*) di mana dua klaster berbeda dihubungkan oleh jembatan derau tipis.

#### Complete Linkage (Jarak Terjauh)
$$d_{\text{complete}}(A, B) = \max_{\mathbf{a} \in A, \mathbf{b} \in B} \|\mathbf{a} - \mathbf{b}\|$$
*Karakteristik*: Menghasilkan klaster kompak dengan diameter seragam, namun sensitif terhadap pencilan.

#### Average Linkage (Jarak Rata-rata)
$$d_{\text{average}}(A, B) = \frac{1}{|A||B|} \sum_{\mathbf{a} \in A} \sum_{\mathbf{b} \in B} \|\mathbf{a} - \mathbf{b}\|$$
*Karakteristik*: Kompromi antara Single dan Complete Linkage; relatif stabil terhadap derau.

#### Ward's Linkage (Minimisasi Varians)
Metode Ward meminimalkan total varians intra-klaster. Kenaikan varians akibat penggabungan klaster $A$ dan $B$ dirumuskan sebagai:
$$\Delta \text{ESS}_{AB} = \frac{|A||B|}{|A| + |B|} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$$
*Karakteristik*: Pilihan default terbaik dalam data science karena menghasilkan partisi yang homogen dan sangat seimbang.

### 7.2 Struktur Dendrogram & Penentuan Ambang Batas Potong (*Cut Threshold*)
Dendrogram adalah diagram pohon yang memvisualisasikan seluruh riwayat penggabungan hirarkis. 
- Sumbu vertikal menunjukkan jarak disimilaritas penggabungan.
- Garis vertikal yang tinggi menunjukkan lompatan jarak yang jauh sebelum dua kelompok bersatu.
- Untuk memperoleh partisi klaster diskrit, garis pemotong horizontal (*horizontal cut line*) ditarik pada ketinggian tertentu. Jumlah garis vertikal yang terpotong merepresentasikan jumlah klaster yang dihasilkan.

```
Jarak Gabung (Distance)
  ^
18+                   +-----------------------+  (Cut Threshold: k=3)
  |                   |                       |
12+         +---------+---------+             |
  |         |                   |             |
 6+    +----+----+         +----+----+        |
  |    |         |         |         |        |
 0+---x1--------x2--------x3--------x4-------x5-----> Sampel Data
```

---

## 8. Metrik Evaluasi Kualitas Klastering

Karena tidak tersedianya label kelas kebenaran (*ground truth*), evaluasi clustering harus mengandalkan kriteria validasi internal yang mengukur kohesi internal dan pemisahan eksternal.

### 8.1 Metrik Evaluasi Internal (Tanpa Ground Truth)

#### 1. Silhouette Score
Rata-rata koefisien siluet dari seluruh sampel:
$$\bar{s} = \frac{1}{N} \sum_{i=1}^N s(i)$$
Rentang nilai $[-1, +1]$. Nilai di atas $0.5$ mengindikasikan struktur klaster yang solid.

#### 2. Davies-Bouldin Index (DBI)
Mengukur rasio keserupaan antara dispersi dalam klaster terhadap jarak antar sentroid:
$$R_{ij} = \frac{s_i + s_j}{\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2}, \quad \text{DBI} = \frac{1}{K} \sum_{i=1}^K \max_{j \neq i} R_{ij}$$
di mana $s_i$ adalah rata-rata jarak anggota klaster ke sentroidnya.
- **Kriteria**: Semakin **rendah** nilai DBI, semakin unggul kualitas partisi (klaster rapat dan saling terpisah jauh).

#### 3. Calinski-Harabasz Index (Variance Ratio Criterion)
Rata-rata rasio dispersi inter-klaster terhadap dispersi intra-klaster:
$$\text{CHI} = \frac{\text{Tr}(B_K) / (K - 1)}{\text{Tr}(W_K) / (N - K)}$$
di mana $B_K$ adalah matriks sebaran antar-klaster dan $W_K$ adalah matriks sebaran dalam klaster.
- **Kriteria**: Semakin **tinggi** nilai CHI, semakin tegas pemisahan antar-klaster.

### 8.2 Metrik Evaluasi Eksternal (Jika Tersedia Ground Truth Pengujian)

#### Adjusted Rand Index (ARI)
Mengukur kesesuaian pasangan elemen antara label prediksi klaster $\mathcal{C}$ dan label aktual $\mathcal{Y}$:
$$\text{ARI} = \frac{\text{Index} - \text{Expected Index}}{\text{Max Index} - \text{Expected Index}}$$
- Bernilai $+1.0$ jika partisi identik sempurna.
- Bernilai $\le 0.0$ jika partisi setara dengan penugasan acak.

---

## 9. Matriks Komparasi & Panduan Pemilihan Algoritma

| Dimensi Evaluasi | K-Means | DBSCAN | Agglomerative Hierarchical |
|---|---|---|---|
| **Paradigma** | Partisi sentroid | Kepadatan ruang (*density*) | Penggabungan bertingkat (*tree*) |
| **Bentuk Klaster** | Cembung / sferis | Arbitrary (bebas, non-linier) | Fleksibel tergantung linkage |
| **Penentuan Jumlah Klaster** | Wajib ditentukan ($K$) | Otomatis via $\epsilon$ & MinPts | Dipilih fleksibel saat pemotongan |
| **Kekebalan Outlier** | Rentan | Sangat tangguh (label noise) | Sedang (sensitif jika single/complete) |
| **Kompleksitas Waktu** | $\mathcal{O}(I \cdot K \cdot N \cdot D)$ | $\mathcal{O}(N \log N)$ s.d. $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ s.d. $\mathcal{O}(N^3)$ |
| **Skalabilitas Data ($N$)** | Sangat Tinggi ($N > 1.000.000$) | Menengah ($N < 100.000$) | Terbatas ($N \le 10.000$) |
| **Parameter Kunci** | $K$, init (`k-means++`) | $\epsilon$, $\text{MinPts}$ | `metric`, `linkage`, `n_clusters` |

---

## 10. Studi Kasus Komprehensif: Telemetri Jaringan Sensor Nirkabel

Sebagai demonstrasi praktis, kami menguji ketiga algoritma pada dataset telemetri jaringan sensor nirkabel berdimensi 6: `signal_strength_dbm`, `packet_loss_pct`, `latency_ms`, `temperature_c`, `voltage_v`, dan `energy_consumption_mah` (360 node).

```
+-------------------------------------------------------------------------------------------+
|               HASIL BENCHMARK EMPIRIS KLASTERING SENSOR NIRKABEL                          |
+-------------------------------------------------------------------------------------------+
| Algoritma            Klaster   Silhouette (Tinggi)  Davies-Bouldin (Rendah)  Noise Detected|
| ----------------------------------------------------------------------------------------- |
| K-Means (k=3)        3         0.5804               0.6419                   0             |
| DBSCAN (eps=1.15)    2         0.6538 (non-noise)   0.4907                   25 (6.94%)    |
| Agglomerative (Ward) 3         0.5739               0.6548                   0             |
+-------------------------------------------------------------------------------------------+
```

### Temuan Domain IoT:
1. **K-Means dan Ward Linkage** secara konsisten merekonstruksi 3 profil operasional utama:
   - Klaster 0: *Normal Operation* (daya sinyal tinggi, latensi rendah, voltase prima).
   - Klaster 1: *Moderate Warning* (latensi sedang, sedikit kehilangan paket data).
   - Klaster 2: *Degraded / High Energy* (konsumsi arus tinggi, degradasi tegangan).
2. **DBSCAN** memberikan nilai tambah diagnostik luar biasa dengan mengidentifikasi **25 node anomali terisolasi** sebagai noise. Node-node ini merepresentasikan anomali perangkat keras parah atau interferensi elektromagnetik ekstrem yang harus diinvestigasi oleh teknisi lapangan.

---

## 11. Latihan Bertingkat (Tiered Exercises)

### Tier 1: Pemahaman Konseptual
1. Mengapa inersia K-Means selalu bernilai nol ketika kita memilih $K = N$? Mengapa $K = N$ bukan solusi yang diinginkan?
   - *Jawaban*: Ketika $K = N$, setiap sampel data menjadi sentroid bagi dirinya sendiri, sehingga jarak sampel ke sentroidnya adalah nol ($\|\mathbf{x}_i - \boldsymbol{\mu}_i\| = 0$). Hal ini tidak memberikan kompresi atau wawasan pengelompokan sama sekali (*overfitting* ekstrem).
2. Sebutkan perbedaan status titik antara *Core Point*, *Border Point*, dan *Noise Point* pada algoritma DBSCAN!
   - *Jawaban*: Core Point memiliki tetangga $\ge \text{MinPts}$ dalam radius $\epsilon$. Border Point memiliki tetangga $< \text{MinPts}$ namun berada di dalam radius $\epsilon$ dari setidaknya satu Core Point. Noise Point bukan Core dan bukan Border.
3. Mengapa *Ward's Linkage* paling banyak digunakan pada data kontinu dibandingkan *Single Linkage*?
   - *Jawaban*: Ward meminimalkan penambahan total varians intra-klaster sehingga menghasilkan klaster yang kompak dan berukuran seimbang. Single Linkage rentan terhadap *chaining effect* (merangkaikan titik-titik derau menjadi jembatan semu).

### Tier 2: Perhitungan Matematis Manual
Diberikan 4 sampel satu dimensi $X = \{2, 4, 10, 12\}$.
Misalkan pada iterasi 0, dua sentroid awal dipilih: $\mu_1 = 2$ dan $\mu_2 = 4$.
1. **Langkah Penugasan**: Hitung penugasan klaster untuk setiap sampel.
   - Jarak ke $\mu_1=2$: $|2-2|=0, |4-2|=2, |10-2|=8, |12-2|=10$
   - Jarak ke $\mu_2=4$: $|2-4|=2, |4-4|=0, |10-4|=6, |12-4|=8$
   - Penugasan: $C_1 = \{2\}$, $C_2 = \{4, 10, 12\}$.
2. **Langkah Pembaruan**: Hitung sentroid baru untuk kedua klaster.
   - $\mu_1^{(1)} = 2$
   - $\mu_2^{(1)} = \frac{4 + 10 + 12}{3} = \frac{26}{3} \approx 8.67$
3. **Langkah Penugasan Kedua**:
   - Sampel 4: jarak ke $\mu_1(2)$ adalah 2; jarak ke $\mu_2(8.67)$ adalah 4.67 $\rightarrow$ Sampel 4 berpindah ke $C_1$!
   - Partisi baru: $C_1 = \{2, 4\} \Rightarrow \mu_1 = 3$; $C_2 = \{10, 12\} \Rightarrow \mu_2 = 11$. Konvergensi tercapai pada iterasi berikutnya.

### Tier 3: Pemecahan Masalah Sistem Komputasi
Sebuah perusahaan logistik memiliki 500.000 titik koordinat pengiriman harian. Tim data science ingin mengelompokkan titik-titik ini menjadi rute kurir efisien.
1. Apakah Agglomerative Hierarchical Clustering cocok untuk kasus ini? Jelaskan kendala kompleksitasnya!
   - *Jawaban*: Sangat tidak cocok. Agglomerative clustering membutuhkan memori $\mathcal{O}(N^2)$ untuk menyimpan matriks jarak dan waktu $\mathcal{O}(N^3)$ (atau $\mathcal{O}(N^2 \log N)$ dengan optimasi). Untuk $N=500.000$, matriks jarak membutuhkan lebih dari $1.000\text{ GB}$ RAM, menyebabkan *Out of Memory* (OOM).
2. Algoritma apa yang direkomendasikan?
   - *Jawaban*: Mini-Batch K-Means atau K-Means berbasis spatial indexing (misal $k$-d tree) yang memiliki kompleksitas linear $\mathcal{O}(KND)$, sangat terukur dan selesai dalam hitungan detik.

---

## 12. Mini Project Mandiri: Sistem Segmentasi Sensor IoT (Rubrik 100 Poin)

Mahasiswa diwajibkan mengimplementasikan pipeline unsupervised clustering terintegrasi pada dataset telemetri sensor baru.

### Rubrik Penilaian Proyek
- **Arsitektur Pipeline Prapemrosesan (25 Poin)**: Standarisasi data tanpa kebocoran informasi, eksplorasi korelasi multivariat.
- **Implementasi Multi-Algoritma (35 Poin)**: Implementasi K-Means, DBSCAN, dan Agglomerative Clustering secara terstruktur menggunakan kelas/fungsi modular.
- **Validasi & Triangulasi Metrik (25 Poin)**: Menghitung kurva Elbow, koefisien Silhouette, DBI, dan CHI, serta menyajikan analisis komparatif dalam tabel markdown rapi.
- **Interpretasi Domain Teknis (15 Poin)**: Memberikan wawasan fisis mengapa partisi yang terbentuk bermakna bagi keandalan sistem IoT.

---

## 13. Referensi Ilmiah Primer

1. **MacQueen, J. (1967)**. *Some methods for classification and analysis of multivariate observations*. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1(14), 281-297.
2. **Lloyd, S. (1982)**. *Least squares quantization in PCM*. IEEE Transactions on Information Theory, 28(2), 129-137.
3. **Arthur, D., & Vassilvitskii, S. (2007)**. *k-means++: The advantages of careful seeding*. Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms, 1027-1035.
4. **Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996)**. *A density-based algorithm for discovering clusters in large spatial databases with noise*. In KDD-96 Proceedings, 226-231.
5. **Rousseeuw, P. J. (1987)**. *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics, 20, 53-65.

