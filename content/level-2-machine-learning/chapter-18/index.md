# Chapter 18: Dimensionality Reduction

Selamat datang pada Chapter 18 dari kurikulum resmi kecerdasan buatan AI Learning Hub. Bab ini mendalami salah satu pilar terpenting dalam pemrosesan data modern: **Reduksi Dimensi (*Dimensionality Reduction*)**. Ketika model kecerdasan buatan dihadapkan pada dataset dengan ratusan hingga ribuan variabel, representasi data menjadi sangat renggang, rentan terhadap multikolinearitas, dan menuntut biaya komputasi yang sangat mahal. Melalui bab ini, Anda akan mempelajari prinsip aljabar linear dan geometri di balik reduksi dimensi linier (PCA, SVD, TruncatedSVD) serta teknik *manifold learning* non-linier (t-SNE dan UMAP) untuk kompresi representasi fitur dan visualisasi data.

---

## 1. Masalah Representasi & Kutukan Dimensi (*Curse of Dimensionality*)

### 1.1 Mengapa Reduksi Dimensi Dibutuhkan?
Dalam rekayasa sistem pembelajaran mesin, penambahan jumlah variabel sering kali tampak menarik secara intuitif. Namun pada praktiknya, fenomena yang dikenal sebagai **Fenomena Hughes (*Hughes Phenomenon*)** menunjukkan bahwa dengan jumlah sampel latih yang tetap terbatas, akurasi prediktif model pada awalnya meningkat seiring bertambahnya dimensi, tetapi kemudian akan merosot tajam ketika dimensi terus ditingkatkan melebihi batas optimal.

```
Akurasi Prediktif
  ^
  |          * * * (Puncak Optimal)
  |        *       *
  |       *         *  <--- Hughes Phenomenon (Kutukan Dimensi)
  |      *           *
  |     *             *
  |    *               * * * * * *
  +---------------------------------------------> Jumlah Dimensi (D)
```

Reduksi dimensi bertujuan mentransformasikan matriks data berdimensi tinggi $\mathbf{X} \in \mathbb{R}^{N \times D}$ menjadi matriks representasi berdimensi rendah $\mathbf{Z} \in \mathbb{R}^{N \times d}$ (di mana $d \ll D$) dengan mempertahankan informasi esensial sebanyak mungkin.

### 1.2 Geometri Ruang Berdimensi Tinggi: Volume Hiperkubus vs Hipersfer
Untuk memahami mengapa ruang berdimensi tinggi bersikap kontraintuitif, perhatikan perbandingan volume antara bola hipersferik berdimensi $D$ dengan jari-jari $r=0.5$ yang diletakkan tepat di dalam hiperkubus satuan berdimensi $D$ bersisi panjang $L=1.0$:

- Volume hiperkubus unit selalu konstan:
  $$V_{\text{kubus}}(D) = 1^D = 1.0$$
- Volume bola hipersferik berdimensi $D$ dirumuskan sebagai:
  $$V_{\text{sferik}}(D) = \frac{\pi^{D/2}}{\Gamma\left(\frac{D}{2} + 1\right)} r^D = \frac{\pi^{D/2}}{\Gamma\left(\frac{D}{2} + 1\right)} (0.5)^D$$

Ketika dimensi $D \to \infty$, volume bola hipersferik meluruh secara eksponensial menuju nol:
$$\lim_{D \to \infty} V_{\text{sferik}}(D) = 0$$

```
+-----------------------------------------------------------------------+
|              KONSENTRASI VOLUME PADA RUANG TINGGI                     |
+-----------------------------------------------------------------------+
|  Dimensi 2 (Persegi):            Dimensi D Tinggi (Hiperkubus):       |
|  +-----------------+             +-----------------+                  |
|  |     *****       |             | *             * | Hampir SELURUH   |
|  |   *********     |             |                 | massa volume     |
|  |  ***********    |             |                 | terkonsentrasi di|
|  |   *********     |             |                 | sudut-sudut      |
|  |     *****       |             | *             * | hiperkubus!      |
|  +-----------------+             +-----------------+                  |
|  (Lingkaran mengisi 78.5%)       (Hipersfer mendekati 0.000%)         |
+-----------------------------------------------------------------------+
```

Konsekuensi geometrisnya: dalam ruang berdimensi tinggi, hampir seluruh volume ruang terkonsentrasi di **sudut-sudut terluar (*corners*)**, meninggalkan bagian pusat ruang dalam kondisi hampir kosong melompong (*empty space phenomenon*).

### 1.3 Konsentrasi Jarak Geometris (*Distance Concentration*)
Dampak paling merusak dari kutukan dimensi adalah fenomena konsentrasi jarak. Dalam ruang berdimensi sangat tinggi, selisih antara jarak ke tetangga terdekat ($d_{\min}$) dan jarak ke titik terjauh ($d_{\max}$) menjadi tidak signifikan dibandingkan dengan jarak itu sendiri:
$$\lim_{D \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} = 0$$

```
+-----------------------------------------------------------------------------------------+
|                  BUKTI EMPIRIS KONSENTRASI JARAK METRIK EUCLIDEAN                       |
+-----------------------------------------------------------------------------------------+
| Dimensi (D)    Jarak Min (d_min)   Jarak Max (d_max)   Rasio d_min / d_max   Diskriminasi|
| --------------------------------------------------------------------------------------- |
| 2              0.0027              1.2916              0.0021                Sangat Tajam|
| 10             0.3946              2.1990              0.1795                Cukup Baik  |
| 100            3.0529              4.9099              0.6218                Mulai Kabur |
| 1000           11.9913             13.8110             0.8682                Hilang Tajam|
+-----------------------------------------------------------------------------------------+
```

Ketika rasio $d_{\min} / d_{\max}$ mendekati $1.0$, konsep ketetanggaan spasial kehilangan makna fisiknya. Algoritma berbasis jarak seperti KNN, K-Means, dan SVM mengalami degradasi performa drastis jika data tidak direduksi dimensinya terlebih dahulu.

---

## 2. Principal Component Analysis (PCA): Fondasi & Penurunan Aljabar

*Principal Component Analysis* (PCA), yang dipelopori oleh Karl Pearson (1901) dan dikembangkan lebih lanjut oleh Harold Hotelling (1933), merupakan teknik reduksi dimensi linier tanpa supervisi paling fundamental di dunia.

```
+-----------------------------------------------------------------------+
|                    ROTASI KOORDINAT PADA PCA                          |
+-----------------------------------------------------------------------+
|  Ruang Fitur Asli (X1, X2):          Ruang Komponen Utama (PC1, PC2): |
|           X2                                  PC2                     |
|           ^     * * *                         ^                       |
|           |   * * *                           |      * * *            |
|           | * * *   (Sumbu PC1)               |    * * * * *          |
|           +---------> X1                      +---------> PC1         |
|                                         (Varians Maksimal pada PC1,   |
|                                          Kovarians Antar-PC = 0)      |
+-----------------------------------------------------------------------+
```

### 2.1 Penurunan Matematis dari Perspektif Maksimisasi Varians
Misalkan kita memiliki matriks data terpusat $\mathbf{X} \in \mathbb{R}^{N \times D}$ di mana rata-rata tiap kolom adalah nol ($\frac{1}{N}\sum_{i=1}^N \mathbf{x}_i = \mathbf{0}$). Kita ingin mencari vektor arah satuan $\mathbf{w}_1 \in \mathbb{R}^D$ ($\|\mathbf{w}_1\|_2 = 1$) sedemikian rupa sehingga varians data hasil proyeksi $z_{i1} = \mathbf{x}_i^T \mathbf{w}_1$ bernilai maksimal.

Varians empiris dari proyeksi $\mathbf{z}_1 = \mathbf{X} \mathbf{w}_1$ adalah:
$$\sigma_{\mathbf{z}_1}^2 = \frac{1}{N - 1} \mathbf{z}_1^T \mathbf{z}_1 = \frac{1}{N - 1} (\mathbf{X} \mathbf{w}_1)^T (\mathbf{X} \mathbf{w}_1) = \mathbf{w}_1^T \left( \frac{1}{N - 1} \mathbf{X}^T \mathbf{X} \right) \mathbf{w}_1 = \mathbf{w}_1^T \mathbf{\Sigma} \mathbf{w}_1$$
di mana $\mathbf{\Sigma} \in \mathbb{R}^{D \times D}$ adalah matriks kovarians sampel.

Masalah optimasi bersyarat dapat dirumuskan sebagai:
$$\max_{\mathbf{w}_1} \mathbf{w}_1^T \mathbf{\Sigma} \mathbf{w}_1 \quad \text{dengan syarat} \quad \mathbf{w}_1^T \mathbf{w}_1 = 1$$

Menggunakan metode Pengali Lagrange (*Lagrange Multipliers*):
$$\mathcal{L}(\mathbf{w}_1, \lambda_1) = \mathbf{w}_1^T \mathbf{\Sigma} \mathbf{w}_1 - \lambda_1 (\mathbf{w}_1^T \mathbf{w}_1 - 1)$$

Menurunkan $\mathcal{L}$ terhadap $\mathbf{w}_1$ dan menyamakannya dengan nol:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_1} = 2 \mathbf{\Sigma} \mathbf{w}_1 - 2 \lambda_1 \mathbf{w}_1 = \mathbf{0} \implies \mathbf{\Sigma} \mathbf{w}_1 = \lambda_1 \mathbf{w}_1$$

Persamaan di atas merupakan **persamaan nilai eigen standar (*eigenvalue equation*)**. Vektor bobot komponen utama pertama $\mathbf{w}_1$ adalah **vektor eigen (*eigenvector*)** dari matriks kovarians $\mathbf{\Sigma}$, dan varians data yang terproyeksikan persis sama dengan **nilai eigen (*eigenvalue*)** yang bersesuaian:
$$\sigma_{\mathbf{z}_1}^2 = \mathbf{w}_1^T \mathbf{\Sigma} \mathbf{w}_1 = \mathbf{w}_1^T (\lambda_1 \mathbf{w}_1) = \lambda_1 (\mathbf{w}_1^T \mathbf{w}_1) = \lambda_1$$

Oleh karena itu, untuk memaksimalkan varians, kita harus memilih vektor eigen yang bersesuaian dengan **nilai eigen terbesar** $\lambda_1$. Sumbu komponen berikutnya ($\mathbf{w}_2, \mathbf{w}_3, \dots$) diperoleh dengan mencari vektor eigen ortogonal berikutnya yang berkorespondensi dengan nilai eigen terbesar kedua, ketiga, dan seterusnya.

### 2.2 Perspektif Ekuivalen: Minimisasi Galat Rekonstruksi
Secara geometris, memproyeksikan data ke subruang berdimensi $k$ yang memaksimalkan varians terbukti secara matematis **persis ekuivalen** dengan meminimalkan jumlah kuadrat galat jarak ortogonal antara titik data asli $\mathbf{x}_i$ dan titik hasil rekonstruksi $\hat{\mathbf{x}}_i$:
$$\min_{\mathbf{W}_k} \sum_{i=1}^N \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2 \iff \max_{\mathbf{W}_k} \text{Tr}(\mathbf{W}_k^T \mathbf{\Sigma} \mathbf{W}_k)$$

---

## 3. Komputasi PCA: Matriks Kovarians & Dekomposisi Spektral

### 3.1 Prosedur Algoritma PCA Lengkap
1. **Pemusatan Data (*Centering*)**: Kurangkan setiap observasi dengan vektor rata-rata kolomnya:
   $$\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i, \quad \mathbf{X}_c = \mathbf{X} - \mathbf{1} \bar{\mathbf{x}}^T$$
2. **Hitung Matriks Kovarians Sampel**:
   $$\mathbf{\Sigma} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c$$
3. **Dekomposisi Spektral Nilai Eigen**:
   $$\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$$
   di mana $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_D]$ adalah matriks ortogonal vektor eigen dan $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_D)$ dengan $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$.
4. **Bentuk Matriks Proyeksi Rank-$k$**: Ambil $k$ vektor eigen pertama:
   $$\mathbf{W}_k = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k] \in \mathbb{R}^{D \times k}$$
5. **Transformasi Proyeksi**:
   $$\mathbf{Z} = \mathbf{X}_c \mathbf{W}_k \in \mathbb{R}^{N \times k}$$

```
+-----------------------------------------------------------------------+
|                    ALUR KOMPUTASI MATRIKS PCA                         |
+-----------------------------------------------------------------------+
|  [Data Mentah X] (N x D)                                              |
|         │                                                             |
|         ▼                                                             |
|  [Pemusatan Data X_c = X - mu]                                        |
|         │                                                             |
|         ▼                                                             |
|  [Matriks Kovarians Sigma = (1/(N-1)) * X_c^T * X_c] (D x D)          |
|         │                                                             |
|         ▼                                                             |
|  [Dekomposisi Eigen: Sigma * v_j = lambda_j * v_j]                    |
|         │                                                             |
|         ▼                                                             |
|  [Urutkan Nilai Eigen Descending & Ambil k Vektor Teratas: W_k] (D x k)|
|         │                                                             |
|         ▼                                                             |
|  [Proyeksi Data Baru: Z = X_c * W_k] (N x k)                          |
+-----------------------------------------------------------------------+
```

### 3.2 Bobot Pemuatan (*Factor Loadings*)
Bobot pemuatan (*factor loading*) mengukur korelasi linier antara fitur asli ke-$j$ dan komponen utama ke-$m$:
$$L_{jm} = v_{jm} \sqrt{\lambda_m}$$
Bobot pemuatan memberikan interpretasi fisis mengenai variabel apa yang paling berkontribusi terhadap masing-masing komponen utama.

---

## 4. Evaluasi Scree Plot & Kriteria Pemilihan Jumlah Komponen ($k$)

### 4.1 Rasio Varians Terjelaskan (*Explained Variance Ratio*)
Proporsi varians total yang dirangkum oleh komponen utama ke-$j$ dirumuskan sebagai:
$$\text{EVR}_j = \frac{\lambda_j}{\sum_{d=1}^D \lambda_d}$$
Rasio kumulatif untuk $k$ komponen pertama adalah:
$$\text{Cumulative EVR}(k) = \sum_{j=1}^k \text{EVR}_j$$

```
Varians Kumulatif (%)
  100+                                         * (k=6: 100%)
     |                             * (k=4: 95.24%)
   90+                 * (k=3: 91.62%)
     |     * (k=2: 87.45%)
   80+  * (k=1: 80.51%)
     |
     +-----+-----+-----+-----+-----+-----+-----> Nomor Komponen (k)
           1     2     3     4     5     6
```

### 4.2 Kriteria Pemilihan Nilai $k$ Standar Industri
1. **Kriteria Siku (*Scree Plot Elbow Rule*)**: Cattell (1966) mengusulkan untuk mencari titik belokan tajam pada kurva nilai eigen. Titik setelah siku merepresentasikan varians sisa yang setara dengan derau acak.
2. **Ambang Batas Varians Kumulatif**: Memilih nilai $k$ terkecil sedemikian rupa sehingga $\text{Cumulative EVR}(k) \ge \alpha$, di mana $\alpha$ umumnya ditetapkan sebesar $90\%$ atau $95\%$.
3. **Kriteria Kaiser-Guttman**: Pada data yang telah distandarisasi ($Z$-score), setiap variabel asli memiliki varians $1.0$. Kriteria Kaiser menyatakan bahwa kita hanya boleh mempertahankan komponen yang memiliki nilai eigen $\lambda_j \ge 1.0$.

---

## 5. Rekonstruksi Data & Kompresi Lossy

PCA merupakan teknik kompresi data linier *lossy*. Untuk memetakan representasi berdimensi rendah $\mathbf{Z} \in \mathbb{R}^{N \times k}$ kembali ke ruang fitur asli $\mathbb{R}^D$:
$$\hat{\mathbf{X}} = \mathbf{Z} \mathbf{W}_k^T + \mathbf{1} \bar{\mathbf{x}}^T$$

Galat rekonstruksi total diukur menggunakan **Norma Frobenius**:
$$\|\mathbf{X} - \hat{\mathbf{X}}\|_F = \sqrt{\sum_{i=1}^N \sum_{d=1}^D (x_{id} - \hat{x}_{id})^2} = \sqrt{(N - 1) \sum_{j=k+1}^D \lambda_j}$$

Jika seluruh $D$ komponen dipertahankan ($k = D$), maka $\hat{\mathbf{X}} = \mathbf{X}$ dan galat rekonstruksi bernilai tepat nol (*lossless*).

---

## 6. Singular Value Decomposition (SVD) & Teorema Eckart-Young-Mirsky

### 6.1 Formulasi Umum SVD
Dekomposisi Nilai Singular (*Singular Value Decomposition*) adalah teorema fundamental aljabar linear yang menyatakan bahwa **setiap matriks riil** $\mathbf{X} \in \mathbb{R}^{N \times D}$ dapat didekomposisi menjadi tiga matriks:
$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma}_{\text{svd}} \mathbf{V}^T$$
- $\mathbf{U} \in \mathbb{R}^{N \times N}$ adalah matriks ortogonal vektor singular kiri ($\mathbf{U}^T \mathbf{U} = \mathbf{I}$).
- $\mathbf{\Sigma}_{\text{svd}} \in \mathbb{R}^{N \times D}$ adalah matriks diagonal nilai singular $s_1 \ge s_2 \ge \dots \ge s_{\min(N,D)} \ge 0$.
- $\mathbf{V} \in \mathbb{R}^{D \times D}$ adalah matriks ortogonal vektor singular kanan ($\mathbf{V}^T \mathbf{V} = \mathbf{I}$).

```
+-----------------------------------------------------------------------+
|                    DEKOMPOSISI SINGULAR VALUE (SVD)                   |
+-----------------------------------------------------------------------+
|    [   X   ]    =    [   U   ]    *    [ Sigma ]    *    [   V^T   ]  |
|     (N x D)           (N x N)           (N x D)           (D x D)     |
|                                                                       |
|  X = sum_{i=1}^r s_i * u_i * v_i^T  (Penjumlahan Matriks Rank-1)      |
+-----------------------------------------------------------------------+
```

### 6.2 Hubungan Ekuivalen SVD dan PCA
Jika data $\mathbf{X}$ telah dipusatkan ($\mathbf{X} = \mathbf{X}_c$):
$$\mathbf{X}^T \mathbf{X} = (\mathbf{U} \mathbf{\Sigma}_{\text{svd}} \mathbf{V}^T)^T (\mathbf{U} \mathbf{\Sigma}_{\text{svd}} \mathbf{V}^T) = \mathbf{V} \mathbf{\Sigma}_{\text{svd}}^T \mathbf{U}^T \mathbf{U} \mathbf{\Sigma}_{\text{svd}} \mathbf{V}^T = \mathbf{V} \mathbf{\Sigma}_{\text{svd}}^2 \mathbf{V}^T$$

Matriks kovarians sampel adalah:
$$\mathbf{\Sigma} = \frac{1}{N - 1} \mathbf{X}^T \mathbf{X} = \mathbf{V} \left( \frac{\mathbf{\Sigma}_{\text{svd}}^2}{N - 1} \right) \mathbf{V}^T$$

Hubungan langsung antara SVD dan PCA:
1. Matriks vektor singular kanan $\mathbf{V}$ **identik** dengan matriks vektor eigen kovarians $\mathbf{V}$.
2. Nilai eigen kovarians $\lambda_j$ bersesuaian persis dengan kuadrat nilai singular dibagi $(N - 1)$:
   $$\lambda_j = \frac{s_j^2}{N - 1}$$
3. Skor proyeksi PCA dapat dihitung langsung tanpa pembentukan matriks kovarians:
   $$\mathbf{Z} = \mathbf{X} \mathbf{V}_k = \mathbf{U}_k \mathbf{\Sigma}_k$$

*Catatan Implementasi*: Pustaka industri modern seperti Scikit-Learn menghitung PCA secara langsung melalui SVD karena algoritma SVD jauh lebih stabil secara numerik terhadap galat pembulatan floating point daripada membentuk matriks $\mathbf{X}^T \mathbf{X}$.

### 6.3 Teorema Eckart-Young-Mirsky
Carl Eckart dan Gale Young (1936) merumuskan teorema fundamental aproksimasi matriks rank rendah (*Low-Rank Matrix Approximation*):

**Pernyataan Teorema**: Diberikan matriks $\mathbf{X}$ dengan dekomposisi SVD $\mathbf{X} = \sum_{i=1}^r s_i \mathbf{u}_i \mathbf{v}_i^T$. Maka matriks rank-$k$ terbaik $\mathbf{X}_k$ yang meminimalkan galat norma Frobenius:
$$\mathbf{X}_k = \arg\min_{\text{rank}(\mathbf{A}) \le k} \|\mathbf{X} - \mathbf{A}\|_F$$
adalah penjumlahan terpotong $k$ suku pertama dekomposisi SVD:
$$\mathbf{X}_k = \sum_{i=1}^k s_i \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$
dan batas galat minimum yang dicapai adalah:
$$\|\mathbf{X} - \mathbf{X}_k\|_F = \sqrt{\sum_{j=k+1}^{\min(N,D)} s_j^2}$$

Teorema ini membuktikan bahwa tidak ada representasi rank-$k$ linier lain yang dapat mengungguli aproksimasi SVD/PCA dalam hal galat kuadratik terkecil.

---

## 7. TruncatedSVD untuk Matriks Jarang (*Sparse Matrices*)

Dalam pemrosesan bahasa alami (NLP, seperti matriks TF-IDF istilah-dokumen) atau analisis graf, matriks data sering kali berdimensi raksasa ($D > 50.000$) namun sangat jarang (*sparse*), di mana $99\%$ entri bernilai nol.

Jika kita memusatkan data dengan mengurangkan rata-rata ($\mathbf{x}_i - \bar{\mathbf{x}}$), seluruh entri nol akan berubah menjadi angka non-nol. Matriks jarang akan seketika berubah menjadi matriks padat (*dense*), menyebabkan ledakan konsumsi memori RAM (*Out-of-Memory error*).

**TruncatedSVD** menghitung dekomposisi nilai singular terpotong langsung pada matriks jarang tanpa melakukan pemusatan data. Jika data sengaja dipusatkan terlebih dahulu, proyeksi TruncatedSVD terbukti identik $100\%$ secara matematis dengan PCA standar.

---

## 8. Reduksi Dimensi dalam Pipeline Machine Learning Bebas Bocor (*Zero Data Leakage*)

Salah satu kesalahan paling fatal dalam rekayasa machine learning adalah melakukan penskalaan (`StandardScaler`) dan reduksi dimensi (`PCA`) pada seluruh dataset sebelum membagi data menjadi *train* dan *test set*.

```
+-----------------------------------------------------------------------+
|                    BAHAYA KEBOCORAN DATA (DATA LEAKAGE)               |
+-----------------------------------------------------------------------+
|  SALAH FATAL (Data Leakage):                                          |
|  [Seluruh Data (Train + Test)] ---> [Fit Scaler & PCA] ---> [Split]   |
|  * Rata-rata, varians, dan sumbu komponen data uji bocor ke model!    |
|                                                                       |
|  BENAR SESUAI STANDAR INDUSTRI (Zero Leakage):                        |
|  [Seluruh Data] ---> [Train-Test Split]                               |
|                            │                                          |
|         ┌──────────────────┴──────────────────┐                       |
|         ▼                                     ▼                       |
|   [X_train]                             [X_test]                      |
|         │                                     │                       |
|         ▼                                     ▼                       |
|   [.fit_transform()]                     [.transform() SAJA]          |
|   (Pelajari mu, sigma, W_k)              (Gunakan parameter train)    |
+-----------------------------------------------------------------------+
```

### Implementasi Pipeline Scikit-Learn
Penerapan disiplin pipeline menjamin integritas metodologi:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2, random_state=42)),
    ('classifier', LogisticRegression())
])

# Fit hanya pada data latih
pipe.fit(X_train, y_train)

# Transformasi dan prediksi otomatis pada data uji
y_pred = pipe.predict(X_test)
```

Pada dataset telemetri sensor kita, reduksi fitur dari 6 dimensi menjadi 2 komponen utama melalui pipeline justru meningkatkan akurasi klasifikasi data uji dari **$94.44\%$** menjadi **$96.67\%$** karena PCA secara efektif membuang derau acak pada komponen minor (PC3 s.d. PC6).

---

## 9. Manifold Learning Non-Linier: t-SNE & UMAP

### 9.1 Keterbatasan Fundamental Reduksi Linier
PCA hanya mampu menemukan subruang linier datar. Jika struktur data intrinsik melengkung secara non-linier seperti manifold kue lapis (*Swiss Roll*) atau kurva spiral, proyeksi linier PCA akan menumpuk lapisan-lapisan yang sebenarnya terpisah menjadi satu gumpalan campur aduk.

```
+-----------------------------------------------------------------------+
|                  KEGAGALAN PCA PADA MANIFOLD NON-LINIER               |
+-----------------------------------------------------------------------+
|  Manifold Melengkung 3D (Swiss Roll):    Proyeksi Linier PCA (2D):    |
|           @ @ @ @ @                      @ @ @ * * * @ @ @            |
|         @           *                    * * @ @ * * @ @ *            |
|        @    * * *    *     ===>          (Lapisan manifold menumpuk   |
|        @   *     *   *                    dan bertabrakan satu sama   |
|         @   * * *   *                     lain secara artifisial)     |
|          @         *                                                  |
|           @ @ @ @ *                                                   |
+-----------------------------------------------------------------------+
```

### 9.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)
Laurens van der Maaten dan Geoffrey Hinton (2008) mengembangkan **t-SNE**, algoritma non-linier yang berfokus menjaga struktur ketetanggaan probabilitas lokal.

#### Mekanisme Aljabar t-SNE:
1. **Ruang Dimensi Tinggi**: Probabilitas bersyarat bahwa $\mathbf{x}_j$ adalah tetangga $\mathbf{x}_i$ dimodelkan dengan distribusi Gaussian:
   $$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
2. **Ruang Proyeksi Rendah (2D/3D)**: Probabilitas ketetanggaan titik proyeksi $\mathbf{y}_i$ dan $\mathbf{y}_j$ dimodelkan dengan **distribusi t-Student dengan 1 derajat kebebasan (Cauchy distribution)**:
   $$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|\mathbf{y}_k - \|\mathbf{y}_l\|^2)^{-1}}$$
3. **Fungsi Biaya**: Meminimalkan Divergensi Kullback-Leibler antara distribusi $P$ dan $Q$:
   $$\text{KL}(P || Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

#### Mengapa Distribusi t-Student Menyelesaikan Crowding Problem?
Distribusi t-Student memiliki ekor yang jauh lebih tebal (*heavy tails*) daripada distribusi Gaussian. Pada jarak moderat, kurva t-Student meluruh lambat ($1/r^2$), sehingga klaster-klaster yang berbeda dapat saling menjauh secara leluasa pada ruang 2D tanpa menyebabkan penumpukan paksa (*crowding*) di area tengah plot.

#### Keterbatasan Kritis t-SNE:
- **Model Transduktif**: t-SNE tidak menghasilkan fungsi pemetaan matematis tetap. Tidak ada fungsi `.transform(X_new)` untuk sampel baru.
- **Jarak Global Palsu**: Jarak antar-klaster yang terpisah jauh pada visualisasi t-SNE tidak mencerminkan jarak sebenarnya di ruang dimensi tinggi.

### 9.3 UMAP (Uniform Manifold Approximation and Projection)
Leland McInnes dkk. (2018) merancang **UMAP**, algoritma manifold learning yang berakar pada teori topologi aljabar dan geometri Riemannian (*Fuzzy Simplicial Sets*).

Keunggulan utama UMAP dibandingkan t-SNE:
1. **Preservasi Struktur Ganda**: Mampu mempertahankan struktur lokal tetangga dekat sekaligus hubungan topologi global antar-kelompok besar.
2. **Efisiensi & Skalabilitas Komputasi**: Kompleksitas $\mathcal{O}(N^{1.14})$ memungkinkan UMAP memproses jutaan sampel data, jauh lebih cepat daripada t-SNE ($\mathcal{O}(N^2)$).
3. **Dukungan Transformasi Induktif**: UMAP menyediakan fungsi `.transform(X_new)`, sehingga dapat diintegrasikan secara langsung ke dalam arsitektur pipeline produksi.

---

## 10. Matriks Komparasi Menyeluruh Algoritma Reduksi Dimensi

| Dimensi Evaluasi | PCA | TruncatedSVD | t-SNE | UMAP |
|---|---|---|---|---|
| **Kategori Model** | Linier | Linier | Manifold Non-Linier | Manifold Non-Linier |
| **Preservasi Struktur** | Varians Global | Varians Global | Lingkungan Lokal | Lokal & Global |
| **Inversi (Reconstruction)** | Ya (`.inverse_transform`) | Ya (`.inverse_transform`) | Tidak | Ya (Aproksimasi) |
| **Transformasi Data Baru** | Ya (`.transform`) | Ya (`.transform`) | Tidak (Transduktif) | Ya (`.transform`) |
| **Sensitivitas Penskalaan** | Sangat Tinggi | Sangat Tinggi | Sedang | Sedang |
| **Kompleksitas Komputasi** | $\mathcal{O}(D^3 + N D^2)$ | $\mathcal{O}(N \cdot k \cdot \text{nnz})$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^{1.14})$ |
| **Skalabilitas Data ($N$)** | Sangat Tinggi ($N > 10^6$) | Sangat Tinggi ($N > 10^6$) | Terbatas ($N < 50.000$) | Tinggi ($N > 10^6$) |
| **Kasus Penggunaan Utama** | Kompresi, Preprocessing | Matriks Jarang (TF-IDF NLP) | Eksplorasi Visual 2D/3D | Visualisasi & Embedding |

---

## 11. Studi Kasus Komprehensif: Telemetri Jaringan Sensor IoT

Kami menguji reduksi dimensi pada 360 observasi sensor nirkabel (6 dimensi: daya sinyal, packet loss, latensi, suhu, voltase, energi mAh).

```
+-------------------------------------------------------------------------------------------+
|               RINGKASAN HASIL EKSPERIMEN REDUKSI DIMENSI SENSOR IOT                       |
+-------------------------------------------------------------------------------------------+
| Metode           Target Dimensi   Varians / KL Div     Waktu (ms)   Akurasi Model Hilir  |
| ----------------------------------------------------------------------------------------- |
| Data Asli (Raw)  6                100.00% Varians      -            94.44%               |
| PCA (k=4)        4                95.24% Varians       6.16 ms      94.44%               |
| PCA (k=2)        2                87.45% Varians       6.08 ms      96.67% (Optimal)     |
| TruncatedSVD     2                87.45% Varians       0.80 ms      96.67%               |
| t-SNE (perp=30)  2                KL Div: 0.5219       388.51 ms    (Visualisasi)        |
+-------------------------------------------------------------------------------------------+
```

### Kesimpulan Rekayasa Sistem:
1. **PCA 2D** mampu memangkas volume data sebesar $66.7\%$ dengan mempertahankan $87.45\%$ varians esensial.
2. Penurunan dimensi mengeliminasi multikolinearitas dan derau acak, terbukti dari peningkatan akurasi pengklasifikasi dari $94.44\%$ ke $96.67\%$.
3. **t-SNE** memetakan ketiga status node (Normal, Warning, Critical) menjadi gugusan pulau yang sangat terpisah jelas pada bidang 2D.

---

## 12. Latihan Bertingkat (Tiered Exercises)

### Tier 1: Pemahaman Konseptual
1. Mengapa pemusatan data (*mean centering*) merupakan syarat mutlak sebelum menghitung dekomposisi nilai eigen pada PCA?
   - *Jawaban*: Jika data tidak dipusatkan ke titik asal (mean=0), vektor komponen utama pertama ($\mathbf{w}_1$) hanya akan mengarah ke posisi pusat massa data dari titik origin $(0, \dots, 0)$, bukan mengarah ke sumbu varians variasi maksimal data.
2. Mengapa TruncatedSVD lebih disukai daripada PCA standar pada analisis teks berbasis matriks TF-IDF?
   - *Jawaban*: Matriks TF-IDF bersifat sangat jarang (*sparse*). Pemusatan data pada PCA standar akan mengubah ribuan entri nol menjadi nilai rata-rata negatif, mengubah matriks jarang menjadi padat (*dense*) dan menyebabkan kegagalan memori RAM (*Out-of-Memory*). TruncatedSVD menghitung dekomposisi langsung tanpa pemusatan.
3. Sebutkan dua alasan mengapa t-SNE tidak cocok digunakan sebagai tahap prapemrosesan fitur (*feature extraction*) dalam sistem inferensi produksi waktu nyata!
   - *Jawaban*: Pertama, t-SNE bersifat transduktif (tidak memiliki fungsi `.transform()` matematis untuk memproyeksikan data sampel baru). Kedua, t-SNE memiliki kompleksitas komputasi kuadratik $\mathcal{O}(N^2)$ yang lambat dan jarak global antar-klasternya tidak mencerminkan jarak sebenarnya.

### Tier 2: Perhitungan Matematis Manual
Diberikan matriks kovarians sampel $2 \times 2$ yang telah dihitung dari data terpusat:
$$\mathbf{\Sigma} = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$$

1. **Hitung Nilai Eigen ($\lambda$)**:
   $$\det(\mathbf{\Sigma} - \lambda \mathbf{I}) = \det \begin{pmatrix} 3 - \lambda & 1 \\ 1 & 3 - \lambda \end{pmatrix} = (3 - \lambda)^2 - 1 = 0$$
   $$\lambda^2 - 6\lambda + 8 = 0 \implies (\lambda - 4)(\lambda - 2) = 0$$
   Nilai eigen: $\lambda_1 = 4$ dan $\lambda_2 = 2$.
2. **Hitung Rasio Varians Terjelaskan**:
   $$\text{Total Varians} = 4 + 2 = 6$$
   $$\text{EVR}_1 = \frac{4}{6} = 66.67\%, \quad \text{EVR}_2 = \frac{2}{6} = 33.33\%$$
3. **Hitung Vektor Eigen Pertama ($\mathbf{v}_1$)**:
   $$(\mathbf{\Sigma} - 4\mathbf{I})\mathbf{v}_1 = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} v_{11} \\ v_{12} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \implies -v_{11} + v_{12} = 0 \implies v_{11} = v_{12}$$
   Normalisasi satuan: $\|\mathbf{v}_1\| = \sqrt{v_{11}^2 + v_{12}^2} = 1 \implies \mathbf{v}_1 = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{pmatrix} \approx \begin{pmatrix} 0.7071 \\ 0.7071 \end{pmatrix}$.

### Tier 3: Rekayasa Sistem & Desain Arsitektur
Sebuah sistem pemantauan gempa bumi memiliki 2.000 sensor seismik yang masing-masing merekam 100 fitur gelombang per detik ($D = 100$). Tim insinyur ingin mengirimkan data ringkas ke server pusat melalui saluran satelit dengan bandwidth sangat terbatas.
1. Rancang arsitektur pipeline kompresi dan rekonstruksi data menggunakan PCA!
   - *Jawaban*:
     - **Tahap Offline (Server)**: Latih model PCA pada data historis terstandarisasi untuk menentukan matriks proyeksi $\mathbf{W}_k \in \mathbb{R}^{100 \times 10}$ ($k=10$, mempertahankan $95\%$ varians). Simpan matriks $\mathbf{W}_k$ dan vektor rata-rata $\bar{\mathbf{x}}$ pada firmware sensor.
     - **Tahap Online (Edge Sensor)**: Setiap detik, sensor memusatkan pembacaan $\mathbf{x}_c = \mathbf{x} - \bar{\mathbf{x}}$ dan memproyeksikannya menjadi vektor kecil $\mathbf{z} = \mathbf{x}_c \mathbf{W}_k \in \mathbb{R}^{10}$. Sensor hanya mentransmisikan 10 nilai floating point (penghematan bandwidth $90\%$).
     - **Tahap Rekonstruksi (Server)**: Server merekonstruksi sinyal seismik lengkap $\hat{\mathbf{x}} = \mathbf{z} \mathbf{W}_k^T + \bar{\mathbf{x}}$.

---

## 13. Mini Project Mandiri: Sistem Kompresi & Visualisasi Seismik IoT (Rubrik 100 Poin)

Mahasiswa diwajibkan membangun modul reduksi dimensi lengkap untuk data telemetri berdimensi jamak.

### Rubrik Penilaian Proyek
- **Arsitektur Pipeline Bebas Kebocoran Data (25 Poin)**: Menghindari kebocoran data dengan membungkus `StandardScaler` dan `PCA` dalam Scikit-Learn Pipeline pada pembagian train-test.
- **Implementasi Komparasi PCA vs SVD (25 Poin)**: Menunjukkan kesetaraan nilai singular SVD terhadap varians PCA dan membuktikan Teorema Eckart-Young-Mirsky secara empiris.
- **Visualisasi Manifold 2D (25 Poin)**: Mengimplementasikan proyeksi t-SNE dengan eksplorasi rentang parameter perplexity (5, 30, 50) dan menyajikan interpretasi klaster.
- **Analisis Galat Rekonstruksi & Kompresi (25 Poin)**: Menghitung kurva galat norma Frobenius dari $k=1$ hingga $k=D$ dan memberikan rekomendasi rasio kompresi optimal bagi sistem industri.

---

## 14. Referensi Ilmiah Primer

1. **Pearson, K. (1901)**. *On lines and planes of closest fit to systems of points in space*. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 2(11), 559-572.
2. **Hotelling, H. (1933)**. *Analysis of a complex of statistical variables into principal components*. Journal of Educational Psychology, 24(6), 417-441.
3. **Eckart, C., & Young, G. (1936)**. *The approximation of one matrix by another of lower rank*. Psychometrika, 1(3), 211-218.
4. **van der Maaten, L., & Hinton, G. (2008)**. *Visualizing data using t-SNE*. Journal of Machine Learning Research, 9(Nov), 2579-2605.
5. **McInnes, L., Healy, J., & Melville, J. (2018)**. *UMAP: Uniform manifold approximation and projection for dimension reduction*. arXiv preprint arXiv:1802.03426.

