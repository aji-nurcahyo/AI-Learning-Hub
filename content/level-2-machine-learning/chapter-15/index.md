# Chapter 15: Decision Trees & Ensemble Learning

**Artificial Intelligence Learning Hub** — Level 2: Machine Learning  
*Created & Developed by Aji Pamungkas Tri Nurcahyo (Institut Teknologi Sumatera)*

---

## Deskripsi Modul & Tujuan Pembelajaran

Modul ini menyajikan pembahasan menyeluruh mengenai model pohon keputusan (*Decision Trees*) dan metode agregasi ansambel (*Ensemble Learning*). Pembahasan mencakup prinsip pemartisian rekursif, formulasi matematis kriteria ketidakmurnian (*Shannon Entropy* dan *Gini Impurity*), algoritma CART (*Classification and Regression Trees*), dinamika kompleksitas serta pemangkasan pohon (*pre-pruning* dan *cost-complexity post-pruning*). Selanjutnya, modul mendalami paradigma *Ensemble Learning*: *Bagging* (*Bootstrap Aggregating*), *Random Forest* dengan *feature subsampling* dan evaluasi *Out-of-Bag* (OOB), serta pembelajaran sekuensial *Boosting* (*AdaBoost* dan *Gradient Boosting*). Seluruh konsep dilengkapi dengan penurunan formula, diagram alur, verifikasi kode Scikit-learn, studi kasus sensor industri, latihan berjenjang, dan mini project.

### Prasyarat Pembelajaran
Sebelum mempelajari bab ini, mahasiswa diwajibkan telah menguasai:
1. **Chapter 06**: Probabilitas, Teori Informasi (Entropi Shannon), dan Aljabar Linier Dasar.
2. **Chapter 12**: Alur Kerja Machine Learning, Pemisahan Dataset Bebas Kebocoran, dan Validasi Silang.
3. **Chapter 13 & 14**: Paradigma Supervised Learning untuk Regresi (MSE, $R^2$) dan Klasifikasi (Akurasi, Presisi, Recall, F1-Score, Confusion Matrix).

### Capaian Pembelajaran Kognitif (Bloom's Taxonomy)
Setelah menuntaskan modul ini, mahasiswa mampu:
1. **Menjelaskan** konsep Decision Tree untuk tugas Classification dan Regression.
2. **Menginterpretasikan** struktur hierarki pohon: root node, internal decision nodes, branches, dan leaf nodes.
3. **Menghitung** Shannon Entropy, Gini Impurity, dan Information Gain secara matematis dari sampel data.
4. **Menjelaskan** mekanisme splitting kontinu dan diskret pada algoritma CART.
5. **Membedakan** binary splitting CART dengan pohon multiway (ID3/C4.5).
6. **Menganalisis** fenomena overfitting pada pohon dan merancang strategi pemangkasan (*pruning*).
7. **Menjelaskan** prinsip Bagging (Bootstrap Aggregating) dalam mereduksi varians model.
8. **Mengimplementasikan** Decision Tree dan Random Forest menggunakan Scikit-learn Pipeline.
9. **Menggunakan** metode Boosting sekuensial (AdaBoost dan Gradient Boosting).
10. **Membandingkan** performa model ensemble dengan baseline Decision Tree melalui validasi silang.
11. **Mengevaluasi** trade-off antara akurasi generalisasi, biaya komputasi, dan interpretability model.

---

## Part A: Decision Tree Foundations

### 1. Konsep Dasar dan Filosofi Pemodelan Pohon
Pohon Keputusan (*Decision Tree*) adalah model pembelajaran terawasi non-parametrik yang memprediksi nilai variabel target dengan mempelajari serangkaian aturan keputusan sederhana (IF–THEN) yang disimpulkan dari fitur data. Berbeda dengan model linier yang memproyeksikan data ke dalam bidang datar kontinu, Decision Tree mempartisi ruang fitur menjadi wilayah-wilayah hiper-persegi panjang (*hyper-rectangles*) yang saling lepas (*mutually exclusive*).

```text
+-------------------------------------------------------------------------+
|                  ALUR STRUKTUR POHON KEPUTUSAN (ANATOMI)                |
+-------------------------------------------------------------------------+
                               [ ROOT NODE ]
                           (Apakah Suhu > 80°C?)
                               /          \
                        (YA)  /            \  (TIDAK)
                             v              v
                   [ INTERNAL NODE ]     [ LEAF NODE ]
                 (Vibrasi > 4.5 mm/s?)   -> Prediksi: NORMAL
                     /          \
              (YA)  /            \  (TIDAK)
                   v              v
             [ LEAF NODE ]   [ LEAF NODE ]
          -> Prediksi: FAULT  -> Prediksi: WARNING
```

### 2. Hubungan dengan Basis Pengetahuan dan Sistem Pakar (Chapter 05)
Pada Chapter 05 (*Knowledge Representation & Reasoning*), aturan IF–THEN disusun secara manual oleh pakar domain manusia (*handcrafted expert system*). Dalam Machine Learning, Decision Tree membalik paradigma ini: algoritma secara otomatis menyaring data mentah historis untuk menemukan kombinasi aturan proposisional optimal yang memaksimalkan kemurnian target tanpa intervensi subjek pakar.

### 3. Pohon Klasifikasi vs Pohon Regresi
1. **Classification Trees**: Target $y \in \{0, 1, \dots, K-1\}$ berupa kategori diskret. Estimasi pada daun terminal adalah modus (kelas mayoritas) atau vektor probabilitas proporsi kelas:
   $$\hat{p}_{mk} = \frac{1}{N_m} \sum_{x_i \in R_m} \mathbb{I}(y_i = k)$$
2. **Regression Trees**: Target $y \in \mathbb{R}$ berupa nilai numerik kontinu. Estimasi pada daun terminal adalah rata-rata lokal sampel di wilayah tersebut:
   $$\hat{c}_m = \frac{1}{N_m} \sum_{x_i \in R_m} y_i$$
   Fungsi regresi yang dihasilkan berbentuk fungsi tangga bertingkat (*piecewise constant*).

---

## Part B: Kriteria Ketidakmurnian Node (*Splitting Criteria*)

### 1. Formulasi Matematika Shannon Entropy
Entropi mengukur derajat ketidakpastian atau keacakan distribusi informasi pada node $S$:
$$H(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$
Jika semua sampel pada node berasal dari kelas yang sama, $p_1 = 1$, maka $H(S) = - (1 \log_2 1) = 0$ (kemurnian mutlak). Pada kasus biner seimbang ($p_1 = p_2 = 0.5$), $H(S) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1.0$ bit.

### 2. Formulasi Matematika Gini Impurity
Gini Impurity mengukur frekuensi salah klasifikasi sampel jika diberi label acak sesuai distribusi kelas lokal:
$$I_G(S) = 1 - \sum_{k=1}^{K} p_k^2 = \sum_{k=1}^{K} p_k (1 - p_k)$$
Untuk kasus 2 kelas seimbang, $I_G(S) = 1 - (0.5^2 + 0.5^2) = 0.50$. Gini lebih hemat komputasi karena meniadakan perhitungan logaritma basis 2.

```text
+-------------------------------------------------------------------------+
|             KOMPARASI KURVA ENTROPI VS GINI IMPURITY (KASUS BINER)      |
+-------------------------------------------------------------------------+
  Nilai Impurity
   1.0 |           ***** (Entropi Shannon, Puncak = 1.0)
       |        *         *
   0.5 |       *  - - - - -  *  (Gini Impurity, Puncak = 0.5)
       |      *               *
   0.0 +-----+--------+--------+-----> Proporsi Kelas Positif (p)
      0.0   0.25     0.50     0.75   1.0
```

### 3. Information Gain (Penurunan Ketidakmurnian)
Information Gain ($IG$) adalah selisih antara ketidakmurnian node induk dengan rata-rata tertimbang ketidakmurnian node anak:
$$IG(S, A) = \text{Imp}(S) - \sum_{v \in \{\text{Left}, \text{Right}\}} \frac{|S_v|}{|S|} \text{Imp}(S_v)$$
Fitur dan ambang pemisah yang menghasilkan $IG$ terbesar dipilih untuk membagi node.

### 4. Contoh Perhitungan Manual Pemisahan Node
Diberikan 10 sampel sensor di node induk (4 NORMAL, 6 FAULT):
- Node Induk: $p_{\text{NORMAL}} = 0.4$, $p_{\text{FAULT}} = 0.6$.
  $$H(S) = - (0.4 \log_2 0.4 + 0.6 \log_2 0.6) = 0.9710 \text{ bits}$$
  $$I_G(S) = 1 - (0.4^2 + 0.6^2) = 0.4800$$
- Kandidat Pemisahan: Suhu $\le 75^\circ\text{C}$ vs Suhu $> 75^\circ\text{C}$:
  - Cabang Kiri (5 sampel: 4 Normal, 1 Fault):
    $$p_N = 0.8, p_F = 0.2 \implies H(S_L) = 0.7219 \text{ bits}, I_G(S_L) = 0.3200$$
  - Cabang Kanan (5 sampel: 0 Normal, 5 Fault [Murni]):
    $$p_N = 0.0, p_F = 1.0 \implies H(S_R) = 0.0000 \text{ bits}, I_G(S_R) = 0.0000$$
- Information Gain Entropi:
  $$IG = 0.9710 - \left( \frac{5}{10}(0.7219) + \frac{5}{10}(0) \right) = 0.9710 - 0.3610 = 0.6100 \text{ bits}$$

---

## Part C: Kompleksitas Pohon & Teknik Pemangkasan (*Pruning*)

### 1. Risiko Overfitting pada Pohon Tanpa Batas (*Unconstrained Tree*)
Pohon keputusan adalah model dengan varians tinggi. Tanpa pembatasan, pohon akan terus bercabang hingga setiap sampel data latih berada di daun murni berukuran 1 sampel. Pohon semacam ini memiliki akurasi latih 100%, namun gagal mengenali pola data uji akibat menghafal derau (*noise*).

```text
+-------------------------------------------------------------------------+
|                  SPEKTRUM KOMPLEKSITAS & PEMANGKASAN POHON              |
+-------------------------------------------------------------------------+
  [ Pohon Terlalu Dangkal ]        [ Pohon Terpangkas Optimal ]        [ Pohon Terlalu Dalam ]
       (Pre-Pruned, d=1)                  (Optimal, d=4)                  (Unpruned, d=15)
     Bias Tinggi (Underfit)            Keseimbangan Terbaik            Varians Tinggi (Overfit)
      Akurasi Uji: 68%                  Akurasi Uji: 91%                 Akurasi Uji: 83%
```

### 2. Pre-Pruning (*Early Stopping*)
Menghentikan pertumbuhan pohon secara proaktif saat kondisi ambang batas tercapai:
- `max_depth`: Kedalaman maksimum dari akar ke daun terjauh.
- `min_samples_split`: Jumlah sampel minimum yang harus dimiliki node untuk diizinkan bercabang.
- `min_samples_leaf`: Jumlah sampel minimum yang wajib ada di setiap daun terminal.

### 3. Post-Pruning (*Minimal Cost-Complexity Pruning*)
Algoritma CART menumbuhkan pohon penuh $T_0$, kemudian memangkas cabang dari bawah ke atas menggunakan fungsi biaya kompleksitas (*cost-complexity parameter* $\alpha \ge 0$):
$$R_\alpha(T) = R(T) + \alpha |T|$$
di mana $R(T)$ adalah total galat klasifikasi pada daun dan $|T|$ adalah jumlah daun terminal. Semakin besar nilai $\alpha$, semakin besar penalti terhadap jumlah daun, sehingga menghasilkan pohon yang lebih ringkas.

---

## Part D: Fondasi Pembelajaran Ansambel (*Ensemble Learning*)

### 1. Prinsip Agregasi & Teorema Juri Condorcet
Pembelajaran ansambel menggabungkan sekelompok model dasar (*weak/base learners*) untuk membangun satu prediktor unggul. Berdasarkan *Condorcet's Jury Theorem*, jika masing-masing dari $B$ model independen memiliki probabilitas kebenaran $p > 0.5$, maka probabilitas suara mayoritas benar mendekati $1.0$ saat $B \to \infty$.

```text
+-------------------------------------------------------------------------+
|                     ARSITEKTUR BAGGING (PARALEL)                        |
+-------------------------------------------------------------------------+
                          [ Dataset Latih Asli (N) ]
                        /            |            \
             Bootstrap /   Bootstrap |   Bootstrap \
                      v              v              v
                 [ Set B_1 ]    [ Set B_2 ]    [ Set B_M ]
                      |              |              |
                      v              v              v
                  [ Tree 1 ]     [ Tree 2 ]     [ Tree M ]
                      \              |              /
                       \             |             /
                        v            v            v
                      [ AGREGASI VOTING / RATA-RATA ]
                                     |
                                     v
                             [ Prediksi Final ]
```

### 2. Bootstrap Sampling & Sampel Out-of-Bag (OOB)
Dari $N$ data asli, diambil $N$ sampel dengan pengembalian (*sampling with replacement*). Peluang suatu sampel tidak pernah terpilih dalam satu putaran bootstrap adalah:
$$P(\text{tidak terpilih}) = \left(1 - \frac{1}{N}\right)^N \xrightarrow[N \to \infty]{} \frac{1}{e} \approx 0.368$$
Artinya, setiap pohon rata-rata dilatih pada $\approx 63.2\%$ data unik. Sisanya $\approx 36.8\%$ sampel disebut sampel **Out-of-Bag (OOB)**. Sampel OOB bertindak sebagai *validation set* mandiri tanpa perlu menyisihkan data latih.

### 3. Random Forest: De-korelasi Pohon via Feature Subsampling
Random Forest (Breiman, 2001) menyempurnakan Bagging dengan memperkenalkan sumber keacakan kedua: pada setiap pemisahan node, algoritma hanya menguji subset acak sebanyak $m$ fitur dari total $p$ fitur:
- Klasifikasi: $m = \lfloor \sqrt{p} \rfloor$.
- Regresi: $m = \lfloor p / 3 \rfloor$.

Feature subsampling mencegah fitur yang dominan terpilih di akar seluruh pohon, sehingga de-korelasi antar-pohon tercapai dan varians ensemble turun drastis:
$$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
Ketika korelasi antar-pohon $\rho \to 0$, varians ensemble mendekati nol seiring bertambahnya jumlah pohon $B$.

---

## Part E: Algoritma Boosting

### 1. Prinsip Pembelajaran Sekuensial Adaptif
Jika Bagging melatih pohon secara paralel untuk menurunkan varians, Boosting melatih pohon secara bertahap (*sekuensial*) untuk menurunkan bias. Setiap model baru dikondisikan untuk fokus memperbaiki kesalahan prediksi model sebelumnya.

```text
+-------------------------------------------------------------------------+
|                     ARSITEKTUR BOOSTING (SEKUENSAL)                     |
+-------------------------------------------------------------------------+
  [ Data Latih ] ---> [ Model 1 ] --(Kalkulasi Galat)--> [ Bobot Sampel Diperbarui ]
                                                                   |
                                                                   v
  [ Prediksi Final ] <--- [ Model 3 ] <--(Residu Galat)-- [ Model 2 ]
```

### 2. AdaBoost (Adaptive Boosting)
Pada AdaBoost, sampel yang salah diprediksi oleh *decision stump* ke-$m$ dinaikkan bobotnya, sedangkan sampel yang benar diturunkan bobotnya. Model akhir menggabungkan seluruh stump berdasarkan tingkat akurasi masing-masing:
$$H(x) = \text{sign}\left( \sum_{m=1}^{M} \alpha_m h_m(x) \right), \quad \alpha_m = \frac{1}{2} \ln\left( \frac{1 - \epsilon_m}{\epsilon_m} \right)$$

### 3. Gradient Tree Boosting
Gradient Boosting merumuskan proses boosting sebagai optimasi numerik penurunan gradien pada ruang fungsi. Setiap pohon baru dilatih untuk memprediksi **residu gradien negatif** dari fungsi *loss*:
$$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$
Model diperbarui dengan mekanisme penyusutan (*shrinkage / learning rate* $\eta \in (0, 1]$):
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

---

## Part F: Evaluasi Komparatif & Atribusi Fitur

### 1. Mean Decrease in Impurity (MDI) dan Keterbatasannya
MDI menghitung rata-rata penurunan ketidakmurnian yang disumbangkan oleh setiap fitur di seluruh pohon dalam hutan.
**Keterbatasan Kritis MDI**:
1. Cenderung melebih-lebihkan kepentingan fitur kontinu berkardinalitas tinggi atau derau acak berfluktuasi tinggi.
2. Ketika dua fitur berkorelasi tinggi, salah satu fitur dapat merebut skor dominan secara acak sementara fitur pasangannya tampak tidak penting.

### 2. Tabel Komparasi Karakteristik Model Pohon & Ansambel

| Parameter Evaluasi | Single Decision Tree | Bagging Classifier | Random Forest | AdaBoost | Gradient Boosting |
|---|---|---|---|---|---|
| **Kapasitas Model** | Rendah - Sedang | Sedang - Tinggi | Sangat Tinggi | Tinggi | Sangat Tinggi |
| **Fokus Reduksi** | - | Varians | Varians | Bias | Bias & Varians |
| **Sensitivitas Outlier** | Tinggi | Rendah | Rendah | Sangat Tinggi | Sedang (Huber Loss) |
| **Kebutuhan Tuning** | Rendah (`max_depth`) | Rendah (`n_est`) | Rendah (`n_est`, `sqrt`) | Sedang (`lr`, `n_est`) | Tinggi (`lr`, `depth`, `n_est`) |
| **Kemudahan Interpretasi**| **Sangat Tinggi (Aturan IF-THEN)** | Rendah (Black-box) | Rendah (Feature Imp) | Sedang | Rendah (Gradien) |
| **Latensi Inferensi** | **Sub-milidetik (< 1 ms)** | Cepat (< 10 ms) | Cepat (< 15 ms) | Cepat (< 20 ms) | Sangat Cepat (< 5 ms) |

---

## Part G: Latihan Soal Terstruktur & Pembahasan

### Soal 1 (Tingkat Dasar): Membaca Struktur Pohon Keputusan
Perhatikan representasi tekstual pohon keputusan berikut:
```text
|--- Suhu > 82.50 °C
|   |--- Vibrasi <= 4.20 mm/s: WARNING (sampel: [0, 18, 2])
|   |--- Vibrasi > 4.20 mm/s: FAULT (sampel: [0, 1, 29])
|--- Suhu <= 82.50 °C
|   |--- Tegangan > 365.00 V: NORMAL (sampel: [42, 2, 0])
|   |--- Tegangan <= 365.00 V: WARNING (sampel: [1, 12, 1])
```
Jika sebuah mesin beroperasi dengan `Suhu = 85.0 °C`, `Vibrasi = 5.10 mm/s`, dan `Tegangan = 380 V`, tentukan prediksi status mesin tersebut dan jelaskan jalur keputusannya!

**Kunci Jawaban Soal 1**:
1. Evaluasi Root Node: Apakah `Suhu > 82.50`? Ya ($85.0 > 82.50$). Ikuti cabang pertama.
2. Evaluasi Internal Node: Apakah `Vibrasi > 4.20`? Ya ($5.10 > 4.20$). Ikuti cabang kedua.
3. Node Daun Tercapai: Daun memiliki distribusi `[NORMAL: 0, WARNING: 1, FAULT: 29]`.
4. **Prediksi Akhir: FAULT** dengan tingkat keyakinan probabilitas $\frac{29}{30} = 96.67\%$.

### Soal 2 (Tingkat Menengah): Menghitung Gini Impurity & Information Gain
Sebuah node internal memiliki 20 sampel (12 Sehat, 8 Rusak). Pengujian fitur membelah data menjadi:
- Cabang Kiri: 10 sampel (9 Sehat, 1 Rusak)
- Cabang Kanan: 10 sampel (3 Sehat, 7 Rusak)
Hitunglah: (a) Gini Impurity node induk, (b) Gini Impurity kedua node anak, dan (c) Penurunan Gini (*Gini Gain*)!

**Kunci Jawaban Soal 2**:
- (a) Gini Node Induk: $I_G(S) = 1 - \left[\left(\frac{12}{20}\right)^2 + \left(\frac{8}{20}\right)^2\right] = 1 - [0.36 + 0.16] = 1 - 0.52 = 0.4800$.
- (b) Gini Node Anak:
  - Anak Kiri: $I_G(S_L) = 1 - \left[\left(\frac{9}{10}\right)^2 + \left(\frac{1}{10}\right)^2\right] = 1 - [0.81 + 0.01] = 0.1800$.
  - Anak Kanan: $I_G(S_R) = 1 - \left[\left(\frac{3}{10}\right)^2 + \left(\frac{7}{10}\right)^2\right] = 1 - [0.09 + 0.49] = 0.4200$.
- (c) Rata-rata tertimbang ketidakmurnian anak:
  $$I_{G,\text{split}} = \frac{10}{20}(0.1800) + \frac{10}{20}(0.4200) = 0.0900 + 0.2100 = 0.3000$$
  $$\text{Gini Gain} = 0.4800 - 0.3000 = 0.1800$$
Pemisahan berhasil mereduksi ketidakmurnian sebesar $0.1800$.

### Soal 3 (Tingkat Mahir): Analisis Dinamika Varians Random Forest
Buktikan secara matematis mengapa meningkatkan jumlah pohon $B$ dalam Random Forest tidak akan menyebabkan overfitting, tetapi hanya membuat estimasi varians konvergen ke suatu nilai batas!

**Kunci Jawaban Soal 3**:
Varians dari rata-rata $B$ variabel acak identik dengan varians $\sigma^2$ dan korelasi pasangan positif $\rho$ adalah:
$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} T_b(x)\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
Ketika $B \to \infty$, suku kedua $\frac{1 - \rho}{B} \sigma^2 \to 0$, sehingga:
$$\lim_{B \to \infty} \text{Var}(\text{Forest}) = \rho \sigma^2$$
Menambah jumlah pohon hanya mengeliminasi varians sampling bootstrap dari ensemble; varians tidak akan kembali naik. Overfitting pada Random Forest dikendalikan oleh kompleksitas masing-masing pohon individu (dikontrol via `max_depth` dan `min_samples_leaf`), bukan oleh nilai $B$.

---

## Part H: Mini Project — Klasifikasi Kerusakan Mesin Industri

### 1. Deskripsi Masalah & Skenario Industri
Pabrik pengolahan semen mengoperasikan motor induksi tegangan tinggi. Mahasiswa diminta mengembangkan arsitektur sistem peringatan dini cerdas menggunakan data telemetri 6 sensor (`temperature_c`, `vibration_mm_s`, `voltage_v`, `current_a`, `operating_hours`, `load_percentage`) untuk mengklasifikasikan kondisi mesin ke dalam status `NORMAL`, `WARNING`, dan `FAULT`.

### 2. Panduan Pelaksanaan & Deliverables
1. **Inspeksi Dataset**: Analisis sebaran 360 sampel pada `data/equipment_fault_sensor_data.csv`.
2. **Baseline Model**: Bangun `DummyClassifier(strategy='most_frequent')` sebagai patokan performa terendah.
3. **Eksplorasi Decision Tree**: Latih pohon tunggal, bandingkan Gini vs Entropy, dan visualisasikan aturan IF-THEN.
4. **Analisis Pemangkasan**: Lakukan eksperimen pre-pruning dan cost-complexity post-pruning (`ccp_alpha`).
5. **Implementasi Ansambel**: Latih Random Forest, AdaBoost, dan Gradient Boosting menggunakan 5-Fold Stratified Cross-Validation.
6. **Laporan Teknis**: Dokumentasikan matriks konfusi, F1-Score tiap kelas, dan rekomendasi model produksi dengan pertimbangan latensi vs keandalan.

### 3. Rubrik Penilaian Mini Project (Skala 100 Poin)

| Komponen Penilaian | Bobot | Kriteria Penuh (Full Credit) | Kriteria Sebagian (Partial Credit) | Kurang Bukti (Insufficient) |
|---|---|---|---|---|
| **Pemahaman Masalah & Dataset** | **15 Poin** | Menganalisis distribusi ketiga kelas, memeriksa ketiadaan missing value, dan memisahkan train/test secara terstratifikasi. | Melakukan split data tanpa stratifikasi kelas target. | Menggunakan seluruh data tanpa pemisahan hold-out. |
| **Implementasi Decision Tree & Pruning** | **20 Poin** | Mengimplementasikan Gini/Entropy, visualisasi aturan, dan analisis post-pruning `ccp_alpha` dengan kurva akurasi. | Melatih pohon tunggal namun tidak melakukan analisis pemangkasan. | Kode pohon gagal berjalan atau tidak tervalidasi. |
| **Implementasi Model Ensemble** | **25 Poin** | Menerapkan Random Forest, AdaBoost, dan Gradient Boosting lengkap dengan evaluasi OOB dan tuning hyperparameter. | Mengimplementasikan hanya 1 model ensemble. | Tidak ada implementasi ensemble sama sekali. |
| **Evaluasi Objektif & Analisis Galat** | **25 Poin** | Evaluasi 5-Fold CV yang adil, analisis Confusion Matrix, F1-Score makro, dan pembahasan keterbatasan MDI feature importance. | Hanya melaporkan akurasi latih atau metrik tunggal. | Tidak ada analisis evaluasi yang terstruktur. |
| **Dokumentasi & Referensi Ilmiah** | **15 Poin** | Kode rapi terdokumentasi (PEP 8), menyertakan kesimpulan teknik berbobot dan minimal 5 referensi akademik sah. | Dokumentasi minim, pembahasan singkat tanpa rujukan. | Tidak ada dokumentasi atau kesimpulan teknis. |

---

## Part I: Referensi Akademik Primer

1. **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5–32.  
   DOI / URL: [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
2. **Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J.** (1984). *Classification and Regression Trees*. Wadsworth & Brooks/Cole Advanced Books & Software.  
   URL: [https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418](https://www.routledge.com/Classification-and-Regression-Trees/Breiman-Friedman-Stone-Olshen/p/book/9780412048418)
3. **Friedman, J. H.** (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. The Annals of Statistics, 29(5), 1189–1232.  
   DOI / URL: [https://doi.org/10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
4. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer New York.  
   URL: [https://hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/)
5. **ACM / IEEE-CS / AAAI**. (2024). *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*.  
   URL: [https://csed.acm.org/](https://csed.acm.org/)

