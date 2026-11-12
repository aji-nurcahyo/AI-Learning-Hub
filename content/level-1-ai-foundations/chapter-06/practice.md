# Panduan Praktik Laboratorium — Chapter 06: Probability, Statistics & Linear Algebra for AI

Selamat datang di modul praktik laboratorium **Chapter 06: Probability, Statistics & Linear Algebra for AI**. Modul ini dirancang untuk menjembatani teori matematis abstrak dengan komputasi numerik konkret menggunakan Python standard library dan konsep-konsep foundational kecerdasan buatan.

---

## 1. Tujuan Praktik

Setelah menyelesaikan seluruh kegiatan praktik pada laboratorium ini, Anda diharapkan mampu:
1. Menjalankan skrip analisis statistik deskriptif untuk data sensor dunia nyata dan mendeteksi anomali/pencilan.
2. Memverifikasi aturan dasar probabilitas, probabilitas bersyarat, dan pengujian independensi secara empiris.
3. Mengimplementasikan Teorema Bayes lengkap dengan Hukum Probabilitas Total dan menganalisis fenomena *Base Rate Fallacy*.
4. Mengoperasikan manipulasi aljabar vektor (penjumlahan, perkalian skalar, norma $L_1$/$L_2$, *Dot Product*, dan *Cosine Similarity*).
5. Melakukan operasi matriks formal (transposisi, perkalian matriks $\mathbf{A}\mathbf{B}$, dan pembedaan terhadap perkalian element-wise Hadamard $\mathbf{A} \odot \mathbf{B}$).
6. Menghitung turunan pertama secara analitis dan membandingkannya dengan metode pendekatan beda pusat (*central difference*).
7. Mengamati konvergensi algoritma *Gradient Descent* pada fungsi kuadratik serta ilustrasi matematis optimasi fungsi *Loss Mean Squared Error* (MSE).

---

## 2. Struktur Berkas Praktik

Seluruh kode sumber mandiri dan notebook interaktif tersimpan pada struktur direktori berikut:

```text
content/level-1-ai-foundations/chapter-06/
├── code/
│   ├── descriptive_statistics.py   # Statistik pemusatan, dispersi, kuartil & outlier
│   ├── probability_basics.py       # Ruang sampel, komplemen, irisan, bersyarat, binomial
│   ├── bayes_theorem.py            # Teorema Bayes & sensitivitas prior sensor alarm
│   ├── vector_operations.py        # Penjumlahan vektor, norma L1/L2, dot product, cosine similarity
│   ├── matrix_operations.py        # Transpose, perkalian matriks formal vs Hadamard
│   ├── derivative_example.py       # Turunan analitis vs numerik, gradien 2D
│   ├── gradient_descent_demo.py    # Gradient Descent univariat & MSE loss update
│   └── README.md                   # Katalog berkas kode dan tabel output terverifikasi
├── notebooks/
│   └── chapter_06_companion.ipynb  # Interactive Jupyter Companion Notebook (Format v4)
├── practice.md                     # Panduan komprehensif praktikum (berkas ini)
└── index.md                        # Modul pembelajaran teori utama (Single Source of Truth)
```

---

## 3. Prasyarat Lingkungan & Persiapan Eksekusi

### Lingkungan Python
- **Python**: Versi 3.10 atau lebih baru (direkomendasikan Python 3.11 atau 3.12).
- **Dependensi Tambahan**: Skrip pada direktori `code/` dirancang **100% menggunakan Python Standard Library** (`math`, `sys`, `collections`, `typing`) sehingga dapat langsung dijalankan tanpa instalasi paket pihak ketiga.
- Jika Anda ingin menjalankan berkas notebook interaktif (`.ipynb`), pastikan Jupyter Notebook atau ekstensi Jupyter di VS Code telah aktif.

### Verifikasi Terminal
Buka terminal pada akar proyek (`D:\Workstation\Repo My Corses\Kecerdasan Buatan`), lalu jalankan pengecekan versi:
```bash
python --version
```
Pastikan menghasilkan `Python 3.11.x` atau `Python 3.12.x`.

---

## 4. Langkah demi Langkah Praktik Mandiri

### Langkah 1: Eksplorasi Statistik Deskriptif & Pencilan
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-06/code/descriptive_statistics.py
```
**Eksperimen yang diamati:**
- Bandingkan nilai **Mean (24.2692 °C)** dan **Median (23.1000 °C)**. Mengapa mean jauh lebih tinggi dari sebagian besar data? Karena data memuat anomali pembacaan `35.0 °C`.
- Amati perbedaan antara **Varians Populasi ($\sigma^2 = 10.1667$)** dan **Varians Sampel ($s^2 = 11.0140$)**. Perbedaan ini timbul karena pembagi pada sampel adalah $N-1$ (koreksi Bessel) untuk menghindari *underestimation*.
- Perhatikan batas atas Tukey ($Q_3 + 1.5 \times \text{IQR} = 26.4250\text{ }^\circ\text{C}$). Data $35.0\text{ }^\circ\text{C}$ sukses terdeteksi sebagai outlier.

### Langkah 2: Probabilitas Dasar & Independensi Stokastik
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-06/code/probability_basics.py
```
**Eksperimen yang diamati:**
- Pada 100 paket data jaringan, probabilitas paket korup $P(A) = 0.10$ dan paket prioritas tinggi $P(B) = 0.30$.
- Nilai $P(A \cap B) = 0.03$. Karena $P(A) \times P(B) = 0.10 \times 0.30 = 0.03$, kedua kejadian terbukti independen secara statistik.
- Perhatikan perhitungan nilai harapan latensi $E[X] = 17.50$ ms berdasarkan distribusi bobot probabilitas.

### Langkah 3: Investigasi Teorema Bayes & Base Rate Fallacy
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-06/code/bayes_theorem.py
```
**Eksperimen yang diamati:**
- Skenario: Sensor kebakaran dengan akurasi/sensitivitas $95\%$ dan tingkat alarm palsu $5\%$.
- Meskipun sensor berbunyi, probabilitas kebakaran sesungguhnya **hanya 16.10%**!
- Mengapa? Karena prior kejadian kebakaran sangat rendah ($1\%$). Jumlah alarm palsu dari $99\%$ kondisi normal jauh melampaui jumlah alarm benar dari $1\%$ kebakaran. Fenomena psikologis dan teknis ini disebut *Base Rate Fallacy*.

### Langkah 4: Aljabar Vektor & Cosine Similarity pada Dokumen
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-06/code/vector_operations.py
```
**Eksperimen yang diamati:**
- Dua dokumen AI dengan proporsi kata sama $\mathbf{u} = [3, 4, 0]$ dan $\mathbf{v} = [6, 8, 0]$ menghasilkan **Cosine Similarity = 1.0000** (sudut $\theta = 0^\circ$).
- Dokumen ekonomi $\mathbf{w} = [0, 0, 5]$ memiliki *dot product* bernilai nol terhadap $\mathbf{u}$, menghasilkan Cosine Similarity = 0.0000 (ortogonal / tidak berkorelasi).

### Langkah 5: Operasi Matriks & Perbedaan Hadamard
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-06/code/matrix_operations.py
```
**Eksperimen yang diamati:**
- Perkalian matriks data $\mathbf{X}(3 \times 2)$ dengan bobot layer $\mathbf{W}(2 \times 3)$ menghasilkan matriks aktivasi $\mathbf{Y}(3 \times 3)$.
- Perhatikan tabel perbedaan output antara perkalian Hadamard ($\mathbf{P} \odot \mathbf{Q}$) yang mengalikan elemen demi elemen pada posisi sama, dibandingkan perkalian matriks formal ($\mathbf{P}\mathbf{Q}$) yang melakukan penjumlahan baris kali kolom (*dot product*).

### Langkah 6: Turunan, Gradien, dan Arah Penurunan Tercuram
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-06/code/derivative_example.py
```
**Eksperimen yang diamati:**
- Bandingkan turunan analitis $f'(x) = 2x - 4$ dengan aproksimasi numerik *central difference*. Galat absolut terbukti sangat kecil ($< 10^{-10}$).
- Titik $x = 2.0$ menghasilkan gradien $0.0$, yang menandai lokasi titik minimum global fungsi kuadratik.

### Langkah 7: Algoritma Gradient Descent & Optimasi Bobot MSE
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-06/code/gradient_descent_demo.py
```
**Eksperimen yang diamati:**
- Mengamati pergerakan nilai parameter $x_t$ dari $-2.0$ menuju nilai optimal $3.0$ dengan laju pembelajaran $\alpha = 0.1$.
- Membandingkan variasi learning rate: $\alpha=0.01$ (lambat), $\alpha=0.1$ (ideal), dan $\alpha=1.05$ (divergen/meledak).
- Mengamati pembaruan bobot model linear $w$ pada fungsi loss Mean Squared Error (MSE), di mana nilai loss turun mendekati $0.0000$ saat bobot mencapai $w = 2.0$.

---

## 5. Pertanyaan Refleksi Laboratorium

1. Mengapa dalam sistem machine learning kita lebih sering menggunakan *Sample Variance* ($N-1$) daripada *Population Variance* ($N$) ketika mengevaluasi data training?
2. Jika sebuah model klasifikasi citra medis memiliki akurasi $99\%$, dalam kondisi apa model tersebut justru menghasilkan lebih banyak diagnosa positif palsu daripada positif sejati?
3. Apa perbedaan konsekuensi komputasi antara perkalian matriks $\mathbf{A}_{m \times k} \mathbf{B}_{k \times n}$ dan perkalian Hadamard $\mathbf{A}_{m \times n} \odot \mathbf{B}_{m \times n}$ dalam arsitektur deep learning?
4. Mengapa pemilihan nilai *Learning Rate* ($\alpha$) yang terlalu besar pada Gradient Descent dapat menyebabkan fungsi loss bernilai `NaN` atau divergen menuju tak hingga?

