# Panduan Praktik Laboratorium — Chapter 11: Introduction to Machine Learning

Selamat datang di laboratorium **Chapter 11: Introduction to Machine Learning**. Praktikum ini merupakan gerbang pembuka menuju **Level 2: Machine Learning**. Anda akan mempraktikkan konsep-konsep fundamental pembelajaran mesin secara empiris menggunakan Python, NumPy, Pandas, dan Scikit-learn.

---

## 1. Tujuan Praktik

Setelah menyelesaikan seluruh langkah praktikum mandiri ini, Anda diharapkan mampu:
1. Membandingkan secara langsung perilaku sistem berbasis aturan manual (*if-else*) dengan model Machine Learning berbasis data (*data-driven*).
2. Mengekstrak dan menyusun matriks fitur 2D ($X$) dan vektor target 1D ($y$) dari data mentah tabular.
3. Mengimplementasikan model Supervised Learning sederhana untuk tugas Regresi dan Klasifikasi.
4. Menerapkan algoritma Unsupervised Learning (K-Means) untuk menemukan pola klaster pada data tanpa label.
5. Membagi dataset menjadi *Training Set* dan *Testing Set* menggunakan `train_test_split` dan membuktikan bahaya ilusi hafalan (*memorization*).
6. Menganalisis fenomena *Underfitting*, *Good Fit*, dan *Overfitting* melalui perbandingan derajat polinomial (*Bias–Variance Trade-off*).
7. Membangun model pembanding dasar (*Baseline Model*) untuk mengevaluasi klaim performa model secara objektif pada data tidak seimbang.

---

## 2. Struktur Direktori Praktikum

Seluruh kode sumber mandiri dan notebook interaktif tersimpan dalam struktur direktori berikut:

```text
content/level-2-machine-learning/chapter-11/
├── code/
│   ├── rule_based_vs_learning.py      # Komparasi sistem aturan manual vs Naive Bayes
│   ├── features_and_labels.py         # Ekstraksi matriks X dan vektor target y
│   ├── supervised_learning_demo.py    # Demonstrasi Regresi Linier dan Klasifikasi Logistik
│   ├── unsupervised_learning_demo.py  # Demonstrasi K-Means clustering data energi 2D
│   ├── training_testing_demo.py       # Pembagian train/test split & generalisasi k-NN
│   ├── overfitting_underfitting_demo.py # Bias-variance trade-off polinomial (deg 1, 3, 15)
│   ├── baseline_comparison.py         # DummyClassifier baseline vs Logistic Regression
│   └── README.md                      # Dokumentasi kode dan tabel hasil eksekusi aktual
├── notebooks/
│   └── chapter_11_companion.ipynb     # Interactive Jupyter Companion Notebook (Format v4)
├── practice.md                        # Panduan praktikum laboratorium (berkas ini)
├── index.md                           # Modul teori komprehensif (Single Source of Truth)
├── slides/
│   └── deck.json                      # Slide presentasi pengajaran Reveal.js (28 slide)
├── audit.md                           # Laporan audit mutu publikasi bab
└── metadata.json                      # Registri metadata resmi kurikulum
```

---

## 3. Prasyarat Lingkungan & Persiapan

Praktikum ini menggunakan lingkungan virtual Python standar (`.venv`) dengan dependensi:
- `python >= 3.10`
- `scikit-learn >= 1.3.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`

Aktivasi virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Verifikasi instalasi pustaka:
```bash
python -c "import sklearn, pandas, numpy; print('Lingkungan laboratorium ML Chapter 11 siap!')"
```

---

## 4. Panduan Eksekusi Langkah demi Langkah

### Langkah 1: Rule-Based vs Machine Learning
Jalankan skrip komparasi:
```bash
python content/level-2-machine-learning/chapter-11/code/rule_based_vs_learning.py
```
**Analisis yang Diperhatikan:**
- Amati bagaimana aturan *if-else* kaku gagal mendeteksi pesan spam saat kata kunci divariasikan ("Dapatkan bonus tanpa bayar").
- Perhatikan bagaimana model Naive Bayes menghitung asosiasi probabilitas dari kumpulan kata latih.

### Langkah 2: Fitur (Features) dan Label (Target)
Jalankan skrip inspeksi struktur data:
```bash
python content/level-2-machine-learning/chapter-11/code/features_and_labels.py
```
**Analisis yang Diperhatikan:**
- Kolom pengenal identitas (`student_id`) dibuang dari matriks $X$ karena tidak memiliki korelasi generalisasi.
- Perhatikan bentuk matriks $X$ yang selalu berdimensi 2D $(N, D)$ dan target $y$ yang berdimensi 1D $(N,)$.

### Langkah 3: Supervised Learning (Regresi & Klasifikasi)
Jalankan modul supervised learning:
```bash
python content/level-2-machine-learning/chapter-11/code/supervised_learning_demo.py
```
**Analisis yang Diperhatikan:**
- Pada regresi, amati bagaimana model memprediksi nilai daya kontinu (Watt) dengan metrik MSE dan $R^2$.
- Pada klasifikasi, amati probabilitas kelas yang diprediksi (`predict_proba`) untuk menentukan batas keputusan.

### Langkah 4: Unsupervised Learning (Clustering)
Jalankan modul clustering tanpa label:
```bash
python content/level-2-machine-learning/chapter-11/code/unsupervised_learning_demo.py
```
**Analisis yang Diperhatikan:**
- Data tidak memiliki kolom jawaban target $y$.
- K-Means mengelompokkan 12 profil energi gedung menjadi 3 kelompok alami berdasarkan jarak Euclidean ke titik pusat (*centroids*).

### Langkah 5: Pembagian Data Latih dan Uji (Train/Test Split)
Jalankan skrip validasi generalisasi:
```bash
python content/level-2-machine-learning/chapter-11/code/training_testing_demo.py
```
**Analisis yang Diperhatikan:**
- Model $k$-NN dengan $k=1$ menghasilkan akurasi data latih sempurna ($100\%$), tetapi akurasi data ujinya anjlok ke $90\%$.
- Model dengan $k=5$ memiliki akurasi latih lebih rendah ($91.4\%$), namun akurasi ujinya mencapai $100\%$ karena tidak menghafal derau acak.

### Langkah 6: Eksplorasi Bias–Variance Trade-off
Jalankan skrip analisis overfitting & underfitting:
```bash
python content/level-2-machine-learning/chapter-11/code/overfitting_underfitting_demo.py
```
**Analisis yang Diperhatikan:**
- Derajat 1: *Underfitting* (High Bias) — Train MSE dan Test MSE sama-sama tinggi karena model terlalu sederhana untuk kurva sinus.
- Derajat 3: *Good Fit* — Train MSE dan Test MSE seimbang dan rendah.
- Derajat 15: *Overfitting* (High Variance) — Train MSE mendekati 0, namun Test MSE meningkat tajam karena model meliuk-liuk menghafal derau.

### Langkah 7: Pengujian Baseline Model pada Data Tidak Seimbang
Jalankan modul komparasi baseline:
```bash
python content/level-2-machine-learning/chapter-11/code/baseline_comparison.py
```
**Analisis yang Diperhatikan:**
- Pada dataset dengan $95\%$ data sah dan $5\%$ penipuan (*fraud*), model `DummyClassifier` yang selalu menebak kelas mayoritas memperoleh akurasi $95\%$, padahal *Balanced Accuracy*-nya hanya $50\%$ (gagal total mendeteksi *fraud*).
- Jangan pernah mengevaluasi akurasi tanpa membandingkannya dengan *baseline*.

---

## 5. Tugas Mandiri & Eksplorasi Lanjutan

1. **Eksplorasi Rasio Pembagian Latih/Uji**: Ubah proporsi `test_size` pada `training_testing_demo.py` menjadi $0.10$ dan $0.50$. Amati bagaimana perubahan ukuran data uji memengaruhi stabilitas estimasi akurasi generalisasi.
2. **Pencarian $k$ Optimal pada $k$-NN**: Tulis loop kecil untuk menguji nilai $k$ dari 1 sampai 15 pada `training_testing_demo.py`. Buat grafik perbandingan antara akurasi latih dan akurasi uji untuk menemukan titik keseimbangan terbaik.
3. **Penyisipan Fitur Baru**: Tambahkan fitur buatan ketiga (misalnya rasio CPU/RAM) pada `supervised_learning_demo.py` dan periksa apakah nilai $R^2$ meningkat.

---

## 6. Rubrik Penilaian Praktikum Laboratorium

| Komponen Penilaian | Bobot | Kriteria Keberhasilan Penuh |
| :--- | :---: | :--- |
| **Pemahaman Struktur Data** | 20% | Pemisahan matriks fitur $X$ dan vektor target $y$ dilakukan secara presisi tanpa kebocoran data. |
| **Eksekusi & Analisis Kode** | 30% | Seluruh 7 skrip Python dijalankan tanpa galat dan keluarannya diinterpretasikan secara ilmiah. |
| **Analisis Overfitting & Baseline** | 30% | Mampu menjelaskan secara grafis/numerik mengapa training error rendah bukan jaminan generalisasi yang baik. |
| **Dokumentasi & Reproducibility** | 20% | Kode rapi, random seed terdokumentasi, dan kesimpulan disusun dengan terminologi teknis yang tepat. |
| **Total Skor** | **100%** | **Standar Kelulusan Laboratorium ML: Minimal 80%** |

