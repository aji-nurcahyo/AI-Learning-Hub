# Contoh Kode Python: Chapter 01 — Introduction to Artificial Intelligence

Direktori ini berisi lima contoh skrip Python mandiri yang mendemonstrasikan konsep dasar sistem berbasis aturan (*Rule-Based System*), klasifikasi kata kunci, kerapuhan *exact matching*, serta kalkulasi eksplisit metrik akurasi dan *confusion counts*.

Seluruh skrip menggunakan pustaka standar Python (*standard library*) murni tanpa ketergantungan pada model komersial berbayar atau unduhan berat.

## Lingkungan Eksekusi & Pengujian
- **Interpreter**: Python 3.12.10 (Windows 64-bit)
- **Ketergantungan Eksternal**: Tidak ada (hanya built-in types dan `typing`)
- **Status Eksekusi**: 100% Lolos pengujian langsung

---

## Daftar Contoh Skrip

### 1. `01_rule_based_recommender.py`
- **Tujuan**: Menunjukkan sistem penentuan keputusan deterministik berbasis aturan kondisional (`if-elif-else`).
- **Cara Menjalankan**:
  ```bash
  python 01_rule_based_recommender.py
  ```
- **Output Aktual**:
  ```text
  === DEMO: SISTEM REKOMENDASI BERBASIS ATURAN ===

  Profil: Mahasiswa A (Minat: computer vision, Level: pemula)
    1. Fondasi Python & Struktur Data Dasar
    2. Dasar Representasi Matriks Citra Digital
    3. Prinsip Responsible AI & Mitigasi Bias

  Profil: Mahasiswa B (Minat: natural language processing, Level: menengah)
    1. Large Language Models & Fine-Tuning
    2. Prinsip Responsible AI & Mitigasi Bias

  Profil: Mahasiswa C (Minat: finance, Level: pemula)
    1. Fondasi Python & Struktur Data Dasar
    2. Dasar-dasar Algoritma AI & Representasi Pengetahuan
    3. Prinsip Responsible AI & Mitigasi Bias
  ```
- **Keterbatasan & Pertanyaan Eksplorasi**:
  - *Keterbatasan*: Logika bersifat kaku (*brittle*). Input di luar kategori yang diantisipasi programmer akan jatuh ke aturan umum atau gagal.
  - *Eksplorasi*: Apa yang terjadi jika pengguna memasukkan minat "pengolahan bahasa alami"? Bagaimana Machine Learning menyelesaikan variasi sinonim tersebut?

---

### 2. `02_keyword_text_classifier.py`
- **Tujuan**: Menunjukkan cara kerja filter teks berbasis kata kunci dan mendemonstrasikan munculnya kesalahan False Positive dan False Negative.
- **Cara Menjalankan**:
  ```bash
  python 02_keyword_text_classifier.py
  ```
- **Output Aktual**:
  ```text
  === DEMO: KEYWORD-BASED TEXT CLASSIFIER ===

  [Kasus 1] Status: BENAR
    Pesan       : "Selamat Anda menang undian berhadiah 100 juta!"
    Label Asli  : SPAM
    Prediksi    : SPAM
    Analisis    : True Positive (Berhasil terdeteksi karena ada 'menang' dan 'berhadiah')

  [Kasus 2] Status: BENAR
    Pesan       : "Besok kita kumpul di perpustakaan jam 10 pagi ya."
    Label Asli  : HAM
    Prediksi    : HAM
    Analisis    : True Negative (Pesan normal tanpa kata kunci spam)

  [Kasus 3] Status: SALAH (ERROR)
    Pesan       : "Buku referensi kuliah ini gratis dipinjam dari perpustakaan."
    Label Asli  : HAM
    Prediksi    : SPAM
    Analisis    : False Positive! (Pesan akademik normal tapi dianggap SPAM karena kata 'gratis')

  [Kasus 4] Status: SALAH (ERROR)
    Pesan       : "Tolong kirimkan sejumlah dana darurat ke rekening ini segera."
    Label Asli  : SPAM
    Prediksi    : HAM
    Analisis    : False Negative! (Pesan penipuan nyata tapi lolos karena memakai kata 'dana', bukan 'transfer')
  ```
- **Keterbatasan & Pertanyaan Eksplorasi**:
  - *Keterbatasan*: Mengabaikan sintaksis dan semantik kalimat. Kata "gratis" dalam konteks buku perpustakaan disamakan dengan spam hadiah.

---

### 3. `03_exact_matching_vs_variation.py`
- **Tujuan**: Membandingkan pencocokan string eksak dengan variasi input alami manusia (kapitalisasi, tanda baca, salah ketik, sinonim).
- **Cara Menjalankan**:
  ```bash
  python 03_exact_matching_vs_variation.py
  ```
- **Output Aktual**:
  ```text
  === PERBANDINGAN: EXACT MATCHING VS VARIASI INPUT ===
  Kueri acuan pada database: 'kapan jadwal uas'

  Input Pengguna                           | Hasil Pencocokan          | Status
  --------------------------------------------------------------------------------
  kapan jadwal uas                         | Jawaban Ditemukan         | SUKSES (Input persis sesuai kamus)
  Kapan jadwal UAS?                        | Tidak Diketahui (None)    | GAGAL (Variasi tanda baca tanda tanya dan huruf kapital)
  kapan jadual uas                         | Tidak Diketahui (None)    | GAGAL (Variasi salah ketik / typo ('jadual'))
  kapan ujian akhir semester dimulai       | Tidak Diketahui (None)    | GAGAL (Variasi sinonim / frasa formal)
  kapan uas diadakan                       | Tidak Diketahui (None)    | GAGAL (Variasi struktur kalimat)
  ```
- **Keterbatasan & Pertanyaan Eksplorasi**:
  - *Keterbatasan*: Exact matching hanya berhasil 20% pada variasi sederhana. Pendekatan vector embeddings atau fuzzy matching diperlukan untuk sistem interaksi nyata.

---

### 4. `04_explicit_accuracy_calc.py`
- **Tujuan**: Menghitung metrik akurasi dari daftar ground truth dan prediksi secara eksplisit menggunakan rumus matematika dasar.
- **Cara Menjalankan**:
  ```bash
  python 04_explicit_accuracy_calc.py
  ```
- **Output Aktual**:
  ```text
  === KALKULASI EKSPLISIT METRIK AKURASI ===

  Jumlah total sampel uji : 10
  Data Aktual (Ground Truth): [1, 0, 0, 1, 0, 0, 1, 1, 0, 1]
  Prediksi Sistem            : [1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
  --------------------------------------------------
  Sampel #01: Aktual=1, Pred=1 -> BENAR
  Sampel #02: Aktual=0, Pred=0 -> BENAR
  Sampel #03: Aktual=0, Pred=1 -> SALAH
  Sampel #04: Aktual=1, Pred=1 -> BENAR
  Sampel #05: Aktual=0, Pred=0 -> BENAR
  Sampel #06: Aktual=0, Pred=0 -> BENAR
  Sampel #07: Aktual=1, Pred=0 -> SALAH
  Sampel #08: Aktual=1, Pred=1 -> BENAR
  Sampel #09: Aktual=0, Pred=0 -> BENAR
  Sampel #10: Aktual=1, Pred=1 -> BENAR
  --------------------------------------------------
  Prediksi benar : 8 dari 10 sampel
  Akurasi Sistem : 0.8000 (80.0%)
  ```
- **Peringatan Penting**:
  - Data dalam skrip ini bersifat sintetis untuk ilustrasi perhitungan, bukan data produksi.

---

### 5. `05_simple_confusion_analysis.py`
- **Tujuan**: Menghitung matriks kontingensi biner (TP, FP, TN, FN) untuk menunjukkan mengapa akurasi tinggi bisa menyesatkan pada data tidak seimbang (*imbalanced data*).
- **Cara Menjalankan**:
  ```bash
  python 05_simple_confusion_analysis.py
  ```
- **Output Aktual**:
  ```text
  === DEMO: CONFUSION COUNTS & ANALISIS ERROR ===
  Total sampel dievaluasi : 20
  Distribusi aktual       : 16 Normal, 4 Fraud
  ----------------------------------------------------
  Tabel Matriks Kontingensi Sederhana (Confusion Matrix):
                     | Prediksi Positif (1) | Prediksi Negatif (0)
  ----------------------------------------------------------------
  Aktual Positif (1) | TP = 2               | FN = 2              
  Aktual Negatif (0) | FP = 2               | TN = 14             
  ----------------------------------------------------------------

  Akurasi Keseluruhan : 80.0%
  Rincian Kesalahan:
  - False Positive (FP) : 2 transaksi normal diblokir keliru (mengganggu pengguna)
  - False Negative (FN) : 2 transaksi penipuan lolos tak terdeteksi (kerugian finansial)
  ```
- **Wawasan Kunci**:
  - Akurasi 80% menyembunyikan kenyataan bahwa 50% transaksi penipuan gagal ditangkap oleh sistem.

