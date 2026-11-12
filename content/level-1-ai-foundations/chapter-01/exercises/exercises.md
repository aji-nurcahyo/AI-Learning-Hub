# Latihan Terstruktur: Chapter 01 — Introduction to Artificial Intelligence

Lembar latihan ini dirancang untuk menguji dan memperdalam pemahaman konseptual, analitis, dan evaluatif Anda terhadap materi Chapter 01. Selesaikan latihan secara mandiri sebelum memeriksa panduan jawaban di bagian akhir dokumen ini.

---

## Bagian A: Tingkat Dasar (Mengidentifikasi Konsep)

### Soal 1: Klasifikasi Sistem Cerdas vs Otomasi Deterministik
Perhatikan lima skenario sistem perangkat lunak berikut. Tentukan apakah masing-masing skenario tergolong **Sistem AI** atau **Otomasi Komputasi Non-AI**. Berikan alasan singkat 1 kalimat untuk setiap jawaban Anda.
1. Script Python yang menghitung total harga belanjaan setelah dipotong diskon member 10% dan pajak 11%.
2. Sistem kamera keamanan bandara yang mendeteksi koper yang ditinggalkan tanpa pemilik selama lebih dari 15 menit menggunakan pengenalan objek visual.
3. Query basis data PostgreSQL yang mengurutkan 10.000 mahasiswa berdasarkan IPK tertinggi secara menurun (*descending*).
4. Filter email yang secara otomatis memindahkan email promosi ke tab khusus berdasarkan pola kata dan reputasi pengirim yang terus diperbarui dari waktu ke waktu.
5. Algoritma pencarian kata eksak (*exact match string searching*) pada dokumen PDF menggunakan fungsi `Ctrl+F`.

### Soal 2: Taksonomi Pendekatan Kecerdasan Buatan
Pasangkan setiap deskripsi berikut dengan istilah yang paling tepat: **Rule-Based System**, **Machine Learning Klasik**, **Deep Learning**, atau **Generative AI**.
- (a) Model memprediksi harga sewa rumah berdasarkan fitur luas tanah, jumlah kamar, dan jarak ke stasiun menggunakan algoritma Linear Regression.
- (b) Sistem penentuan kelayakan kredit perbankan yang menggunakan pohon keputusan logika `IF gaji > 10jt AND riwayat_kredit == 'LANCAR' THEN 'SETUJU'`.
- (c) Arsitektur Convolutional Neural Network (CNN) 50-layer yang mendeteksi retakan mikro pada sayap pesawat terbang langsung dari foto beresolusi tinggi.
- (d) Model Transformer berskala besar yang menghasilkan paragraf ringkasan eksekutif dari transkrip rekaman rapat secara otomatis.

---

## Bagian B: Tingkat Menengah (Membandingkan Pendekatan & Menghitung Metrik)

### Soal 3: Anatomi Sistem Cerdas
Sebuah rumah sakit swasta berencana membangun prototipe sistem pendeteksi aritmia jantung otomatis berbasis rekaman elektrokardiogram (EKG) berdurasi 30 detik.
Uraikan kelima komponen sistem cerdas tersebut sesuai prinsip anatomis:
1. **Input**: Apakah bentuk fisik data mentah yang diterima sistem?
2. **Representation**: Bagaimana sinyal analog mentah tersebut diubah menjadi representasi numerik yang dapat diproses algoritma?
3. **Processing**: Apa peran algoritma pemodelan dalam sistem ini?
4. **Output**: Apakah keluaran spesifik yang disajikan kepada dokter jaga?
5. **Evaluation**: Bagaimana cara mengukur keandalan sistem tersebut sebelum diuji coba pada pasien nyata?

### Soal 4: Kalkulasi Metrik Evaluasi & Analisis Kesalahan
Sebuah model kecerdasan buatan diuji pada 100 sampel transaksi keuangan digital untuk mendeteksi transaksi penipuan (*fraud detection*). Berikut data hasil evaluasi:
- Transaksi normal riil: 90 transaksi
- Transaksi penipuan riil: 10 transaksi

Hasil prediksi model menunjukkan:
- Dari 10 transaksi penipuan riil, model berhasil menangkap 7 transaksi sebagai penipuan, sedangkan 3 transaksi lolos dianggap normal.
- Dari 90 transaksi normal riil, model memprediksi 85 transaksi sebagai normal, namun 5 transaksi normal keliru diblokir sebagai penipuan.

**Pertanyaan:**
1. Hitung nilai **True Positive (TP)**, **False Positive (FP)**, **True Negative (TN)**, dan **False Negative (FN)**!
2. Hitung nilai **Akurasi keseluruhan** model tersebut!
3. Jelaskan dampak bisnis nyata dari kesalahan **False Positive** dan **False Negative** dalam konteks perbankan ini! Kesalahan mana yang membawa kerugian lebih fatal?

---

## Bagian C: Tingkat Lanjut (Analisis Kebutuhan & Risiko Sistem)

### Soal 5: Analisis Kelayakan Masalah (AI vs Non-AI Decision)
Sebuah dinas perpustakaan daerah ingin membuat sistem otomatisasi untuk menangani pertanyaan masyarakat seputar:
1. Jam operasional perpustakaan dan syarat pendaftaran anggota baru.
2. Rekomendasi buku bacaan yang cocok bagi pembaca yang menyukai novel misteri bertema sejarah Indonesia.

**Instruksi Analisis:**
- Untuk kebutuhan (1), pendekatan mana yang paling tepat: *Rule-Based System / FAQ Database sederhana* atau *LLM Generative AI*? Jelaskan risiko jika dinas memaksakan penggunaan LLM untuk kebutuhan tersebut!
- Untuk kebutuhan (2), mengapa pendekatan pencarian kata kunci eksak (*exact matching*) sering kali mengecewakan pengguna? Bagaimana Machine Learning dapat memberikan solusi yang lebih baik?

### Soal 6: Evaluasi Risiko dan Tata Kelola Responsible AI
Sebuah startup edutech merancang sistem AI untuk memprediksi mahasiswa yang berisiko tidak lulus tepat waktu. Model ini dilatih menggunakan data akademik historis selama 7 tahun terakhir, yang memuat nilai ujian, indeks prestasi semester, daerah asal sekolah, dan status pekerjaan orang tua.

Berdasarkan 5 pilar **Responsible AI** (Fairness, Transparency, Privacy, Accountability, Safety):
1. **Fairness**: Identifikasi potensi bias yang mungkin timbul jika fitur latar belakang sosial ekonomi (seperti pekerjaan orang tua atau asal sekolah) dimasukkan ke dalam pelatihan model!
2. **Transparency**: Mengapa mahasiswa yang diprediksi "berisiko tinggi tidak lulus" berhak menerima penjelasan (*explainability*), bukan hanya skor angka probabilitas?
3. **Human-in-the-Loop**: Rancang mekanisme keterlibatan dosen pembimbing akademik (*Human-in-the-Loop*) dalam menindaklanjuti keluaran sistem ini!

---

# Pembahasan dan Panduan Jawaban

> **Petunjuk bagi Mahasiswa:** Buka bagian ini hanya setelah Anda berupaya menjawab seluruh pertanyaan secara mandiri.

### Pembahasan Soal 1
1. **Non-AI**: Perhitungan diskon dan pajak adalah formula deterministik beraturan tetap tanpa proses inferensi probabilistik atau adaptasi pola data.
2. **Sistem AI**: Melibatkan visi komputer (Computer Vision) untuk membedakan objek koper dari latar belakang visual yang berubah-ubah di lingkungan dinamis.
3. **Non-AI**: Operasi pengurutan basis data adalah algoritma sorting klasik (misal QuickSort/B-Tree indexing) dengan kriteria pasti.
4. **Sistem AI**: Menggunakan klasifikasi teks berbasis pembelajaran mesin yang memperbarui representasi fitur dan batas keputusan dari data email baru.
5. **Non-AI**: Pencocokan string eksak adalah algoritma string matching deterministik (seperti Boyer-Moore atau KMP).

### Pembahasan Soal 2
- (a) **Machine Learning Klasik** (Regresi Linier mempelajari bobot numerik dari fitur tabel terstruktur).
- (b) **Rule-Based System** (Logika diskrit IF-THEN eksplisit yang disusun langsung oleh pakar kredit).
- (c) **Deep Learning** (CNN multi-layer mengekstrak representasi visual langsung dari piksel mentah tanpa rekayasa fitur manual).
- (d) **Generative AI** (Model Transformer menyintesis teks baru melalui pemodelan probabilitas token bahasa).

### Pembahasan Soal 3
1. **Input**: Sinyal voltase analog atau deret waktu digital (time-series voltage) rekaman elektroda EKG per milidetik.
2. **Representation**: Ekstraksi fitur gelombang (interval gelombang P-Q-R-S-T, frekuensi detak jantung via Fourier Transform) atau spektrogram 2D sinyal.
3. **Processing**: Model klasifikasi (misal Random Forest atau 1D-Convolutional Network) menghitung probabilitas keberadaan pola ritme abnormal (aritmia).
4. **Output**: Peringatan visual pada layar monitor (misal: "Terdeteksi Indikasi Fibrilasi Atrium, Probabilitas 91%") beserta penanda segmen waktu EKG yang anomali.
5. **Evaluation**: Pengujian pada dataset uji independen yang telah dianotasi oleh konsorsium dokter spesialis jantung, menggunakan metrik Sensitivitas (Recall) dan Spesifisitas.

### Pembahasan Soal 4
1. **Kuadran Confusion Matrix**:
   - $\text{TP} = 7$ (Penipuan terdeteksi benar)
   - $\text{FN} = 3$ (Penipuan lolos tidak terdeteksi)
   - $\text{FP} = 5$ (Transaksi normal salah diblokir)
   - $\text{TN} = 85$ (Transaksi normal lolos benar)
2. **Kalkulasi Akurasi**:
   $$\text{Akurasi} = \frac{\text{TP} + \text{TN}}{\text{Total Sampel}} = \frac{7 + 85}{100} = \frac{92}{100} = 92.0\%$$
3. **Analisis Dampak Bisnis**:
   - **False Positive (5 kasus)**: Nasabah jujur yang sedang berbelanja tiba-tiba kartunya terblokir. Dampaknya: komplain nasabah, penurunan reputasi layanan, dan biaya verifikasi panggilan telepon bank.
   - **False Negative (3 kasus)**: Penipuan nyata berhasil membobol saldo nasabah tanpa ada peringatan. Dampaknya: kerugian uang riil dan potensi tuntutan hukum.
   - **Kesimpulan**: Dalam perbankan, **False Negative jauh lebih fatal** daripada False Positive. Model dengan akurasi tinggi (92%) ini sebenarnya masih sangat berbahaya karena membiarkan 30% penipuan lolos (3 dari 10)!

### Pembahasan Soal 5
- **Kebutuhan (1)**: Paling tepat menggunakan **Rule-Based System / Basis Data FAQ terstruktur**. Alasannya: jam buka dan syarat pendaftaran adalah informasi faktual pasti yang tidak boleh keliru. Jika memaksakan LLM tanpa sistem kendali ketat (*grounding*), LLM dapat berhalusinasi mengarang jam operasional yang salah kepada masyarakat.
- **Kebutuhan (2)**: Exact matching gagal karena pembaca tidak selalu mengetik kata "misteri sejarah". Seseorang mungkin mencari "cerita detektif zaman perang kemerdekaan". Machine Learning (dengan representasi *embedding* vektor semantik) mampu menangkap kesamaan makna di balik variasi frasa tersebut.

### Pembahasan Soal 6
1. **Fairness & Bias**: Memasukkan status sosial ekonomi atau asal daerah dapat menyebabkan sistem menghukum mahasiswa dari daerah tertinggal secara sistemik, semata-mata karena data masa lalu menunjukkan angka kelulusan daerah tersebut lebih rendah akibat keterbatasan fasilitas lama. Ini melanggengkan diskriminasi historis. Fitur harus difokuskan pada aktivitas akademik objektif saat ini.
2. **Transparency**: Mahasiswa berhak mengetahui faktor apa yang membuat mereka diberi label berisiko (misal: "Kehadiran praktikum di bawah 70% dan nilai kuis 1 rendah") agar mereka tahu aspek konkret apa yang harus diperbaiki untuk mencegah ketidaklulusan.
3. **Human-in-the-Loop**: Sistem AI hanya memberikan notifikasi peringatan dini kepada Dosen Wali/PA. Dosen Wali kemudian memanggil mahasiswa untuk sesi konseling personal guna memahami kendala riil (apakah masalah sakit, keluarga, atau metode belajar), sebelum mengambil tindakan pembinaan akademik.

