# Chapter 01: Introduction to Artificial Intelligence

Selamat datang di modul pembelajaran perdana Artificial Intelligence Learning Hub. Modul ini meletakkan fondasi konseptual, analitis, dan metodologis untuk memahami apa itu Artificial Intelligence, bagaimana berbagai paradigma di dalamnya berkembang, bagaimana sebuah sistem cerdas dirancang dari hulu ke hilir, serta batasan-batasan nyata yang membedakan komputasi cerdas dari klaim berlebihan (*hype*).

Modul ini adalah **satu sumber utama pembelajaran** yang memuat seluruh materi konseptual, studi kasus terapan, lembar latihan terstruktur berjenjang, pembahasan soal, panduan mini project, rubrik penilaian berbobot 100 poin, serta referensi rujukan ilmiah resmi.

---

## 1. Mengapa Mempelajari Artificial Intelligence?

Dalam beberapa dekade terakhir, ilmu komputer telah mengalami lompatan besar. Pada era komputasi klasik, manusia memprogram komputer dengan menuliskan algoritma deterministik langkah demi langkah. Pendekatan ini sangat unggul untuk menyelesaikan masalah dengan aturan pasti, seperti menghitung neraca keuangan, mengurutkan data angka, atau memproses transaksi basis data.

Namun, dunia nyata dipenuhi oleh masalah yang **tidak dapat diselesaikan dengan aturan kaku**:
- Bagaimana mengenali wajah seseorang dari foto yang diambil pada sudut miring dan pencahayaan redup?
- Bagaimana memahami maksud kalimat manusia yang sarat kiasan, konteks lokal, atau salah ketik?
- Bagaimana mendeteksi transaksi penipuan baru yang sengaja menyamarkan perilakunya di antara jutaan transaksi normal?

Artificial Intelligence (AI) hadir untuk menjawab persoalan-persoalan tersebut. AI memungkinkan sistem komputasi menyelesaikan tugas-tugas perseptual, kognitif, dan probabilistik yang sebelumnya hanya dapat dikerjakan oleh kecerdasan manusia. Menguasai AI bukan sekadar mempelajari kode pemrograman, melainkan memahami cara memodelkan ketidakpastian dunia nyata ke dalam struktur matematika dan algoritma yang bertanggung jawab.

---

## 2. Definisi Kerja Artificial Intelligence dan Keterbatasannya

Secara historis, para ilmuwan telah mengusulkan banyak definisi untuk Artificial Intelligence. Dalam literatur akademik standar ilmu komputer (seperti Russell & Norvig), AI didefinisikan secara operasional:

> **Definisi Kerja:**
> **Artificial Intelligence** adalah subdisiplin ilmu komputer yang berfokus pada perancangan agen perangkat lunak atau sistem komputasi yang mampu mempersepsikan lingkungannya (*perception*), memproses informasi, menarik kesimpulan (*reasoning*), dan mengambil tindakan terarah guna mencapai tujuan tertentu secara rasional.

### Keterbatasan Definisi "Kecerdasan"
Penting untuk menyadari bahwa istilah "cerdas" dalam AI adalah **analogi fungsional**, bukan kesetaraan biologis:
1. **Bukan Kesadaran (*No Sentience*)**: Sistem AI tidak memiliki kesadaran, perasaan, intensi personal, atau pemahaman eksistensial. Sebuah model pengenal wajah tidak "melihat" atau "mengagumi" wajah; ia hanya menghitung operasi matriks pada nilai piksel.
2. **Kecerdasan Sempit (*Narrow AI*)**: Semua sistem AI yang beroperasi di dunia saat ini adalah *Artificial Narrow AI* (ANI). Sistem ini sangat mahir mengerjakan satu domain tugas spesifik (misalnya mendiagnosis foto rontgen dada atau bermain catur), tetapi sama sekali tidak dapat mentransfer kemampuannya untuk mengemudikan mobil atau membuat puisi tanpa pelatihan ulang total.
3. **Efek AI (*The AI Effect*)**: Ada kecenderungan paradoks di mana ketika sebuah masalah AI berhasil dipecahkan dan dipahami secara luas (misalnya *optical character recognition* atau pencarian rute terpendek GPS), orang cenderung tidak lagi menganggapnya sebagai "kecerdasan", melainkan sekadar "komputasi biasa".

---

## 3. AI sebagai Bidang yang Mencakup Beragam Pendekatan

Kerap terjadi kesalahpahaman bahwa "AI sama dengan Machine Learning" atau "semua AI harus belajar dari data". Pandangan ini keliru. AI adalah payung disiplin ilmu yang sangat luas yang mencakup dua rumpun pendekatan besar:

```
                          ┌───────────────────────────────────────────┐
                          │     ARTIFICIAL INTELLIGENCE (AI)          │
                          │   Bidang studi sistem agen cerdas        │
                          └─────────────────────┬─────────────────────┘
                                                │
             ┌──────────────────────────────────┴──────────────────────────────────┐
             ▼                                                                     ▼
┌─────────────────────────────┐                                       ┌─────────────────────────────┐
│  Symbolic AI / Rule-Based   │                                       │   Data-Driven / Learning    │
│  • Sistem Berbasis Aturan   │                                       │   • Machine Learning        │
│  • Logika Proposisi & First-│                                       │   • Deep Learning           │
│    Order Logic              │                                       │   • Reinforcement Learning  │
│  • Pencarian Graf & Heuristik                                       │   • Generative AI & LLM     │
│  • Ontologi & Knowledge Base│                                       │                             │
└─────────────────────────────┘                                       └─────────────────────────────┘
```

1. **Symbolic AI (Pendekatan Berbasis Pengetahuan & Aturan)**: Memanipulasi simbol-simbol eksplisit dan kaidah logika formal yang didefinisikan oleh pakar manusia.
2. **Data-Driven AI (Pendekatan Berbasis Pembelajaran Data)**: Menemukan fungsi matematis atau bobot parameter probabilitas secara otomatis dari observasi dataset.

Keduanya adalah bagian sah dari AI. Bahkan banyak sistem industri modern yang sukses menggabungkan keduanya menjadi sistem hibrida (*neuro-symbolic AI*).

---

## 4. Rule-Based System dan Symbolic Approaches

Sebelum dominasi Machine Learning modern, **Rule-Based System** (Sistem Berbasis Aturan) adalah tulang punggung kecerdasan buatan, terutama pada era kejayaan *Expert Systems* (Sistem Pakar).

### Cara Kerja Rule-Based System
Sistem berbasis aturan terdiri dari tiga komponen inti:
1. **Knowledge Base (Basis Pengetahuan)**: Kumpulan fakta terverifikasi mengenai domain tertentu (misalnya daftar gejala penyakit medis atau regulasi perpajakan).
2. **Rule Engine / Inference Engine (Mesin Inferensi)**: Mekanisme logika (seperti *Forward Chaining* atau *Backward Chaining*) yang mencocokkan fakta saat ini dengan aturan bersyarat `IF kondisi THEN aksi`.
3. **Working Memory (Memori Kerja)**: Status sementara dari data pengguna atau sesi interaksi saat ini.

### Perbandingan: Rule-Based vs Machine Learning

| Dimensi | Rule-Based System | Machine Learning |
|---|---|---|
| **Sumber Logika** | Dirumuskan langsung secara manual oleh ahli manusia | Diinferensikan secara otomatis oleh algoritma dari dataset |
| **Kebutuhan Data** | Tidak memerlukan dataset besar untuk pelatihan | Memerlukan data berkualitas dan berlabel dalam jumlah memadai |
| **Transparansi (*Auditability*)** | 100% dapat dilacak (setiap keputusan memiliki rantai aturan jelas) | Sering kali bersifat *black-box* (terutama model berparameter besar) |
| **Ketahanan terhadap Variasi** | Sangat kaku (*brittle*); gagal bila menemui kasus di luar aturan | Adaptif dan mampu menggeneralisasi variasi pola baru |
| **Biaya Pemeliharaan** | Eksponensial membengkak jika jumlah aturan mencapai ribuan | Bergantung pada pipeline *monitoring*, retraining, dan pembersihan data |

> **Kapan Menggunakan Rule-Based System?**
> Jika domain masalah Anda memiliki regulasi hukum yang kaku (misalnya ketentuan eligibility beasiswa universitas, kalkulasi tarif pajak bea cukai, atau protokol medis wajib), sistem berbasis aturan sering kali jauh lebih unggul, aman, dan dapat dipertanggungjawabkan daripada model Machine Learning yang bersifat probabilistik.

---

## 5. Hubungan AI, Machine Learning, dan Deep Learning

Hubungan ketiga istilah ini sering kali digambarkan sebagai lingkaran konsentris, meskipun dalam praktiknya batas-batas antar disiplin ini terus berkembang dinamis:

```
┌────────────────────────────────────────────────────────┐
│ ARTIFICIAL INTELLIGENCE                                │
│ Kemampuan sistem komputasi meniru fungsi kognitif      │
│ ┌────────────────────────────────────────────────────┐ │
│ │ MACHINE LEARNING                                   │ │
│ │ Belajar pola representasi langsung dari data       │ │
│ │ ┌────────────────────────────────────────────────┐ │ │
│ │ │ DEEP LEARNING                                  │ │ │
│ │ │ Neural Network multi-layer (lapisan mendalam)  │ │ │
│ │ │ ┌────────────────────────────────────────────┐ │ │ │
│ │ │ │ GENERATIVE AI & LLM                        │ │ │ │
│ │ │ │ Sintesis konten baru via model fondasi     │ │ │ │
│ │ │ └────────────────────────────────────────────┘ │ │ │
│ │ └────────────────────────────────────────────────┘ │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

1. **Artificial Intelligence (AI)**: Konsep terluas mengenai mesin yang mampu bertindak secara rasional.
2. **Machine Learning (ML)**: Pendekatan spesifik di dalam AI di mana komputer tidak diprogram aturan logikanya secara manual, melainkan menggunakan algoritma statistik (seperti Linear Regression, Support Vector Machine, Decision Tree, atau K-Means) untuk meminimalkan *error* prediksi pada data.
3. **Deep Learning (DL)**: Cabang dari Machine Learning yang menggunakan arsitektur jaringan saraf tiruan berlapisan dalam (*Artificial Neural Networks*). Deep Learning memecahkan kendala utama ML klasik: kebutuhan rekayasa fitur manual (*hand-crafted feature engineering*). Deep Learning mampu belajar mengekstraksi fitur mentah (seperti piksel foto atau gelombang audio) secara hierarkis bertingkat.

---

## 6. Generative AI dan LLM dalam Konteks Tersebut

Dalam beberapa tahun terakhir, istilah **Generative AI** (GenAI) dan **Large Language Model** (LLM) menjadi pusat perhatian publik. Di manakah letak posisinya dalam arsitektur AI?

- **Discriminative vs Generative**:
  - *Model Diskriminatif* (ML/DL Tradisional): Mempelajari batas keputusan untuk membedakan kategori. Pertanyaan yang dijawab: *"Apakah email ini spam atau bukan?"* atau *"Apakah foto ini kucing atau anjing?"* ($P(Y|X)$).
  - *Model Generatif* (GenAI): Mempelajari distribusi probabilitas gabungan untuk membentuk data baru yang menyerupai data pelatihan. Pertanyaan yang dijawab: *"Buatkan gambar kucing baru yang realistis"* atau *"Tuliskan draf proposal penelitian bertema AI"* ($P(X)$ atau $P(X|Y)$).
- **Large Language Models (LLM)**: Model deep learning berarsitektur Transformer dengan miliaran parameter yang dilatih pada korpus teks internet berskala masif. Tugas intinya sangat sederhana namun berdaya guna: memprediksi probabilitas kemunculan token kata berikutnya (*next token prediction*).

> **Peringatan Ilmiah:**
> Kefasihan tata bahasa (*fluency*) dari LLM bukan bukti bahwa model memiliki penalaran logis sejati atau pemahaman semantik. LLM bekerja atas dasar korelasi statistik token kata, sehingga rentan menghasilkan pernyataan yang terdengar sangat meyakinkan namun salah secara faktual (*hallucination*).

---

## 7. Training dan Inference pada Pendekatan Berbasis Model

Sistem berbasis model Machine Learning beroperasi dalam dua fase siklus hidup yang terpisah secara tegas:

### 1. Fase Pelatihan (Training Phase)
- **Aktivitas**: Algoritma menerima data latih (*training data*) beserta label targetnya (pada supervised learning). Model menghitung prediksi, mengukur seberapa jauh selisih prediksi dengan nilai riil menggunakan fungsi kerugian (*loss function*), lalu memperbarui bobot parameter internalnya.
- **Karakteristik Komputasi**: Sangat intensif komputasi, membutuhkan waktu berjam-jam hingga berminggu-minggu, sering kali memerlukan GPU paralel atau klaster server akselerator.

### 2. Fase Inferensi (Inference Phase)
- **Aktivitas**: Model yang bobot parameternya telah dibekukan (*frozen weights*) dipasang ke sistem produksi (*deployment*). Ketika pengguna memasukkan data baru yang belum pernah dilihat model, model melakukan satu kali komputasi maju (*forward pass*) untuk menghasilkan output prediksi.
- **Karakteristik Komputasi**: Cepat (dalam hitungan milidetik), hemat daya, dan dapat dijalankan di CPU, perangkat mobile, atau sistem edge tertanam.

```
[DATA LATIH] ────> [ PROSES TRAINING ] ────> [ MODEL TERLATIH (BOBOT TETAP) ]
                         ▲                                │
                         │ (Optimasi Bobot)                ▼
                    [LOSS FUNCTION]              [ PROSES INFERENCE ] <─── [DATA BARU]
                                                          │
                                                          ▼
                                                  [OUTPUT PREDIKSI]
```

---

## 8. Anatomi Sistem AI: Input, Representation, Processing, Output & Evaluation

Setiap sistem kecerdasan buatan di dunia nyata dapat dibedah ke dalam lima blok fungsional berurutan:

```
┌─────────┐      ┌────────────────┐      ┌────────────┐      ┌──────────┐      ┌────────────┐
│  INPUT  │ ───> │ REPRESENTATION │ ───> │ PROCESSING │ ───> │  OUTPUT  │ ───> │ EVALUATION │
│ Data    │      │ Vektorisasi    │      │ Komputasi  │      │ Label /  │      │ Validasi   │
│ Mentah  │      │ Numerik        │      │ Model / IF │      │ Tindakan │      │ Metrik     │
└─────────┘      └────────────────┘      └────────────┘      └──────────┘      └────────────┘
```

1. **Input (Masukan Data Mentah)**:
   Data yang dikumpulkan dari dunia nyata: teks ulasan pengguna, berkas audio suara mikrofon, matriks sensor IoT suhu dan getaran, atau gambar citra kamera.
2. **Representation (Representasi Fitur Numerik)**:
   Komputer tidak memahami teks bahasa Indonesia atau warna secara intuitif; komputer hanya dapat memproses angka. Data mentah harus diubah menjadi representasi matematis (*feature vector*, *bag-of-words*, *TF-IDF*, atau *vector embeddings*). Kualitas representasi menentukan batas performa sistem.
3. **Processing (Pemrosesan Logika / Model)**:
   Eksekusi inferensi pada representasi numerik. Pada sistem berbasis aturan, ini adalah pencocokan ekspresi Boolean; pada model ML/DL, ini adalah serangkaian perkalian matriks bobot dan fungsi aktivasi non-linier.
4. **Output (Keluaran & Tindakan)**:
   Hasil inferensi yang bermakna bagi pengguna: probabilitas kategori, nilai estimasi harga, bounding box deteksi objek, teks respon jawaban, atau perintah gerak aktuator motor.
5. **Evaluation (Evaluasi Kinerja Sistem)**:
   Pengukuran sistematis terhadap keandalan output menggunakan metrik objektif (seperti Akurasi, Presisi, Recall, F1-Score, Mean Squared Error, atau *Human-in-the-loop review*).

---

## 9. Contoh Penerapan Sehari-hari

Mahasiswa berinteraksi dengan lusinan sistem AI setiap hari tanpa selalu menyadarinya:

1. **Klasifikasi Pesan & Filter Spam**: Memproses teks email yang masuk, memetakan frekuensi kata kunci dan kredibilitas pengirim, lalu memutuskan apakah pesan harus masuk ke kotak masuk utama atau folder spam.
2. **Sistem Rekomendasi Konten**: Platform media streaming dan e-commerce memodelkan riwayat tontonan atau pembelian sebelumnya guna merekomendasikan video atau produk yang relevan menggunakan teknik *collaborative filtering* atau *deep ranking*.
3. **Prediksi Kebutuhan Energi Listrik**: Fasilitas pembangkit menggunakan model regresi deret waktu (*time-series forecasting*) berbasis input historis suhu cuaca, hari kerja/libur, dan tren beban untuk memprediksi pasokan daya jam demi jam.
4. **Analisis Data Sensor Industri / Pertanian**: Perangkat sensor IoT di lahan pertanian membaca kelembaban tanah dan suhu udara; model klasifikasi mendeteksi dini tanda-tanda stres tanaman sebelum daun menguning.
5. **Computer Vision pada Kendaraan Modern**: Kamera mobil mendeteksi marka jalan, pejalan kaki, dan kendaraan sekitar secara *real-time* untuk memberikan peringatan tabrakan dini (*Lane Departure Warning*).
6. **Asisten Produktivitas Berbasis LLM**: Merangkum makalah ilmiah panjang menjadi poin-poin eksekutif atau menerjemahkan draf fungsi kode antar bahasa pemrograman.

---

## 10. Kapan AI Layak Digunakan dan Kapan Solusi Sederhana Lebih Tepat?

Salah satu tanda kedewasaan seorang insinyur kecerdasan buatan adalah **mengetahui kapan TIDAK menggunakan AI**. 

Menerapkan Machine Learning atau Deep Learning membawa konsekuensi biaya mahal: kebutuhan data berlabel, infrastruktur komputasi pelatihan, risiko *model drift*, dan ketidakmampuan menjamin kepastian 100%.

### Panduan Alur Keputusan Rekayasa (Decision Framework)

```
                              [ MASALAH BARU ]
                                     │
                 Apakah ada aturan/rumus pasti 100%?
                                 /       \
                              (YA)       (TIDAK)
                               /           \
                 Gunakan Solusi              Apakah ada data historis yang
                 Deterministik (If/SQL)      cukup & representatif?
                                             /                    \
                                          (YA)                   (TIDAK)
                                           /                        \
                    Apakah aturan terlalu rumit         Kumpulkan data dulu atau
                    untuk ditulis manual?               gunakan Heuristik Sederhana
                         /               \
                       (YA)              (TIDAK)
                        /                  \
                Pertimbangkan ML / DL      Gunakan Rule-Based System
```

### Kapan Solusi Sederhana (Deterministik) Lebih Tepat?
- Perhitungan matematika pasti (kalkulator, perhitungan diskon kupon belanja).
- Validasi form (panjang password minimal 8 karakter, format email harus memiliki simbol `@`).
- Alur kerja bisnis yang tunduk pada kepatuhan regulasi hukum mutlak di mana setiap deviasi 0.01% tidak dapat ditoleransi.

### Kapan AI / Machine Learning Benar-benar Dibutuhkan?
- Tugas berbasis persepsi manusia (mengenali ucapan audio, visi komputer).
- Masalah dengan aturan yang terlalu banyak dan dinamis (rekomendasi musik dari jutaan lagu).
- Pola yang terus berevolusi secara bermusuhan (*adversarial*), seperti deteksi serangan siber dan spam.

---

## 11. Keterbatasan Fundamental Sistem AI

Agar tidak terjebak dalam ekspektasi semu, setiap praktisi AI wajib memahami lima titik kerapuhan sistem cerdas:

1. **Kualitas Data (*Garbage In, Garbage Out*)**:
   Model pembelajaran mesin tidak dapat membedakan fakta sejati dari kesalahan input. Jika dataset pelatihan mengandung label yang salah, data usang, atau duplikasi ekstrem, model akan mempelajari kekeliruan tersebut sebagai pola kebenaran.
2. **Kerapuhan Generalisasi (*Out-of-Distribution Failure*)**:
   Model AI bekerja sangat baik pada data yang memiliki distribusi statistik mirip dengan data pelatihannya. Namun, ketika lingkungan nyata berubah secara drastis (*distribution shift*—misalnya mobil otonom yang dilatih di jalanan beraspal kering diuji di daerah berlumpur lebat), kinerja model dapat anjlok tanpa peringatan.
3. **Bias Algoritmik dan Diskriminasi Historis**:
   Dataset pelatihan mencerminkan kondisi masa lalu manusia. Jika data penerimaan kerja historis didominasi kelompok demografi tertentu, model AI akan belajar memberi skor lebih rendah kepada kelompok minoritas, memperkuat ketidakadilan sosial secara terselubung.
4. **Ketidakpastian dan Ketiadaan Penjelasan (*Black-Box Problem*)**:
   Banyak arsitektur Deep Learning modern beroperasi dengan puluhan juta parameter bobot. Ketika model menolak permohonan pinjaman seseorang atau salah mendiagnosis tumor, sulit untuk menjelaskan alasan matematis spesifik di balik keputusan tersebut kepada korban.
5. **Halusinasi pada Model Generatif**:
   LLM tidak memiliki modul verifikasi fakta internal (*ground truth checker*). Model dirancang untuk menghasilkan untaian kata yang koheren secara statistik, bukan yang benar secara faktual. Akibatnya, model dapat dengan fasih mengutip pasal undang-undang yang tidak pernah ada di dunia nyata.

---

## 12. Peran Manusia dan Pengantar Responsible AI

Kecerdasan Buatan bukan pengganti akal budi dan tanggung jawab manusia, melainkan alat bantu kognitif (*cognitive amplifier*). Oleh karena itu, perancangan sistem AI modern wajib mengintegrasikan kerangka kerja **Responsible AI** (merujuk pada rekomendasi UNESCO dan NIST AI Risk Management Framework):

1. **Fairness (Keadilan)**: Memastikan sistem tidak menghasilkan disparitas dampak negatif terhadap kelompok rentan atau latar belakang tertentu.
2. **Transparency & Explainability (Transparansi & Kejelasan)**: Pengguna berhak mengetahui kapan mereka berinteraksi dengan AI dan apa pertimbangan utama di balik keputusan otomatis yang memengaruhi hidup mereka.
3. **Privacy & Data Governance (Privasi & Perlindungan Data)**: Menjaga kerahasiaan data pengguna, menerapkan prinsip minimisasi data, dan tidak menggunakan data pribadi tanpa persetujuan eksplisit.
4. **Accountability (Akuntabilitas)**: Sistem perangkat lunak tidak dapat dimintai pertanggungjawaban hukum; organisasi dan insinyur pengembang lah yang bertanggung jawab penuh atas dampak sistem yang mereka publikasikan.
5. **Human-in-the-Loop (Keterlibatan Manusia)**: Pada aplikasi berdampak tinggi (*high-stakes decisions* seperti peradilan pidana, diagnostik bedah kritis, atau seleksi beasiswa utama), AI hanya boleh bertindak sebagai perekomendasi keputusan (*decision-support system*), dengan keputusan final tetap berada di tangan manusia yang kompeten.

---

## 13. Ringkasan Materi

- **Artificial Intelligence** adalah bidang komputasi luas yang mencakup pendekatan berbasis aturan simbolik (*Rule-Based*) dan pendekatan berbasis pembelajaran data (*Machine Learning*).
- **Deep Learning** adalah spesialisasi ML yang menggunakan Neural Network banyak lapis untuk mengotomasi representasi fitur dari data mentah berdimensi tinggi.
- **Generative AI & LLM** memprediksi dan menyintesis data baru berdasarkan probabilitas statistik, bukan pemahaman sadar.
- Siklus hidup sistem AI berbasis model terbagi menjadi fase **Training** (pembentukan bobot parameter) dan **Inference** (penggunaan model untuk prediksi).
- **Akurasi saja tidak cukup**: selalu periksa kesalahan tipe I (*False Positive*) dan tipe II (*False Negative*).

---

## 14. Latihan Terstruktur

Kerjakan latihan berikut secara mandiri untuk menguji pemahaman konseptual dan analitis Anda.

### Bagian A: Tingkat Dasar (Identifikasi Konsep)

**Soal 1: AI vs Otomasi Deterministik Non-AI**
Tentukan apakah masing-masing kasus berikut tergolong **Sistem AI** atau **Otomasi Non-AI**:
1. Script Python yang menghitung total belanjaan setelah diskon 10% dan PPN 11%.
2. Kamera keamanan bandara yang mendeteksi koper tak bertuan selama 15 menit menggunakan model pengenalan visual.
3. Query basis data PostgreSQL yang mengurutkan 10.000 mahasiswa berdasarkan IPK tertinggi secara menurun.
4. Filter email yang secara otomatis memindahkan email promosi berdasarkan kata dan reputasi pengirim yang adaptif.
5. Pencarian kata eksak (*exact match*) pada dokumen PDF menggunakan fungsi `Ctrl+F`.

**Soal 2: Taksonomi Pendekatan AI**
Pasangkan setiap deskripsi berikut dengan istilah: **Rule-Based System**, **Machine Learning Klasik**, **Deep Learning**, atau **Generative AI**:
- (a) Model memprediksi harga sewa rumah dari fitur luas tanah, jumlah kamar, dan jarak ke stasiun menggunakan Linear Regression.
- (b) Sistem kelayakan kredit perbankan menggunakan aturan `IF gaji > 10jt AND riwayat == 'LANCAR' THEN 'SETUJU'`.
- (c) Arsitektur CNN 50-layer mendeteksi retakan mikro pada sayap pesawat langsung dari foto resolusi tinggi.
- (d) Model Transformer menghasilkan paragraf ringkasan eksekutif dari transkrip rekaman rapat secara otomatis.

### Bagian B: Tingkat Menengah (Anatomi & Evaluasi Metrik)

**Soal 3: Anatomi Sistem Cerdas**
Rumah sakit merancang sistem deteksi aritmia jantung dari rekaman elektrokardiogram (EKG) 30 detik. Uraikan lima komponennya:
1. **Input**: Apakah bentuk fisik data mentah yang diterima?
2. **Representation**: Bagaimana sinyal analog mentah diubah menjadi representasi numerik?
3. **Processing**: Apa peran algoritma pemodelan dalam sistem ini?
4. **Output**: Apakah keluaran spesifik yang disajikan kepada dokter jaga?
5. **Evaluation**: Bagaimana cara mengukur keandalan sistem sebelum diuji pada pasien nyata?

**Soal 4: Kalkulasi Metrik Evaluasi & Analisis Kesalahan**
Model deteksi fraud diuji pada 100 transaksi (90 transaksi normal riil, 10 transaksi fraud riil):
- Dari 10 fraud riil: 7 terdeteksi benar sebagai fraud, 3 lolos dianggap normal.
- Dari 90 normal riil: 85 diprediksi normal benar, 5 keliru diblokir sebagai fraud.
1. Hitung nilai True Positive (TP), False Positive (FP), True Negative (TN), dan False Negative (FN)!
2. Hitung nilai Akurasi keseluruhan model tersebut!
3. Jelaskan dampak bisnis nyata dari kesalahan False Positive dan False Negative! Mana yang lebih fatal dalam perbankan?

### Bagian C: Tingkat Lanjut (Analisis Kelayakan & Responsible AI)

**Soal 5: Analisis Kelayakan Masalah (AI vs Non-AI)**
Perpustakaan ingin mengotomatiskan dua kebutuhan:
1. Jam operasional dan syarat pendaftaran anggota baru.
2. Rekomendasi buku bagi pembaca yang menyukai novel misteri bertema sejarah Indonesia.
- Untuk kebutuhan (1), pendekatan mana yang tepat: *Rule-Based FAQ* atau *LLM Generative AI*? Apa risikonya jika memaksakan LLM?
- Untuk kebutuhan (2), mengapa pencarian kata kunci eksak sering mengecewakan dan bagaimana Machine Learning membantu?

**Soal 6: Evaluasi Risiko & Responsible AI**
Sistem AI memprediksi mahasiswa yang berisiko tidak lulus tepat waktu berdasarkan nilai ujian, IPK semester, daerah asal sekolah, dan pekerjaan orang tua.
1. **Fairness**: Identifikasi potensi bias jika latar belakang sosial ekonomi (pekerjaan orang tua / daerah asal) dimasukkan ke model!
2. **Transparency**: Mengapa mahasiswa berhak menerima penjelasan (*explainability*), bukan hanya skor angka probabilitas?
3. **Human-in-the-Loop**: Rancang mekanisme keterlibatan dosen pembimbing akademik dalam menindaklanjuti prediksi ini!

---

## 15. Pembahasan dan Panduan Jawaban Latihan

### Pembahasan Soal 1
1. **Non-AI**: Rumus diskon dan pajak adalah formula deterministik statis tanpa inferensi pola data.
2. **Sistem AI**: Menggunakan Computer Vision untuk membedakan objek koper dari lingkungan dinamis.
3. **Non-AI**: Operasi pengurutan basis data adalah algoritma sorting klasik (misal QuickSort/B-Tree) dengan kriteria pasti.
4. **Sistem AI**: Klasifikasi teks adaptif yang memperbarui batas keputusan dari data email baru.
5. **Non-AI**: Pencocokan string eksak deterministik (seperti Boyer-Moore atau KMP).

### Pembahasan Soal 2
- (a) **Machine Learning Klasik** (Regresi Linier mempelajari bobot numerik dari fitur tabel terstruktur).
- (b) **Rule-Based System** (Logika diskrit IF-THEN eksplisit yang disusun langsung oleh pakar kredit).
- (c) **Deep Learning** (CNN multi-layer mengekstrak representasi visual langsung dari piksel mentah tanpa rekayasa fitur manual).
- (d) **Generative AI** (Model Transformer menyintesis teks baru melalui pemodelan probabilitas token bahasa).

### Pembahasan Soal 3
1. **Input**: Sinyal voltase analog atau deret waktu digital rekaman elektroda EKG per milidetik.
2. **Representation**: Ekstraksi fitur gelombang (interval gelombang P-Q-R-S-T, frekuensi detak via Fourier Transform) atau spektrogram 2D sinyal.
3. **Processing**: Model klasifikasi menghitung probabilitas ritme abnormal (aritmia).
4. **Output**: Peringatan visual pada layar monitor (misal: "Terdeteksi Indikasi Fibrilasi Atrium, Probabilitas 91%") beserta penanda segmen waktu EKG yang anomali.
5. **Evaluation**: Pengujian pada dataset uji independen yang telah dianotasi oleh dokter spesialis jantung, menggunakan metrik Sensitivitas (Recall) dan Spesifisitas.

### Pembahasan Soal 4
1. **Kuadran Confusion Matrix**:
   - $\text{TP} = 7$ (Penipuan terdeteksi benar)
   - $\text{FN} = 3$ (Penipuan lolos tidak terdeteksi)
   - $\text{FP} = 5$ (Transaksi normal salah diblokir)
   - $\text{TN} = 85$ (Transaksi normal lolos benar)
2. **Kalkulasi Akurasi**:
   $$\text{Akurasi} = \frac{\text{TP} + \text{TN}}{\text{Total Sampel}} = \frac{7 + 85}{100} = \frac{92}{100} = 92.0\%$$
3. **Analisis Dampak Bisnis**:
   - **False Positive (5 kasus)**: Nasabah jujur yang sedang bertransaksi kartunya terblokir keliru (menimbulkan komplain dan biaya verifikasi CS).
   - **False Negative (3 kasus)**: Penipuan nyata lolos membobol uang nasabah (kerugian finansial riil dan risiko hukum).
   - **Kesimpulan**: Dalam perbankan, **False Negative jauh lebih fatal** daripada False Positive. Model dengan akurasi 92% ini masih sangat berbahaya karena membiarkan 30% penipuan lolos (3 dari 10)!

### Pembahasan Soal 5
- **Kebutuhan (1)**: Paling tepat menggunakan **Rule-Based FAQ terstruktur**. Jam operasional dan syarat pendaftaran adalah fakta pasti. Memaksakan LLM tanpa *grounding* ketat berisiko memicu halusinasi jam buka palsu kepada masyarakat.
- **Kebutuhan (2)**: Exact matching gagal karena pembaca bisa mencari dengan frasa bervariasi (misal "cerita detektif zaman perjuangan kemerdekaan"). Machine Learning dengan *vector embeddings* mampu menangkap kemiripan makna semantik.

### Pembahasan Soal 6
1. **Fairness & Bias**: Memasukkan status ekonomi atau daerah asal membuat sistem menghukum mahasiswa dari daerah tertinggal semata-mata karena data masa lalu mencatat fasilitas lama yang minim. Ini melanggengkan diskriminasi historis. Fitur harus difokuskan pada aktivitas akademik objektif saat ini.
2. **Transparency**: Mahasiswa berhak mengetahui faktor konkret apa yang memicu label risiko (misal: "Kehadiran praktikum < 70% dan nilai kuis rendah") agar mereka tahu tindakan perbaikan apa yang harus diambil.
3. **Human-in-the-Loop**: Sistem AI hanya memberikan notifikasi peringatan dini kepada Dosen Wali. Dosen Wali kemudian melakukan konseling personal untuk menggali akar masalah riil sebelum mengambil tindakan pembinaan akademik.

---

## 16. Tugas Mini Project: Analisis Kelayakan Solusi AI

### Deskripsi Tugas
Sebagai calon perekayasa kecerdasan buatan, keterampilan mendasar sebelum menulis kode adalah **analisis kelayakan sistem** (*problem framing & feasibility study*). Pilih satu studi kasus permasalahan nyata di lingkungan sekitar Anda (kampus, layanan publik, transportasi, pertanian, atau UMKM), lalu susun **Laporan Analisis Kelayakan Solusi AI**.

> **Catatan Pelaksanaan:**
> Pada Chapter 01, tugas ini **TIDAK mewajibkan Anda melatih model Machine Learning atau menulis kode aplikasi penuh**. Fokus utama tugas adalah ketajaman perumusan masalah, penalaran pemilihan pendekatan, desain arsitektur data/evaluasi, serta analisis risiko sistem cerdas secara mendalam dan terukur.

### Sembilan Deliverables Laporan
1. **Deskripsi Masalah**: Persoalan nyata yang ingin diatasi, urgensinya, dan dampak negatif jika dibiarkan.
2. **Pengguna dan Konteks Operasional**: Siapa pengguna sistem, di mana sistem dijalankan (web, mobile, edge device), dan kendala lingkungan operasinya.
3. **Solusi Non-AI sebagai Pembanding (Baseline)**: Cara menyelesaikan masalah tersebut saat ini tanpa AI (misal aturan manual, form kertas, query database sederhana) beserta kelemahannya.
4. **Pendekatan AI yang Dipertimbangkan**: Perbandingan kesesuaian antara Rule-Based, ML Klasik, Deep Learning, atau Generative AI, beserta justifikasi pendekatan terpilih.
5. **Spesifikasi Input dan Output**: Format data mentah, representasi fitur numerik, dan keluaran yang disajikan ke pengguna.
6. **Kebutuhan Data dan Tata Kelola**: Sumber data (*provenance*), estimasi volume, protokol anotasi/labeling, serta privasi dan lisensi.
7. **Protokol Evaluasi dan Metrik Keberhasilan**: Metrik kuantitatif terukur (Accuracy, Precision, Recall, MAE) dan alasan pemilihannya.
8. **Analisis Risiko dan Strategi Mitigasi**: Dampak False Positive vs False Negative, potensi bias historis data, dan desain mekanisme *Human-in-the-Loop*.
9. **Keterbatasan Sistem**: Kondisi di mana sistem diprediksi gagal (*failure modes*) dan kapan pengguna dilarang bergantung buta pada sistem.

---

## 17. Rubrik Penilaian Mini Project (Total 100 Poin)

| Kriteria Penilaian | Bobot | Nilai Penuh (Full Marks) | Nilai Parsial (Partial Marks) | Tidak Terpenuhi (Unsatisfactory) |
|---|---|---|---|---|
| **1. Perumusan Masalah & Konteks** | **20 poin** | Masalah didefinisikan secara spesifik, berbasis fakta nyata, konteks pengguna jelas, dan perbandingan solusi non-AI (*baseline*) diuraikan secara tajam dengan kelemahannya. | Masalah dipaparkan umum atau kurang kontekstual; solusi non-AI disebutkan tetapi kurang mendalam perbandingannya. | Masalah kabur, artifisial, atau solusi non-AI tidak dibahas sama sekali. |
| **2. Ketepatan Konsep & Pemilihan Pendekatan** | **25 poin** | Membedakan secara akurat antara Rule-Based, ML, DL, dan GenAI; memberikan justifikasi rekayasa yang kokoh mengapa pendekatan AI terpilih memang dibutuhkan dan tidak berlebihan (*over-engineering*). | Memilih pendekatan AI yang masuk akal namun pembedaan paradigma kurang mendalam atau justifikasi kebutuhan AI masih lemah. | Salah mengidentifikasi konsep AI/ML, atau memaksakan penggunaan AI yang tidak masuk akal untuk masalah yang sangat sederhana. |
| **3. Desain Input, Output & Evaluasi** | **20 poin** | Input, representasi fitur, dan output didefinisikan secara konkret; metrik evaluasi dipilih secara tepat dengan mempertimbangkan trade-off False Positive vs False Negative. | Input/output didefinisikan tetapi representasi teknis kurang jelas; metrik evaluasi hanya menyebutkan "akurasi" tanpa analisis trade-off error. | Desain alur data tidak jelas dan metrik evaluasi tidak diuraikan atau tidak relevan dengan masalah. |
| **4. Analisis Risiko & Keterbatasan** | **20 poin** | Menelaah secara kritis dampak kegagalan sistem, bias data, privasi pengguna, mekanisme Human-in-the-Loop yang konkret, serta batasan kondisi operasional sistem. | Mengidentifikasi beberapa risiko umum (misal sekadar menyebut 'bisa bias') tanpa strategi mitigasi nyata atau mekanisme manusia yang jelas. | Mengabaikan analisis risiko, menganggap sistem selalu benar, atau tidak menyebutkan keterbatasan sistem. |
| **5. Kejelasan Komunikasi & Referensi** | **15 poin** | Laporan terstruktur rapi, bahasa Indonesia formal yang alami, istilah teknis bahasa Inggris digunakan secara konsisten dan tepat, menyertakan minimal 2 referensi primer yang kredibel. | Struktur laporan cukup baik; ada beberapa ketidakkonsistenan istilah teknis; referensi kurang lengkap. | Laporan berantakan, tidak terbaca, plagiarisme, atau tanpa referensi rujukan ilmiah. |

---

## 18. Referensi Primer & Landasan Kurikulum

1. **ACM/IEEE-CS/AAAI Computer Science Curricula 2023 (CS2023)**:
   - Area: *Artificial Intelligence (AI) Knowledge Area*.
   - Rujukan: [https://csed.acm.org/](https://csed.acm.org/)
2. **UNESCO AI Competency Framework for Students (2024)**:
   - Rujukan: [https://www.unesco.org/en/articles/ai-competency-framework-students](https://www.unesco.org/en/articles/ai-competency-framework-students)
3. **NIST AI Risk Management Framework (AI RMF 1.0)**:
   - Rujukan: [https://www.nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)
4. **Google Machine Learning Crash Course**:
   - Rujukan: [https://developers.google.com/machine-learning/crash-course/](https://developers.google.com/machine-learning/crash-course/)
