# Panduan Praktik Laboratorium: Natural Language Processing Fundamentals

Selamat datang di modul panduan praktik laboratorium **Natural Language Processing Fundamentals** (Bab 27). Pada laboratorium ini, Anda akan mempraktikkan fondasi pemrosesan teks alami, representasi fitur statistik dan vektor semantik, serta arsitektur jaringan saraf rekuren sekuensial berbasis **Python**, **NumPy**, **Scikit-learn**, dan **PyTorch** pada jalur komputasi CPU: mulai dari pra-pemrosesan teks adaptif (pelestarian kata negasi), tokenisasi regex dan manajemen kosakata (token `<pad>`, `<unk>`, `<bos>`, `<eos>`), pemodelan Bag-of-Words dan derivasi analitis TF-IDF, penangkapan konteks lokal N-gram, klasifikasi teks dengan isolasi partisi 3-way bebas kebocoran data (*zero data leakage*), representasi vektor padat (*Word Embeddings*) dan geometri *Cosine Similarity*, hingga implementasi lintasan maju Simple RNN, arsitektur gerbang LSTM dan GRU, serta benchmarking komparatif performa sekuensial pada perangkat CPU.

---

## 1. Lingkungan Komputasi & Persyaratan Eksekusi

### Kebutuhan Perangkat Lunak:
- Python: Versi 3.10+ (Direkomendasikan Python 3.12).
- Pustaka Inti: `torch>=2.0.0` (Jalur CPU terverifikasi pada PyTorch 2.15.0+cpu), `scikit-learn>=1.3.0` (teruji 1.9.1), `numpy>=2.0.0` (teruji 2.5.4).
- Akselerator Perangkat Keras: Bersifat opsional. Seluruh 12 skrip pengujian, model klasifikasi teks Scikit-learn, operasi embedding lookup, dan perbandingan sekuensial RNN/LSTM/GRU dirancang untuk dieksekusi secara instan pada CPU komputer standar dalam hitungan detik tanpa mewajibkan GPU.

### Struktur Repositori Praktik Bab 27:
```
content/level-3-deep-learning/chapter-27/
├── code/
│   ├── README.md
│   ├── text_preprocessing.py
│   ├── tokenization.py
│   ├── bag_of_words.py
│   ├── tfidf_representation.py
│   ├── ngram_features.py
│   ├── text_classification.py
│   ├── word_embeddings.py
│   ├── cosine_similarity.py
│   ├── simple_rnn.py
│   ├── lstm_sequence_model.py
│   ├── gru_sequence_model.py
│   └── rnn_lstm_gru_comparison.py
├── experiments/
│   └── results_ch27.json
├── notebooks/
│   ├── generate_notebook.py
│   └── chapter_27_companion.ipynb
├── slides/
│   ├── build_slides.py
│   └── deck.json
├── index.md
├── practice.md
├── metadata.json
└── audit.md
```

---

## 2. Panduan Langkah-demi-Langkah Praktik Laboratorium

### Langkah 1: Pra-Pemrosesan Teks & Pelestarian Negasi
Jalankan skrip pra-pemrosesan teks untuk memahami pembersihan teks dan mitigasi pembalikan sentimen:
```bash
python content/level-3-deep-learning/chapter-27/code/text_preprocessing.py
```
- **Fokus Pengamatan**:
  - Amati bagaimana URL, mention, tanda baca berlebih, dan karakter berulang dibersihkan menggunakan regex.
  - Perhatikan pemetaan kata singkatan dan bahasa gaul (*slang normalization*) seperti 'gk' $\rightarrow$ 'tidak' dan 'bgt' $\rightarrow$ 'sangat'.
  - Uji kasus kritis: pastikan kata negasi ('tidak', 'bukan', 'jangan') **tidak terhapus** oleh filter stopword, sehingga frasa *'tidak enak'* tidak tereduksi menjadi *'enak'*.

### Langkah 2: Tokenisasi, Kosakata & Penanganan OOV
Pelajari proses pemecahan kalimat ke dalam token diskret dan penyeragaman panjang sekuens:
```bash
python content/level-3-deep-learning/chapter-27/code/tokenization.py
```
- **Fokus Pengamatan**:
  - Pahami peran token-token khusus: `<pad>` (indeks 0), `<unk>` (indeks 1), `<bos>` (indeks 2), dan `<eos>` (indeks 3).
  - Amati proses pemetaan token kata ke indeks integer (`token2id`) dan rekonstruksi balik (`id2token`).
  - Perhatikan bagaimana kata baru di luar kosakata data latih (*Out-of-Vocabulary* / OOV) secara otomatis dipetakan ke token `<unk>`.
  - Uji pemotongan (*truncation*) dan penambahan bantalan (*padding*) agar sekuens memiliki panjang tepat `max_len`.

### Langkah 3: Representasi Bag-of-Words (BoW) & Analisis Sparsity
Bandingkan implementasi BoW dari dasar terhadap Scikit-learn CountVectorizer:
```bash
python content/level-3-deep-learning/chapter-27/code/bag_of_words.py
```
- **Fokus Pengamatan**:
  - Amati pembentukan matriks frekuensi istilah dokumen $\times$ kosakata $(N \times V)$.
  - Hitung persentase kelangkaan matriks (*matrix sparsity*): perhatikan tingginya proporsi elemen bernilai nol.
  - Buktikan batasan ketiadaan urutan (*order invariance*): kalimat *'Saya tidak benci kamu'* dan *'Kamu tidak benci saya'* menghasilkan vektor BoW yang persis identik.

### Langkah 4: Derivasi & Perhitungan Analitis TF-IDF
Hitung bobot TF-IDF langkah-demi-langkah dan verifikasi terhadap Scikit-learn:
```bash
python content/level-3-deep-learning/chapter-27/code/tfidf_representation.py
```
- **Fokus Pengamatan**:
  - Telusuri perhitungan Document Frequency (DF) dan nilai Inverse Document Frequency (IDF) dengan formula terhalus (*smoothed IDF*): $\text{idf}(t) = \ln((1+N)/(1+\text{df}(t))) + 1$.
  - Perhatikan bagaimana kata yang sangat umum (DF tinggi) memperoleh bobot IDF rendah, sementara kata spesifik (DF rendah) memperoleh bobot tinggi.
  - Buktikan bahwa hasil perhitungan manual setelah normalisasi L2 memiliki deviasi $0.00 \times 10^0$ (identik sempurna) terhadap `TfidfVectorizer` Scikit-learn.

### Langkah 5: Penangkapan Konteks Frasa via Fitur N-gram
Eksplorasi ekstraksi n-gram untuk mengatasi kelemahan model unigram kata tunggal:
```bash
python content/level-3-deep-learning/chapter-27/code/ngram_features.py
```
- **Fokus Pengamatan**:
  - Amati ekstraksi Unigram ($n=1$), Bigram ($n=2$), dan Trigram ($n=3$) menggunakan sliding window.
  - Buktikan bagaimana bigram `tidak_ramah` memisahkan sentimen negatif secara mandiri dibandingkan unigram `ramah`.
  - Amati fenomena ledakan kosakata (*vocabulary explosion*) saat rentang n-gram diperluas ke trigram.

### Langkah 6: Pipeline Klasifikasi Teks Terisolasi Bebas Kebocoran Data
Bangun dan evaluasi model klasifikasi ulasan sentimen pelanggan secara menyeluruh:
```bash
python content/level-3-deep-learning/chapter-27/code/text_classification.py
```
- **Fokus Pengamatan**:
  - Pahami partisi data terstratifikasi 3-way: Train (70%), Validation (15%), dan Test (15%).
  - Pastikan bahwa `TfidfVectorizer` di-fit **HANYA pada data training**, sedangkan data validasi dan uji hanya melalui operasi `transform()` untuk mencegah kebocoran data (*zero data leakage*).
  - Bandingkan performa Multinomial Naive Bayes dan Logistic Regression terhadap Dummy Baseline.
  - Analisis kesalahan prediksi (*error analysis*) pada matriks konfusi sampel test set.

### Langkah 7: Word Embeddings & Operasi Dense Lookup
Pelajari representasi vektor padat terdistribusi dan operasi lookup memori:
```bash
python content/level-3-deep-learning/chapter-27/code/word_embeddings.py
```
- **Fokus Pengamatan**:
  - Bandingkan representasi sparse one-hot ($V$ dimensi) terhadap dense word embedding ($D$ dimensi).
  - Amati struktur tabel bobot matriks embedding $E \in \mathbb{R}^{V \times D}$.
  - Buktikan bahwa pemilihan baris `E[indices]` pada NumPy menghasilkan nilai yang identik sempurna dengan `torch.nn.Embedding`.
  - Verifikasi bahwa vektor pada token `<pad>` (`padding_idx=0`) selalu bernilai nol pada seluruh dimensi.

### Langkah 8: Geometri Ruang Vektor & Cosine Similarity
Hitung derajat kemiripan sudut semantik antar-vektor kata:
```bash
python content/level-3-deep-learning/chapter-27/code/cosine_similarity.py
```
- **Fokus Pengamatan**:
  - Buktikan sifat geometris: vektor paralel searah menghasilkan kesamaan kosinus $+1.0$, vektor tegak lurus (ortogonal) menghasilkan $0.0$, dan berlawanan arah menghasilkan $-1.0$.
  - Telusuri pengelompokan semantik sederhana: kata dalam domain serupa ('kucing' dan 'anjing') memiliki skor kosinus tinggi.
  - Pahami batasan metodologis: kesamaan kosinus mengukur orientasi koordinat matematis dan tidak boleh diklaim secara sepihak sebagai kebenaran ontologis mutlak.

### Langkah 9: Forward Pass Simple RNN Berbasis Komputasi Rekuren
Uji mekanisme pembaruan hidden state langkah waktu demi waktu pada Simple RNN:
```bash
python content/level-3-deep-learning/chapter-27/code/simple_rnn.py
```
- **Fokus Pengamatan**:
  - Amati loop waktu rekuren: $\mathbf{h}_t = \tanh(\mathbf{x}_t \mathbf{W}_{xh}^T + \mathbf{h}_{t-1} \mathbf{W}_{hh}^T + \mathbf{b})$.
  - Perhatikan bagaimana hidden state $\mathbf{h}_{t-1}$ berfungsi sebagai memori konteks langkah waktu sebelumnya.
  - Buktikan bahwa implementasi NumPy mandiri menghasilkan tensor luaran seluruh waktu dan final hidden state yang identik (< 1e-6) terhadap PyTorch `nn.RNN`.

### Langkah 10: Arsitektur Gated LSTM & Jalur Bebas Hambatan Cell State
Pelajari mekanisme gerbang dan penghitungan parameter analitis pada unit sel LSTM:
```bash
python content/level-3-deep-learning/chapter-27/code/lstm_sequence_model.py
```
- **Fokus Pengamatan**:
  - Pahami peran Cell State ($c_t$) linier dalam mempertahankan informasi jangka panjang dan memitigasi vanishing gradients.
  - Amati fungsi tiga gerbang: Forget Gate ($f_t$), Input Gate ($i_t$), dan Output Gate ($o_t$).
  - Buktikan kesesuaian formula parameter analitis $4 \times (D \cdot H + H^2 + 2H)$ terhadap jumlah bobot aktual pada PyTorch `nn.LSTM`.

### Langkah 11: Arsitektur Gated Recurrent Unit (GRU) & Efisiensi Parameter
Analisis penyederhanaan gerbang pada GRU:
```bash
python content/level-3-deep-learning/chapter-27/code/gru_sequence_model.py
```
- **Fokus Pengamatan**:
  - Pelajari penggabungan cell state dan hidden state serta peran Update Gate ($z_t$) dan Reset Gate ($r_t$).
  - Buktikan secara komputasional bahwa lapisan GRU menghemat tepat $25.0\%$ parameter gerbang ($3/4$) dibandingkan LSTM pada dimensi seimbang.
  - Amati bentuk tensor input dan output pada forward pass klasifikasi sekuens.

### Langkah 12: Benchmarking Komparatif Sekuensial (Simple RNN vs LSTM vs GRU)
Jalankan pengujian empiris komparasi ketiga keluarga model sekuensial pada CPU:
```bash
python content/level-3-deep-learning/chapter-27/code/rnn_lstm_gru_comparison.py
```
- **Fokus Pengamatan**:
  - Amati perbandingan jumlah parameter: Simple RNN (1.698) < GRU (3.714) < LSTM (4.722).
  - Telaah waktu pelatihan (ms) per epoch pada perangkat CPU standar.
  - Bandingkan akurasi validasi akhir: LSTM mencapai performa tertinggi berkat jalur cell state yang stabil.
  - Periksa berkas log hasil eksperimen yang tersimpan di `experiments/results_ch27.json`.

---

## 3. Latihan Laboratorium Mandiri (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. **Perhitungan Matriks Bag-of-Words**:
   Diberikan korpus mini dengan kosakata terurut: `['anjing', 'berlari', 'kucing', 'tidur']`. Tentukan vektor representasi Bag-of-Words untuk dokumen: *"kucing tidur dan kucing berlari"*!
   - *Jawaban*:
     - Frekuensi kemunculan:
       - 'anjing': 0 kali
       - 'berlari': 1 kali
       - 'kucing': 2 kali
       - 'tidur': 1 kali
     - Vektor BoW $[0, 1, 2, 1]$.

2. **Perhitungan IDF Analitis**:
   Dalam sebuah korpus berisi $N = 99$ dokumen, istilah *'kecerdasan'* muncul dalam $\text{DF} = 9$ dokumen. Berapakah nilai IDF terhalus (*smoothed IDF*) istilah tersebut menggunakan formula Scikit-learn $\text{IDF}(t) = \ln \left( \frac{1+N}{1+\text{DF}(t)} \right) + 1$?
   - *Jawaban*:
     - $\text{IDF} = \ln \left( \frac{1 + 99}{1 + 9} \right) + 1 = \ln \left( \frac{100}{10} \right) + 1 = \ln(10) + 1$
     - $\ln(10) \approx 2.3026$
     - $\text{IDF} \approx 2.3026 + 1 = 3.3026$.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Data Leakage pada Pipeline Teks**:
   Seorang mahasiswa mengeksekusi kode klasifikasi teks berikut:
   ```python
   vec = TfidfVectorizer()
   X_all = vec.fit_transform(corpus)  # Fit pada seluruh data
   X_train, X_test, y_train, y_test = train_test_split(X_all, labels)
   clf.fit(X_train, y_train)
   ```
   Jelaskan mengapa kode di atas mengandung kesalahan metodologis fatal (*data leakage*) dan bagaimana memperbaikinya!
   - *Jawaban*:
     - *Penyebab Galat*: Melakukan `fit_transform` pada seluruh korpus sebelum partisi data menyebabkan pembentukan kosakata dan perhitungan Inverse Document Frequency (IDF) terkontaminasi oleh informasi dari `X_test`. Model dilatih menggunakan representasi yang secara tidak sah telah 'melihat' distribusi frekuensi kata pada data uji.
     - *Solusi Perbaikan*: Pisahkan data mentah terlebih dahulu dengan `train_test_split(corpus, labels)`. Terapkan `vec.fit_transform(X_train_raw)` pada data latih, dan gunakan `vec.transform(X_test_raw)` (tanpa `fit`) pada data uji, atau bungkus keduanya dalam `Pipeline([('tfidf', TfidfVectorizer()), ('clf', Classifier())])`.

2. **Pembalikkan Sentimen Akibat Stopwords**:
   Jelaskan mengapa penghapusan stopword secara agresif dapat merusak akurasi model klasifikasi sentimen pada kalimat yang memuat konstruksi negasi ganda atau kontras!
   - *Jawaban*:
     - Daftar stopword standar kerap memuat kata negasi (*tidak, bukan, belum, jangan*) dan kata perangkai kontras (*tetapi, namun, melainkan*). Pada analisis sentimen, kata-kata ini memegang peranan krusial sebagai pengubah kutub makna (*valence shifters*). Menghapus 'tidak' dari kalimat *"Layanan ini tidak ramah"* mereduksi kalimat menjadi *"Layanan ramah"*, memutarbalikkan sentimen negatif menjadi positif secara keliru.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Analisis Matematis Vanishing Gradient pada BPTT**:
   Turunkan mengapa pada Simple RNN, gradien ketergantungan waktu $\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_1}$ rentan lenyap menuju nol (*vanish*) saat panjang sekuens $T$ bertambah besar, dan tunjukkan secara struktural bagaimana penambahan linier pada Cell State LSTM $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ memitigasi masalah tersebut!
   - *Jawaban*:
     - Pada Simple RNN, $\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b})$. Berdasarkan aturan rantai kalkulus:
       $$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_1} = \prod_{j=2}^T \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \prod_{j=2}^T \operatorname{diag}(1 - \mathbf{h}_j^2) \mathbf{W}_{hh}^T$$
       Karena $|1 - \mathbf{h}_j^2| \le 1.0$, jika nilai eigen terbesar matriks bobot $\lambda_{\max}(\mathbf{W}_{hh}) < 1.0$, perkalian berantai $T-1$ matriks akan menyusutkan magnitudo gradien secara eksponensial $O(\lambda^{T-1}) \rightarrow 0$ saat $T \ge 15$.
     - Pada LSTM, turunan terhadap Cell State masa lalu adalah:
       $$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t$$
       Perambatan gradien melalui waktu menjadi:
       $$\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_1} = \prod_{j=2}^T \mathbf{f}_j$$
       Jika Forget Gate mendekati 1 ($\\mathbf{f}_t \approx 1$), gradien dapat mengalir mundur melintasi puluhan hingga ratusan langkah waktu tanpa peluruhan eksponensial, bertindak sebagai *error carousel* bebas hambatan.

---

## 4. Mini Project Chapter 27

### Judul Tugas Proyek:
**"Text Classification and Sequence Modeling on a Small Corpus"**

### Deskripsi Tugas:
Rancang, bangun, dan evaluasi pipeline klasifikasi teks dan pemodelan sekuensial mandiri pada korpus ulasan atau dokumen pendek (sintetis atau dataset publik berlisensi terbuka). Anda diminta untuk:
1. Mendefinisikan masalah klasifikasi teks (misal klasifikasi sentimen ulasan atau deteksi kategori topik) dan mendokumentasikan karakteristik dataset.
2. Mengimplementasikan pra-pemrosesan teks yang tepat dengan penanganan tanda baca, normalisasi singkatan, dan pelestarian kata negasi kritis.
3. Melakukan tokenisasi, membangun kosakata unik terstandar, dan menerapkan padding serta truncation.
4. Membangun model baseline tradisional (TF-IDF + Naive Bayes / Logistic Regression) dengan isolasi partisi 3-way bebas kebocoran data.
5. Membangun model sekuensial berbasis neural (Simple RNN atau LSTM/GRU) menggunakan PyTorch pada CPU.
6. Membandingkan performa model tradisional vs model neural (akurasi, Macro F1, waktu komputasi, dan ukuran parameter).
7. Melakukan analisis galat (*error analysis*) mendalam pada sampel teks yang gagal diprediksi dengan benar.
8. Mendokumentasikan pertimbangan etika NLP (Responsible AI), bias representasi bahasa, dan batas keandalan sistem.

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Dataset and Text Understanding** | 15 Poin | Perumusan masalah klasifikasi teks lugas, dokumentasi korpus dan label kelas transparan, pemahaman struktur data linguistik, serta kepatuhan etika lisensi data. |
| **2. Preprocessing and Representations** | 20 Poin | Pipeline pra-pemrosesan teks bersih dengan pelestarian kata negasi yang terbukti tepat, tokenisasi rapi dengan manajemen token khusus (`<pad>`, `<unk>`), serta perhitungan TF-IDF/BoW yang benar. |
| **3. Baseline and Neural Implementation** | 30 Poin | Implementasi baseline tradisional Scikit-learn bebas kebocoran data (*zero data leakage*), perancangan model sekuensial PyTorch (RNN/LSTM/GRU) modular pada CPU, dan siklus pelatihan yang stabil. |
| **4. Evaluation and Interpretation** | 20 Poin | Evaluasi komparatif komprehensif (Akurasi, Macro F1, Matriks Konfusi) tanpa fabrikasi metrik, diseksi galat prediksi pada kasus sulit, serta pemahaman kritis atas trade-off model statistik vs neural. |
| **5. Documentation and References** | 15 Poin | Kode modular terstruktur bersih (*clean code*), dokumentasi replikasi eksperimen lengkap, refleksi etis Responsible AI (bias data teks), dan pencantuman 5 referensi akademik primer terverifikasi. |

---

## 5. Checklist Kesiapan Penyerahan

- [ ] Seluruh 12 skrip Python di direktori `code/` dapat dieksekusi tanpa galat (`Exit Code: 0`).
- [ ] Pipeline pra-pemrosesan teks terbukti memelihara kata negasi ('tidak', 'bukan') tanpa membalikkan sentimen.
- [ ] Tokenisasi, kamus kata, dan token khusus (`<pad>`, `<unk>`, `<bos>`, `<eos>`) tervalidasi.
- [ ] Nilai perhitungan analitis TF-IDF terbukti cocok identik (< 1e-6) terhadap Scikit-learn.
- [ ] Ekstraksi fitur N-gram mendemonstrasikan penangkapan konteks frasa dan analisis ledakan kosakata.
- [ ] Model klasifikasi teks Scikit-learn menerapkan isolasi partisi 3-way bebas kebocoran data (*zero data leakage*).
- [ ] Lookup tabel Word Embedding terbukti ekuivalen terhadap PyTorch `nn.Embedding`.
- [ ] Sifat geometris Cosine Similarity (+1.0, 0.0, -1.0) dan pencarian tetangga terdekat terverifikasi.
- [ ] Lintasan maju Simple RNN murni NumPy cocok numerik (< 1e-6) terhadap PyTorch `nn.RNN`.
- [ ] Perhitungan analitis jumlah parameter LSTM dan GRU cocok persis terhadap lapisan PyTorch aktual.
- [ ] Benchmarking komparatif RNN vs LSTM vs GRU berhasil dijalankan pada CPU dan log tersimpan di `experiments/results_ch27.json`.
- [ ] Notebook pendamping `chapter_27_companion.ipynb` tervalidasi bersih dalam format v4.

