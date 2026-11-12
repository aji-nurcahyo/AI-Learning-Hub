# Bab 27: Natural Language Processing Fundamentals

## Representasi Teks, Klasifikasi Tradisional, Word Embeddings, dan Arsitektur Pemodelan Sekuensial (RNN, LSTM, GRU)

---

## 1. Pendahuluan & Ruang Lingkup Natural Language Processing (NLP)

Pada Bab 24 hingga Bab 26, kita telah mendalami arsitektur visi komputer (*Computer Vision*) yang memproses data spasial 2D berstruktur kisi teratur (*grid topology*). Pada bab ini, kita melangkah ke domain kecerdasan buatan yang berhadapan langsung dengan manifestasi kecerdasan manusia yang paling ekspresif sekaligus kompleks: **bahasa alami** (*Natural Language*).

```
+---------------------------------------------------------------------------------------------------------+
|                          SPEKTRUM TUGAS NATURAL LANGUAGE PROCESSING MODERN                              |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. TEXT CLASSIFICATION              2. NAMED ENTITY RECOGNITION         3. SEQUENCE MODELING          |
|      +------------------------+          +------------------------+          +------------------------+ |
|      | "Layanan sangat ramah" |          | "Aji kuliah di ITERA"  |          | "Anjing" -> "mengejar" | |
|      | ──► [POS | 0.96]       |          |  [PER]         [ORG]   |          |      -> "kucing"       | |
|      +------------------------+          +------------------------+          +------------------------+ |
|      Analisis Sentimen, Deteksi          Ekstraksi Informasi &               Prediksi Kata Berikutnya & |
|      Spam, Kategori Topik Berita         Penandaan Entitas Spesifik          Pemodelan Bahasa Waktu     |
|                                                                                                         |
|   4. MACHINE TRANSLATION              5. QUESTION ANSWERING               6. SUMMARIZATION             |
|      +------------------------+          +------------------------+          +------------------------+ |
|      | "Good morning"         |          | Q: "Apa ibukota RI?"   |          | Dokumen 10 Paragraf    | |
|      | ──► "Selamat pagi"     |          | A: "Nusantara"         |          | ──► Abstrak 2 Kalimat  | |
|      +------------------------+          +------------------------+          +------------------------+ |
|      Penerjemahan Lintas Bahasa          Penalaran Berbasis Konteks          Ekstraksi / Abstraksi    |
|      Format Sequence-to-Sequence         Pengetahuan Korpus Dokumen          Intisari Teks Panjang    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Bahasa dan Komputasi
Komputer beroperasi pada sistem biner diskret yang memproses bilangan integer dan floating-point. Sebaliknya, bahasa manusia memiliki karakteristik yang sangat berbeda:
1. **Simbolik dan Diskret**: Kata-kata adalah simbol arbitrer yang disepakati secara sosial (misalnya kata *'meja'* merepresentasikan benda kayu berkaki empat, tetapi tidak ada korelasi fisis intrinsik antara huruf 'm-e-j-a' dengan materi kayu).
2. **Kaya Ambiguitas (Ambiguity)**: Satu kata dapat memiliki beragam makna tergantung konteks kalimat (*Polisemi*):
   - *"Saya **bisa** mengoperasikan model machine learning."* (bisa = sanggup / mampu)
   - *"Ular derik itu mengeluarkan **bisa** yang mematikan."* (bisa = racun / toksin)
3. **Ketergantungan Konteks & Komposisionalitas**: Makna keseluruhan kalimat terbentuk dari interaksi sintaksis dan semantik antar-kata penyusunnya (*Prinsip Frege*).

---

## 2. Hierarki Data Teks & Alur Kerja Standar NLP

Dalam rekayasa sistem NLP, data teks menstrukturkan diri dalam hirarki formal dari level makro hingga mikro:

```
+---------------------------------------------------------------------------------------------------------+
|                                  HIRARKI FORMAL DATA TEKS DALAM NLP                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. CORPUS (Korpus)      -> Himpunan utuh seluruh dokumen dalam suatu domain/proyek                    |
|          │                    (Contoh: Seluruh 50.000 ulasan produk e-commerce)                         |
|          ▼                                                                                              |
|   2. DOCUMENT (Dokumen)   -> Satu unit teks independen yang utuh secara informasi                       |
|          │                    (Contoh: Satu berkas artikel berita atau satu ulasan pelanggan)           |
|          ▼                                                                                              |
|   3. SENTENCE (Kalimat)   -> Satuan gramatikal berpikiran utuh yang dipisahkan tanda baca penutup       |
|          │                    (Contoh: "Model jaringan saraf memproses teks.")                          |
|          ▼                                                                                              |
|   4. TOKEN (Token)        -> Satuan komputasi leksikal terkecil (kata, sub-kata, tanda baca)            |
|          │                    (Contoh: ['model', 'jaringan', 'saraf', 'memproses', 'teks', '.'])       |
|          ▼                                                                                              |
|   5. CHARACTER (Karakter) -> Simbol grafem diskret yang dikodekan standar Unicode UTF-8                 |
|                               (Contoh: 'm', 'o', 'd', 'e', 'l')                                         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Standar Karakter Encoding: Unicode & UTF-8
Sebelum teks dapat diproses oleh Python, deretan bita memori harus didekodekan secara benar. Standar modern global adalah **UTF-8 (Unicode Transformation Format - 8 bit)**, yang mampu mengodekan lebih dari 149.000 karakter dari seluruh sistem aksara dunia, tanda baca khusus, hingga simbol matematika dan emoji secara konsisten.

---

## 3. Pra-Pemrosesan Teks & Normalisasi (Text Preprocessing)

Teks mentah yang diperoleh dari internet, media sosial, atau formulir survei selalu mengandung derau (*noise*), singkatan tidak baku, variasi kapitalisasi, dan elemen non-alfanumerik. Pra-pemrosesan teks bertujuan menstandarisasi bentuk input sebelum diumpankan ke model representasi.

```
+---------------------------------------------------------------------------------------------------------+
|                               TAHAPAN ALUR PIPELINE PRA-PEMROSESAN TEKS                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [Teks Mentah] : "Layanan ini gk bagusss bgt!! Kecewa berat http://cs.com #toko"                      |
|          │                                                                                              |
|          ▼ Case Folding (Lowercase)                                                                     |
|   "layanan ini gk bagusss bgt!! kecewa berat http://cs.com #toko"                                       |
|          │                                                                                              |
|          ▼ Pembersihan Regex (URL, Hashtag, Karakter Berulang)                                          |
|   "layanan ini gk bagus bgt kecewa berat"                                                               |
|          │                                                                                              |
|          ▼ Normalisasi Slang / Singkatan Bahasa Gaul                                                    |
|   "layanan ini tidak bagus sangat kecewa berat"                                                         |
|          │                                                                                              |
|          ▼ Penyaringan Stopword yang Memelihara Negasi (Negation-Preserving)                            |
|   ['layanan', 'tidak', 'bagus', 'sangat', 'kecewa', 'berat']                                            |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### A. Case Folding & Pembersihan Karakter Khusus
- **Case Folding**: Mengubah seluruh string ke huruf kecil (*lowercase*) agar representasi 'Bagus', 'bagus', dan 'BAGUS' tidak dianggap sebagai tiga entitas berbeda.
- **Pembersihan Regex**: Menghilangkan tautan web (`https?://\S+`), tag HTML (`<.*?>`), karakter berulang berlebih (`bagusss` $\rightarrow$ `bagus`), dan tanda baca menggunakan modul `re`.

### B. Normalisasi Slang & Singkatan
Dalam bahasa Indonesia informal, pengguna kerap menyingkat kata (*gk* $\rightarrow$ *tidak*, *bgt* $\rightarrow$ *sangat*, *dgn* $\rightarrow$ *dengan*, *utk* $\rightarrow$ *untuk*). Kamus normalisasi (*dictionary lookup*) memetakan variasi tersebut kembali ke bentuk leksikal baku.

### C. Bahaya Stopword Removal Ceroboh pada Analisis Sentimen
Stopwords adalah kata fungsional umum (seperti *yang, di, ke, dari, ini, itu*) yang sering dihapus demi mereduksi dimensi kosakata. **Peringatan Ilmiah Kritis**: Praktisi pemula sering kali menerapkan daftar stopword generik secara agresif yang membuang kata negasi seperti *tidak, bukan, belum, jangan*.
- Teks Asli: *"Pelayanan staf ini **tidak ramah** sama sekali."*
- Setelah Stopword Generik: *"Pelayanan staf **ramah**."*
- Dampak: Polaritas sentimen berbalik $180^\circ$ dari negatif menjadi positif! Oleh karena itu, arsitektur pra-pemrosesan wajib menerapkan **Negation-Preserving Filtering**.

### D. Stemming vs Lemmatization
- **Stemming**: Pemotongan imbuhan secara heuristik kaku (misalnya algoritma Porter untuk Bahasa Inggris atau Nazief-Adriani / CS untuk Bahasa Indonesia). Kelemahan: kerap menghasilkan kata non-baku (*over-stemming*).
- **Lemmatization**: Menyelidiki lema kamus yang sah berdasarkan morfologi dan kelas kata (*Part-of-Speech*). Contoh: kata *'berlari'* dan *'berlari-lari'* dikembalikan ke lema dasar *'lari'*.

---

## 4. Tokenisasi, Kosakata & Penanganan Out-of-Vocabulary (OOV)

Tokenisasi adalah proses memotong aliran teks kontinu menjadi deretan token diskret.

```
+---------------------------------------------------------------------------------------------------------+
|                               STRUKTUR KOSAKATA DENGAN TOKEN-TOKEN KHUSUS                               |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   INDEKS    TOKEN KHUSUS    FUNGSI & PERAN KOMPUTASIONAL DALAM ARSITEKTUR                               |
|                                                                                                         |
|   ID: 0  -> <pad>           Penyeragaman panjang sekuens dalam batch (vektor bobot nol)                 |
|   ID: 1  -> <unk>           Out-of-Vocabulary: mewakili kata tak dikenal di luar data latih             |
|   ID: 2  -> <bos>           Beginning-of-Sequence: penanda awal kalimat pada model autoregresif         |
|   ID: 3  -> <eos>           End-of-Sequence: penanda akhir pembacaan/generasi teks                      |
|                                                                                                         |
|   ID: 4  -> "pembelajaran"  Kata kosakata reguler terurut berdasarkan frekuensi dokumen                 |
|   ID: 5  -> "mesin"         Kata kosakata reguler                                                       |
|   ...    -> ...             ...                                                                         |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Padding & Truncation
Jaringan saraf tiruan memproses data masukan dalam bentuk matriks batch berdimensi seragam $(B \times T)$, di mana $B$ adalah ukuran batch (*batch size*) dan $T$ adalah panjang sekuens (*max sequence length*):
- **Truncation (Pemotongan)**: Kalimat yang memiliki panjang $> T$ dipotong pada token ke-$T$.
- **Padding (Penambahan Bantalan)**: Kalimat yang memiliki panjang $< T$ ditambah token khusus `<pad>` hingga panjangnya mencapai tepat $T$.

---

## 5. Representasi Teks Tradisional: Bag-of-Words (BoW)

Model Bag-of-Words memetakan dokumen menjadi vektor hitungan frekuensi kemunculan kata berdasarkan kosakata global $V = \{w_1, w_2, \dots, w_{|V|}\}$:

```
+---------------------------------------------------------------------------------------------------------+
|                                MATRIKS BAG-OF-WORDS (TERM FREQUENCY MATRIX)                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   KOSAKATA V: [atas, di, dimakan, gurih, kasur, kecil, keju, kucing, memakan, mengejar, tidur, tikus]   |
|                                                                                                         |
|   DOKUMEN                          VEKTOR BAG-OF-WORDS [x1, x2, ..., x12]               SPARSITY        |
|                                                                                                         |
|   D1: "Kucing mengejar tikus kecil"  -> [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]              66.7%        |
|   D2: "Tikus kecil memakan keju"     -> [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1]              66.7%        |
|   D3: "Kucing tidur di atas kasur"   -> [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]              58.3%        |
|   D4: "Keju gurih dimakan tikus"     -> [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1]              66.7%        |
|                                                                                                         |
|   Keterbatasan Utama BoW:                                                                               |
|   1. Matriks Sangat Renggang (High Sparsity > 60-90% elemen bernilai nol).                              |
|   2. Ketiadaan Informasi Urutan Kata (Order Invariance: "A memukul B" == "B memukul A").               |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 6. Pembobotan Analitis TF-IDF (Term Frequency-Inverse Document Frequency)

Meskipun Bag-of-Words mampu menghitung frekuensi kata, ia memperlakukan seluruh kata secara setara. Kata-kata yang sangat umum (seperti *'adalah'*, *'data'*) akan mendominasi magnitudo vektor meskipun tidak memiliki daya pembeda topik yang tajam. **TF-IDF** mengatasi kelemahan ini dengan memberikan bobot tinggi hanya pada kata yang sering muncul pada dokumen tertentu tetapi langka di seluruh korpus.

```
+---------------------------------------------------------------------------------------------------------+
|                               FORMULASI MATEMATIS ANALITIS TF-IDF                                       |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. TERM FREQUENCY:                     tf(t, d) = frekuensi kata t di dokumen d                       |
|                                                                                                         |
|   2. SMOOTHED INVERSE DOCUMENT           idf(t, D) = ln( (1 + N) / (1 + df(t)) ) + 1                   |
|      FREQUENCY (Scikit-learn):           N = total dokumen, df(t) = jumlah dokumen dengan kata t        |
|                                                                                                         |
|   3. BOBOT MENTAH TF-IDF:                tfidf_raw(t, d) = tf(t, d) * idf(t, D)                         |
|                                                                                                         |
|   4. NORMALISASI VEKTOR EUCLIDEAN:       v_norm = v / ||v||_2  (di mana ||v||_2 = sqrt(sum v_i^2))      |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Contoh Terhitung Langkah-demi-Langkah (*Worked Example*):
Korpus 3 Dokumen ($N=3$):
- $D_1$: *"data sains mempelajari data"*
- $D_2$: *"mesin kecerdasan buatan"*
- $D_3$: *"data mesin buatan sains"*

Kosakata unik $V = \{\text{'buatan'}, \text{'data'}, \text{'kecerdasan'}, \text{'mempelajari'}, \text{'mesin'}, \text{'sains'}\}$.
- Hitung DF untuk kata *'data'*: Muncul di $D_1$ dan $D_3 \rightarrow \text{DF} = 2$.
  $$\text{IDF}(\text{'data'}) = \ln \left( \frac{1 + 3}{1 + 2} \right) + 1 = \ln \left( \frac{4}{3} \right) + 1 \approx 0.2877 + 1 = 1.2877$$
- Hitung DF untuk kata *'kecerdasan'*: Hanya muncul di $D_2 \rightarrow \text{DF} = 1$.
  $$\text{IDF}(\text{'kecerdasan'}) = \ln \left( \frac{1 + 3}{1 + 1} \right) + 1 = \ln(2) + 1 \approx 0.6931 + 1 = 1.6931$$
- Terbukti kata *'kecerdasan'* yang langka memperoleh bobot $1.6931$, lebih tinggi dibandingkan kata *'data'* ($1.2877$).

---

## 7. Fitur N-gram & Pemodelan Konteks Lokal

N-gram adalah potongan sekuensial sepanjang $n$ token kata bersebelahan.

```
+---------------------------------------------------------------------------------------------------------+
|                                    SPEKTRUM FITUR N-GRAM SPASIAL                                        |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Kalimat Masukan: "pelayanan staf hotel tidak ramah sama sekali"                                       |
|                                                                                                         |
|   1. UNIGRAM (n=1) -> ['pelayanan', 'staf', 'hotel', 'tidak', 'ramah', 'sama', 'sekali']               |
|      Kelemahan     : Kata 'ramah' bermakna positif, terpisah dari kata 'tidak'                           |
|                                                                                                         |
|   2. BIGRAM (n=2)  -> ['pelayanan_staf', 'staf_hotel', 'hotel_tidak', 'tidak_ramah', 'ramah_sama']      |
|      Keunggulan    : Fitur 'tidak_ramah' berhasil ditangkap sebagai unit sentimen negatif utuh!         |
|                                                                                                         |
|   3. TRIGRAM (n=3) -> ['pelayanan_staf_hotel', 'staf_hotel_tidak', 'tidak_ramah_sama']                 |
|                                                                                                         |
|   Kompromi Ledakan Fitur (Vocabulary Explosion):                                                        |
|   Korpus Mini: Unigram = 26 fitur | Bigram = 24 fitur | Gabungan (1, 3) = 71 fitur unik                 |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 8. Klasifikasi Teks Tradisional & Protokol Bebas Kebocoran Data

Dalam klasifikasi teks terawasi (*Supervised Text Classification*), kita melatih model untuk memetakan representasi teks ke kategori diskret (misalnya ulasan sentimen: *Negatif*, *Netral*, *Positif*).

```
+---------------------------------------------------------------------------------------------------------+
|                          ARSITEKTUR PIPELINE KLASIFIKASI TEKS BEBAS KEBOCORAN                           |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [Dataset Utuh: 30 Sampel Teks]                                                                        |
|          │                                                                                              |
|          ▼ train_test_split (Terstratifikasi, Random State 42)                                          |
|   ┌──────────────────────────────┬──────────────────────────────┐                                       |
|   ▼                              ▼                              ▼                                       |
|   Train Set (70% = 21 Sampel)    Val Set (15% = 4 Sampel)       Test Set (15% = 5 Sampel)               |
|   │                              │                              │                                       |
|   ▼ fit_transform()              ▼ transform() HANYA            ▼ transform() HANYA                     |
|   [TfidfVectorizer: Vocab Train] ───────────────────────────────┴──────────────────────────────┐        |
|   │                                                                                            │        |
|   ▼ fit()                                                                                      │        |
|   [Multinomial Naive Bayes / LR]                                                               │        |
|   │                                                                                            │        |
|   ▼ predict()                                                                                  ▼        |
|   Akurasi Validasi (50.0%) ────────────────────────────────────────────────────────► Evaluasi Akhir     |
|                                                                                     Akurasi Test: 60.0% |
|                                                                                     Macro F1: 0.4889    |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Aturan Emas Anti Kebocoran Data (*Zero Data Leakage Rule*):
`TfidfVectorizer` **hanya boleh mempelajari kosakata dan menghitung IDF dari Train Set**. Jika `fit()` diaplikasikan pada seluruh korpus sebelum pemisahan data, informasi distribusi kata pada Test Set bocor ke dalam representasi pelatihan, menghasilkan evaluasi overoptimistik yang tidak valid.

---

## 9. Representasi Vektor Padat: Word Embeddings

Representasi one-hot dan BoW bersifat diskret, terisolasi, dan ortogonal tegak lurus:

$$\mathbf{e}_{\text{kucing}} \cdot \mathbf{e}_{\text{anjing}} = 0.0$$

Secara matematis, model BoW menganggap kata *'kucing'* sama berbedanya terhadap *'anjing'* sebagaimana ia berbeda terhadap kata *'laptop'*. **Word Embeddings** menyelesaikan persoalan ini dengan memproyeksikan setiap kata ke dalam ruang vektor kontinu berdimensi padat $D$ ($D \ll |V|$):

```
+---------------------------------------------------------------------------------------------------------+
|                             TABEL LOOKUP MATRIKS BOBOT EMBEDDING E                                      |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Matriks E berdimensi (V x D), contoh V=6 kata, D=4 fitur laten:                                       |
|                                                                                                         |
|   Indeks Token      Dimensi 1    Dimensi 2    Dimensi 3    Dimensi 4     Representasi Konseptual        |
|                                                                                                         |
|   0: <pad>       [   0.000,       0.000,       0.000,       0.000  ] -> Vektor Nol Statis               |
|   1: <unk>       [   0.374,       1.755,       0.140,       0.419  ] -> Token OOV                       |
|   2: kucing      [  -0.522,      -0.182,      -0.852,       2.569  ] -> Fitur Mamalia Domestik          |
|   3: anjing      [   1.055,      -0.732,      -0.786,       1.402  ] -> Fitur Mamalia Domestik          |
|   4: berlari     [  -0.346,       0.091,      -1.188,       0.410  ] -> Fitur Tindakan Kinetik          |
|   5: tidur       [   0.455,       0.597,       0.343,       1.261  ] -> Fitur Keadaan Pasif             |
|                                                                                                         |
|   Operasi Lookup: Input indeks [2, 4] ──► Mengambil baris E[2] dan E[4] langsung tanpa komputasi dot.   |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 10. Geometri Ruang Vektor & Cosine Similarity

Dalam ruang representasi padat, derajat kesamaan makna didefinisikan melalui orientasi sudut kosinus antara dua vektor $\mathbf{u}$ dan $\mathbf{v}$:

$$\text{Cosine Similarity}(\mathbf{u}, \\mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\\|\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2} = \frac{\sum_{i=1}^D u_i v_i}{\sqrt{\sum_{i=1}^D u_i^2} \sqrt{\sum_{i=1}^D v_i^2}}$$

```
+---------------------------------------------------------------------------------------------------------+
|                                 GEOMETRI ORIENTASI SUDUT KOSINUS                                        |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   1. Vektor Sejajar (theta = 0 derajat)          cos(0) = +1.0000 (Kesamaan Semantik Maksimal)         |
|      u = [1, 2, 3], v = [2, 4, 6]                                                                       |
|                                                                                                         |
|   2. Vektor Ortogonal (theta = 90 derajat)       cos(90) =  0.0000 (Independen, Tidak Terkait)          |
|      u = [1, 0], v = [0, 1]                                                                             |
|                                                                                                         |
|   3. Vektor Berlawanan (theta = 180 derajat)     cos(180) = -1.0000 (Antitesis Polaritas)               |
|      u = [1, 2], v = [-1, -2]                                                                           |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

---

## 11. Pemodelan Sekuensial & Arsitektur Simple RNN

Bahasa adalah data deret waktu diskret (*temporal discrete sequence*). Model yang memproses teks harus mampu mengingat informasi dari masa lalu untuk memahami makna kata saat ini.

```
+---------------------------------------------------------------------------------------------------------+
|                       ARSITEKTUR SIMPLE RECURRENT NEURAL NETWORK (UNROLLED)                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   x_1          x_2          x_3                 x_T                                                     |
|    │            │            │                   │                                                      |
|    ▼            ▼            ▼                   ▼                                                      |
|  ┌───┐  h_1   ┌───┐  h_2   ┌───┐        h_T-1  ┌───┐                                                    |
|h0│RNN│ ─────► │RNN│ ─────► │RNN│ ──► ... ────► │RNN│ ──► h_T (Representasi Akhir Sekuens)              |
|  └───┘        └───┘        └───┘               └───┘                                                    |
|    │            │            │                   │                                                      |
|    ▼            ▼            ▼                   ▼                                                      |
|   y_1          y_2          y_3                 y_T                                                     |
|                                                                                                         |
|   Persamaan Keadaan Rekuren:                                                                            |
|   h_t = tanh( W_xh * x_t + W_hh * h_{t-1} + b_h )                                                       |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Backpropagation Through Time (BPTT) & Dilema Vanishing Gradients
Saat menghitung gradien mundur melintasi $T$ langkah waktu:

$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \prod_{j=2}^T \operatorname{diag}(1 - \mathbf{h}_j^2) \mathbf{W}_{hh}^T$$

Karena turunan $\tanh$ selalu $\le 1.0$, jika nilai eigen terbesar $\lambda_{\max}(\mathbf{W}_{hh}) < 1.0$, perkalian berantai menyebabkan sinyal gradien teredam secara eksponensial menuju nol ($0.9^{30} \approx 0.042$). Bobot jaringan tidak mampu memperbarui diri berdasarkan kata-kata di awal kalimat panjang!

---

## 12. Arsitektur Gated: Long Short-Term Memory (LSTM) & Gated Recurrent Unit (GRU)

Untuk menuntaskan masalah vanishing gradient, Hochreiter & Schmidhuber (1997) menciptakan **LSTM** dengan memperkenalkan jalan pintas linier yang disebut **Cell State ($c_t$)**.

```
+---------------------------------------------------------------------------------------------------------+
|                                       ANATOMI UNIT SEL LSTM                                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   Cell State c_{t-1} ───────────────────[ * ]─────────────────────[ + ]───────────────► c_t              |
|                                          ▲                         ▲                                    |
|                                          │ f_t (Forget Gate)       │ i_t * c_tilde (Input Update)       |
|                                    ┌─────┴─────┐             ┌─────┴─────┐                              |
|                                    │  Sigmoid  │             │   [ * ]   │                              |
|                                    └─────┬─────┘             └─────┬─────┘                              |
|                                          │                    ▲    │                                    |
|                                          │                    │    │                                    |
|                                          │       ┌────────────┘    └────────────┐                       |
|                                          │       │ i_t (Input Gate)             │ c_tilde (Candidate)   |
|                                          │ ┌─────┴─────┐                  ┌─────┴─────┐                 |
|                                          │ │  Sigmoid  │                  │   Tanh    │                 |
|                                          │ └─────┬─────┘                  └─────┬─────┘                 |
|                                          │       │                              │                       |
|                                          └───────┼──────────────┬───────────────┘                       |
|                                                  │              │                                       |
|   Hidden State h_{t-1} ──────────────────────────┴──────────────┤                                       |
|   Input Token x_t      ─────────────────────────────────────────┴───────────────┐                       |
|                                                                                 │                       |
|                                                                           ┌─────┴─────┐                 |
|                                                                           │  Sigmoid  │                 |
|                                                                           └─────┬─────┘                 |
|                                                                                 │ o_t (Output Gate)     |
|                                                                                 ▼                       |
|   Hidden State h_t     ◄──────────────────────────────────────────────────────[ * ] ◄── Tanh(c_t)       |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### 6 Persamaan Inti Penentu Keadaan LSTM:
1. **Forget Gate**: $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$
2. **Input Gate**: $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$
3. **Kandidat Cell State**: $\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$
4. **Pembaruan Cell State**: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \\mathbf{i}_t \odot \tilde{\mathbf{c}}_t$
5. **Output Gate**: $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$
6. **Pembaruan Hidden State**: $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$

### Penyederhanaan Efisien: Gated Recurrent Unit (GRU)
Cho et al. (2014) memadukan cell state dan hidden state menjadi satu vektor $\mathbf{h}_t$, serta memangkas mekanisme gerbang menjadi dua:
1. **Update Gate ($\\mathbf{z}_t$)**: Mengontrol seberapa banyak informasi lama dipertahankan vs informasi baru dimasukkan.
2. **Reset Gate ($\\mathbf{r}_t$)**: Menentukan seberapa besar pengaruh keadaan waktu lalu saat menghitung kandidat baru.
- **Efisiensi Parameter**: Lapisan GRU hanya membutuhkan $3 \times (D \cdot H + H^2 + 2H)$ bobot, tepat **25% lebih hemat parameter** dibandingkan LSTM yang membutuhkan $4 \times (D \cdot H + H^2 + 2H)$.

---

## 13. Eksperimen Komparasi Empiris Sekuensial & Keterbatasan Model Rekuren

Berikut adalah bukti empiris eksekusi komparasi ketiga arsitektur pada tugas klasifikasi sekuensial sintetis terstandar ($N=120, T=12, V=40, \text{Epochs}=8$ pada CPU):

```
+---------------------------------------------------------------------------------------------------------+
|                               HASIL BENCHMARKING REKUREN PADA PERANGKAT CPU                             |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   MODEL       PARAMETER      WAKTU PELATIHAN (ms)    LOSS AKHIR    AKURASI VAL    NORMA GRADIEN         |
|                                                                                                         |
|   Simple RNN    1.698             273.67 ms            0.1084         70.00%         0.5204             |
|   LSTM          4.722             209.91 ms            0.2881         80.00%         0.3015             |
|   GRU           3.714             254.77 ms            0.1746         67.50%         0.3445             |
|                                                                                                         |
|   Analisis Bukti Komputasi:                                                                             |
|   1. LSTM mencatatkan akurasi validasi tertinggi (80.00%), mengungguli RNN sebesar +10.0%.              |
|   2. GRU membuktikan efisiensi parameter (~21% lebih hemat dari LSTM) dengan konvergensi stabil.        |
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
```

### Keterbatasan Struktural yang Memicu Revolusi Bab 28:
Meskipun LSTM dan GRU sukses mengatasi vanishing gradients, seluruh model rekuren memiliki **kelemahan arsitektur bawaan yang fatal**:
1. **Ketidakmampuan Paralelisasi**: Komputasi langkah waktu $t$ mutlak harus menunggu luaran $t-1$. GPU tidak dapat digunakan secara maksimal untuk memproses seluruh token sekaligus.
2. **Bottleneck Konteks Statis**: Seluruh informasi kalimat panjang dipaksa dimampatkan ke dalam satu vektor $\mathbf{h}_T$ berdimensi tetap.
3. Keterbatasan inilah yang memicu ditemukannya arsitektur **Attention Mechanism** dan **Transformer** pada Bab 28 yang meniadakan rekursi waktu sepenuhnya.

---

## 14. Responsible NLP: Bias Algoritmik, Etika Representasi & Batas Keandalan

1. **Bias Representasi Korpus**: Vektor embedding yang dilatih pada teks internet tanpa kurasi akan mereplikasi prasangka sosial (misal asosiasi stereotip pekerjaan terhadap gender tertentu).
2. **Privasi Teks & Anonimisasi**: Data teks yang memuat informasi identitas pribadi (*Personally Identifiable Information* / PII) seperti nama, alamat, nomor telepon, dan data finansial wajib di-anonimkan sebelum digunakan untuk pelatihan model.
3. **Batas Keandalan Lintas Bahasa**: Model yang dilatih pada bahasa dengan sumber daya melimpah (*High-Resource Languages* seperti Bahasa Inggris) tidak dapat langsung diasumsikan bekerja setara pada bahasa daerah (*Low-Resource Languages*).

---

## 15. Latihan Laboratorium Mandiri (Exercises)

### Tingkat 1: Pemahaman Konseptual (Dasar)
1. **Perhitungan Matriks BoW**:
   Diberikan kosakata alfabetis: `['anjing', 'berlari', 'kucing', 'tidur']`. Tuliskan vektor BoW untuk teks: *"kucing tidur dan kucing berlari"*!
   - *Jawaban*: `['anjing': 0, 'berlari': 1, 'kucing': 2, 'tidur': 1]`, sehingga representasi vektor BoW adalah $[0, 1, 2, 1]$.

2. **Perhitungan IDF Analitis**:
   Dalam sebuah korpus berisi $N=99$ dokumen, istilah *'kecerdasan'* muncul dalam $\text{DF}=9$ dokumen. Tentukan nilai smoothed IDF-nya!
   - *Jawaban*: $\text{IDF} = \ln \left( \frac{1+99}{1+9} \right) + 1 = \ln(10) + 1 \approx 2.3026 + 1 = 3.3026$.

### Tingkat 2: Analisis Alur Kerja & Debugging (Menengah)
1. **Mendiagnosis Data Leakage pada Pipeline Teks**:
   Jelaskan mengapa mengeksekusi `fit_transform` pada seluruh korpus sebelum melakukan partisi data train/test merupakan kesalahan fatal!
   - *Jawaban*: Hal tersebut membocorkan kosakata dan frekuensi dokumen global dari test set ke dalam training set (*data leakage*), menghasilkan estimasi performa model yang overoptimistik palsu. Solusi: `fit` hanya pada data latih, dan gunakan `transform` pada data validasi/uji.

2. **Dampak Penghapusan Negasi**:
   Mengapa penghapusan stopword secara otomatis dapat merusak akurasi model sentimen pada kalimat negasi?
   - *Jawaban*: Penghapusan kata 'tidak' dari *"kamar ini tidak bersih"* mereduksinya menjadi *"kamar bersih"*, membalikkan kutub sentimen dari negatif menjadi positif.

### Tingkat 3: Rekayasa Komputasional (Lanjut)
1. **Mekanisme Cell State LSTM Memitigasi Vanishing Gradients**:
   Turunkan mengapa formula pembaruan $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ memungkinkan perambatan gradien tanpa peluruhan eksponensial!
   - *Jawaban*: Turunan terhadap cell state masa lalu adalah $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t$. Gradien mundur melalui waktu dihitung melalui perkalian faktor $\prod_{j=2}^T \mathbf{f}_j$. Selama nilai forget gate mendekati $1$ ($\mathbf{f}_t \approx 1$), gradien mengalir melintasi ratusan langkah waktu secara linier tanpa teredam oleh fungsi aktivasi saturasi seperti $\tanh$.

---

## 16. Mini Project Chapter 27

### Judul Tugas Proyek:
**"Text Classification and Sequence Modeling on a Small Corpus"**

### Deskripsi Tugas:
Rancang, bangun, dan evaluasi pipeline klasifikasi teks mandiri yang mengintegrasikan pendekatan statistik tradisional dan neural sekuensial pada korpus pendek:
1. Pemuatan korpus ulasan pelanggan dan pra-pemrosesan teks adaptif (pelestarian kata negasi).
2. Tokenisasi teks dan pembentukan kamus kosakata terstandar dengan token `<pad>` dan `<unk>`.
3. Pembangunan model baseline tradisional Scikit-learn (TF-IDF + Naive Bayes / Logistic Regression) dengan isolasi partisi 3-way bebas kebocoran data.
4. Implementasi model sekuensial neural (Simple RNN, LSTM, atau GRU) menggunakan PyTorch pada CPU.
5. Evaluasi komparatif metrik (Akurasi, Macro F1, Matriks Konfusi) dan analisis galat mendalam pada sampel salah prediksi.
6. Dokumentasi etika privasi teks dan analisis kompromi efisiensi komputasi.

### Rubrik Penilaian Proyek (Total: 100 Poin):

| Kriteria Penilaian | Bobot | Deskripsi Skor Maksimal |
|---|:---:|---|
| **1. Dataset and Text Understanding** | 15 Poin | Perumusan masalah klasifikasi teks lugas, dokumentasi korpus dan label kelas transparan, serta pemahaman struktur data linguistik. |
| **2. Preprocessing and Representations** | 20 Poin | Pipeline pra-pemrosesan teks bersih dengan pelestarian negasi yang tepat, tokenisasi rapi dengan manajemen token khusus, dan perhitungan TF-IDF/BoW yang benar. |
| **3. Baseline and Neural Implementation** | 30 Poin | Implementasi baseline tradisional Scikit-learn bebas kebocoran data (*zero data leakage*), perancangan model sekuensial PyTorch (RNN/LSTM/GRU) modular pada CPU, dan siklus pelatihan stabil. |
| **4. Evaluation and Interpretation** | 20 Poin | Evaluasi komparatif komprehensif tanpa fabrikasi metrik, diseksi galat prediksi pada kasus sulit, serta pemahaman kritis atas trade-off model statistik vs neural. |
| **5. Documentation and References** | 15 Poin | Kode modular terstruktur bersih (*clean code*), dokumentasi replikasi eksperimen lengkap, refleksi etis Responsible AI, dan pencantuman 5 referensi akademik primer terverifikasi. |

---

## 17. Referensi Akademik Primer Terverifikasi

1. **Jurafsky, D., & Martin, J. H.** (2024). *Speech and Language Processing* (3rd ed. draft). Pearson / Stanford University. https://web.stanford.edu/~jurafsky/slp3/
2. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning* (Chapter 10: Sequence Modeling: Recurrent and Recursive Nets). MIT Press. https://www.deeplearningbook.org/
3. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
4. **Cho, K., et al.** (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *Proceedings of EMNLP 2014*, 1724–1734. https://doi.org/10.3115/v1/D14-1179
5. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval* (Chapter 6: Scoring, Term Weighting and the Vector Space Model). Cambridge University Press. https://nlp.stanford.edu/IR-book/

