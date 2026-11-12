"""
Chapter 27 - Natural Language Processing Fundamentals
Script Pembangkit Slide Deck Reveal.js (deck.json)
Menghasilkan 36 slide Reveal.js terstruktur, semantik, ramah KaTeX, dan dilengkapi speaker notes mendalam.
"""

import json
from pathlib import Path

slides = [
    {
        "kind": "cover",
        "title": "Natural Language Processing Fundamentals",
        "body": "### Bab 27 — Artificial Intelligence Learning Hub\n\n**Representasi Teks, Klasifikasi Tradisional, Word Embeddings, dan Arsitektur Pemodelan Sekuensial (RNN, LSTM, GRU)**\n\n*Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*",
        "notes": "Selamat datang di Bab 27. Pada modul ini, kita melangkah dari representasi spasial citra menuju representasi temporal dan sekuensial bahasa alami (Natural Language Processing). Fokus kita adalah memahami bagaimana teks manusia ditransformasikan ke dalam representasi numerik dan diproses oleh arsitektur rekuren."
    },
    {
        "kind": "content",
        "title": "Mengapa Pemrosesan Bahasa Alami (NLP) Krusial?",
        "body": "Bahasa adalah media utama manusia mengekspresikan pengetahuan, emosi, dan penalaran:\n\n- **Mayoritas Data Tak Terstruktur**: Lebih dari 80% data digital korporat dan internet berwujud teks mentah (dokumen, email, interaksi obrolan, rekam medis).\n- **Jembatan Manusia dan Mesin**: Komputer bekerja pada bilangan biner diskret, sementara bahasa manusia sarat ambiguitas, metafora, dan variasi sintaksis.\n- **Fondasi AI Modern**: Pemahaman NLP fundamental merupakan prasyarat mutlak sebelum mempelajari arsitektur Transformer dan Large Language Models.",
        "notes": "Tekankan bahwa NLP bukan sekadar manipulasi string, melainkan rekayasa komputasi untuk mengekstraksi struktur makna (semantik) dari sinyal simbolik yang sangat ambigu."
    },
    {
        "kind": "content",
        "title": "Tujuan Pembelajaran (Learning Objectives)",
        "body": "Setelah menyelesaikan modul ini, mahasiswa diharapkan mampu:\n\n1. **Menjelaskan** ruang lingkup dan alur kerja standar NLP (*preprocessing* $\\rightarrow$ *representation* $\\rightarrow$ *modeling*).\n2. **Mengimplementasikan** pra-pemrosesan teks, tokenisasi, normalisasi, dan penanganan kata negasi.\n3. **Menghitung** pembobotan Bag-of-Words dan TF-IDF secara analitis dan numerik.\n4. **Membedakan** representasi vektor renggang (*sparse*) dengan *dense word embeddings*.\n5. **Membangun** pipeline klasifikasi teks bebas kebocoran data (*zero data leakage*).\n6. **Menganalisis** pemodelan sekuensial dan mekanisme gerbang pada Simple RNN, LSTM, dan GRU.",
        "notes": "Sampaikan bahwa bab ini mencakup dua pilar: pendekatan tradisional berbasis statistik (TF-IDF, Naive Bayes) dan pendekatan neural berbasis sekuens temporal (RNN, LSTM, GRU)."
    },
    {
        "kind": "content",
        "title": "Spektrum Aplikasi NLP di Dunia Industri",
        "body": "Teknologi NLP telah mentransformasi beragam vertikal industri modern:\n\n- **Text Classification & Sentiment Analysis**: Klasifikasi tiket bantuan teknis, moderasi konten media sosial, dan analisis opini pasar saham.\n- **Information Extraction & NER**: Ekstraksi entitas bernama (nama orang, organisasi, tanggal, dosis medis) dari dokumen kontrak hukum atau resep klinis.\n- **Machine Translation & Question Answering**: Penerjemahan lintas bahasa dan asisten virtual otomatis untuk layanan nasabah.\n- **Spam Filtering & Phishing Detection**: Penyaringan ancaman siber berbasis analisis pola leksikal dan semantik pesan.",
        "notes": "Berikan contoh konkret di Indonesia, misalnya klasifikasi aduan masyarakat pada portal lapor pemerintah atau chatbot layanan pelanggan perbankan nasional."
    },
    {
        "kind": "content",
        "title": "Tantangan Utama Bahasa Alami: Ambiguitas & Konteks",
        "body": "Berbeda dengan bahasa pemrograman yang deterministik, bahasa alami memiliki ambiguitas bawaan:\n\n- **Ambiguitas Leksikal (Polisemi)**: Kata 'bisa' dapat bermakna *mampu* atau *racun ular*; kata 'apel' dapat merujuk pada *buah* atau *upacara militer*.\n- **Ambiguitas Sintaksis**: 'Polisi menembak perampok dengan pistol' (siapa yang memegang pistol?).\n- **Variasi Kultural & Slang**: Singkatan pesan instan ('gk', 'tdk', 'bgt', 'mks') dan penggunaan sarkasme yang membalikkan polaritas makna teks.",
        "notes": "Soroti bahwa mesin tidak memiliki penalaran akal sehat (common sense) bawaan manusia, sehingga tahap pra-pemrosesan dan pemodelan konteks menjadi sangat kritis."
    },
    {
        "kind": "split",
        "title": "Hierarki Data Teks: Dari Korpus ke Token",
        "body": "### Korpus, Dokumen, dan Kalimat\n\n- **Corpus (Korpus)**: Himpunan utuh seluruh teks dalam proyek NLP (misal seluruh ulasan marketplace).\n- **Document (Dokumen)**: Satuan teks independen (satu ulasan atau satu artikel berita).\n- **Sentence (Kalimat)**: Satuan gramatikal yang diakhiri tanda titik atau intonasi penutup.\n\n---\n\n### Token dan Karakter\n\n- **Token**: Unit komputasi terkecil dalam pemrosesan teks (bisa berupa kata, sub-kata, atau karakter).\n- **Vocabulary (Kosakata)**: Himpunan seluruh token unik yang dikenali oleh sistem NLP.\n- **Character Encoding**: Standar Unicode (UTF-8) yang memetakan karakter multibahasa ke representasi biner memori.",
        "notes": "Jelaskan definisi formal ini agar mahasiswa tidak tertukar antara korpus, dokumen, dan token."
    },
    {
        "kind": "content",
        "title": "Tahapan Pra-Pemrosesan Teks (Text Preprocessing)",
        "body": "Pra-pemrosesan mengubah teks mentah manusia menjadi sekuens token yang bersih dan konsisten:\n\n1. **Case Folding**: Mengubah seluruh huruf menjadi huruf kecil (*lowercase*) untuk menyeragamkan 'Bagus', 'bagus', dan 'BAGUS'.\n2. **Noise Removal**: Menghilangkan tautan URL, tag HTML, mention, dan karakter non-alfanumerik menggunakan ekspresi reguler (*regex*).\n3. **Slang & Abbreviation Normalization**: Memetakan kata tidak baku ke bentuk formal baku ('gk' $\\rightarrow$ 'tidak', 'bgt' $\\rightarrow$ 'sangat').\n4. **Punctuation Handling**: Memisahkan atau menghapus tanda baca tanpa merusak makna struktural.",
        "notes": "Ingatkan bahwa aturan pembersihan teks harus disesuaikan dengan domain masalah. Jangan melakukan penghapusan secara buta tanpa memahami dampaknya pada data."
    },
    {
        "kind": "content",
        "title": "Tokenisasi: Dari Teks ke Rangkaian Simbol Diskret",
        "body": "Tokenisasi adalah proses memotong aliran teks kontinu menjadi deretan token diskret:\n\n- **Whitespace Tokenization**: Memotong berdasarkan karakter spasi putih. Sederhana namun gagal menangani tanda baca yang melekat erat pada kata ('bagus!' $\\ne$ 'bagus').\n- **Regex-based Tokenization**: Memisahkan kata alfanumerik sekaligus memperlakukan tanda baca sebagai token mandiri menggunakan pola `\\w+|[^\\w\\s]`.\n- **Subword Tokenization (BPE / WordPiece)**: Memecah kata majemuk atau berimbuhan rumit menjadi unit sub-kata ('pembelajaran' $\\rightarrow$ 'pem-', 'belajar', '-an') untuk mengatasi masalah Out-of-Vocabulary (OOV).",
        "notes": "Beri pengantar singkat tentang Byte-Pair Encoding (BPE) sebagai jembatan menuju arsitektur modern Transformer."
    },
    {
        "kind": "content",
        "title": "Normalisasi Bahasa: Stemming vs Lemmatization",
        "body": "Mengurangi variasi bentuk kata berimbuhan ke bentuk dasarnya (*root form*):\n\n- **Stemming (Pemotongan Morfologis Heuristik)**: Menghapus awalan/akhiran berdasarkan aturan heuristik kaku (misal algoritma Porter atau Nazief-Adriani untuk Bahasa Indonesia). Cepat namun rentan menghasilkan kata non-baku (*over-stemming* atau *under-stemming*).\n- **Lemmatization (Reduksi Berbasis Leksikon & POS)**: Mengembalikan kata ke lema kamus yang valid secara linguistik menggunakan analisis morfologis dan konteks kelas kata (*Part-of-Speech*).\n- **Pertimbangan Praktis**: Pada model neural modern, stemming sering ditinggalkan demi menjaga kekayaan semantik imbuhan asli.",
        "notes": "Tekankan perbedaan utama: Stemmer memotong teks secara algoritmik heuristik, sedangkan Lemmatizer selalu merujuk pada kamus morfologi resmi."
    },
    {
        "kind": "case",
        "title": "Studi Kasus: Bahaya Stopword Removal pada Analisis Sentimen",
        "body": "### Jebakan Penghapusan Kata Negasi\n\nBanyak praktisi secara otomatis menerapkan daftar stopword generik untuk menghapus kata-kata umum seperti 'dan', 'yang', 'ini', dan 'tidak':\n\n- Teks Asli: *'Makanan ini tidak enak dan pelayanannya tidak ramah.'*\n- Setelah Stopword Removal Agresif: *'Makanan enak pelayanan ramah.'*\n\n### Dampak Katastrofik\nPolaritas sentimen berbalik $180^\\circ$ dari **Sangat Negatif** menjadi **Sangat Positif** hanya karena kata negasi 'tidak' ikut terhapus! Oleh karena itu, saring stopword secara selektif dengan mempertahankan kelompok kata negasi (*negation-aware filtering*).",
        "notes": "Gunakan slide studi kasus ini untuk memicu diskusi kelas mengenai pentingnya domain context dalam NLP pipeline."
    },
    {
        "kind": "split",
        "title": "Manajemen Kosakata & Penanganan Out-of-Vocabulary (OOV)",
        "body": "### Pembangunan Vocabulary\n\n- Kosakata dibangun dari himpunan kata unik pada data latih (*training set*).\n- Kata dengan frekuensi sangat rendah ($< \\text{min\\_freq}$) disaring untuk menekan dimensi fitur.\n\n---\n\n### Token Khusus Rekayasa (Special Tokens)\n\n- `<pad>` (Index 0): Mengisi posisi kosong agar panjang sekuens dalam batch seragam.\n- `<unk>` (Index 1): Mewakili kata baru di luar kosakata data latih (*Out-of-Vocabulary*).\n- `<bos>` / `<eos>`: Menandai awal (*Beginning*) dan akhir (*End*) suatu sekuens kalimat.",
        "notes": "Jelaskan bahwa token OOV adalah keniscayaan dalam dunia nyata karena bahasa manusia selalu berkembang memunculkan istilah baru."
    },
    {
        "kind": "content",
        "title": "Model Representasi Klasik: Bag-of-Words (BoW)",
        "body": "Model Bag-of-Words memetakan dokumen menjadi vektor hitungan frekuensi kemunculan kata:\n\n- Setiap dimensi dalam vektor mewakili satu kata unik dari kosakata global $V$.\n- Nilai sel $M_{i, j}$ adalah frekuensi kemunculan kata $j$ pada dokumen $i$.\n- **Kelebihan**: Sederhana diimplementasikan, mudah diinterpretasikan secara statistik, dan sangat cepat dilatih.\n- **Keterbatasan Utama**: Mengabaikan total urutan kata (*order invariance*). Kalimat 'Saya tidak membenci kamu' dan 'Kamu tidak membenci saya' menghasilkan vektor BoW yang persis identik!",
        "notes": "Tunjukkan bahwa BoW adalah representasi himpunan multiset murni yang bersifat amnesia terhadap sintaksis gramatikal."
    },
    {
        "kind": "content",
        "title": "Formulasi Matematis TF-IDF (Term Frequency-IDF)",
        "body": "TF-IDF memberikan bobot tinggi pada kata yang sering muncul dalam suatu dokumen, namun jarang muncul di dokumen-dokumen lain dalam korpus:\n\n$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$\n\nFormula Inverse Document Frequency terhalus (*smoothed IDF* standar Scikit-learn):\n\n$$\\text{IDF}(t, D) = \\ln \\left( \\frac{1 + N}{1 + \\text{DF}(t)} \\right) + 1$$\n\nDiikuti normalisasi vektor Euclidean (L2-norm) untuk menyetarakan pengaruh panjang dokumen:\n\n$$\\mathbf{v}_{\\text{norm}} = \\frac{\\mathbf{v}}{\\|\\mathbf{v}\\|_2} = \\frac{\\mathbf{v}}{\\sqrt{\\sum_{j} v_j^2}}$$",
        "notes": "Jelaskan bahwa penambahan angka 1 pada penyebut dan pembilang bertujuan mencegah pembagian dengan nol dan nilai negatif pada logaritma."
    },
    {
        "kind": "content",
        "title": "Menangkap Konteks Lokal dengan Fitur N-gram",
        "body": "N-gram adalah urutan kontinu dari $n$ token kata yang bersebelahan dalam teks:\n\n- **Unigram ($n=1$)**: ['pelayanan', 'tidak', 'ramah']\n- **Bigram ($n=2$)**: ['pelayanan\\_tidak', 'tidak\\_ramah']\n- **Trigram ($n=3$)**: ['pelayanan\\_tidak\\_ramah']\n\n### Keunggulan & Kompromi Komputasi:\n- **Menangkap Frasa & Negasi**: Bigram 'tidak\\_ramah' menjadi fitur mandiri pembawa sentimen negatif, terpisah dari unigram 'ramah'.\n- **Ledakan Kosakata (*Vocabulary Explosion*)**: Jumlah kemungkinan n-gram tumbuh eksponensial $O(|V|^n)$, memperparah kelangkaan matriks (*sparsity*) dan kebutuhan RAM.",
        "notes": "Jelaskan trade-off antara kekayaan konteks lokal dengan dimensi matriks fitur yang membengkak."
    },
    {
        "kind": "content",
        "title": "Alur Kerja Klasifikasi Teks Tradisional",
        "body": "Pipeline klasifikasi teks yang kokoh memerlukan pemisahan ketat antar-tahap:\n\n1. **Data Split 3-Way**: Membagi data ke dalam Train (70%), Validation (15%), dan Test (15%) secara terstratifikasi.\n2. **Isolasi Vektorisasi (Pencegahan Data Leakage)**: Kosakata dan parameter IDF **wajib di-fit HANYA pada data latih**. Data validasi dan uji hanya boleh melalui operasi `transform()`.\n3. **Model Estimator**: Memanfaatkan algoritma probabilistik seperti Multinomial Naive Bayes atau regresi linear ter-regularisasi (Logistic Regression).\n4. **Scikit-learn Pipeline**: Mengemas `TfidfVectorizer` dan estimator ke dalam satu objek tunggal untuk mencegah kebocoran informasi evaluasi.",
        "notes": "Tekankan bahaya data leakage: jika vectorizer di-fit pada seluruh dataset sebelum split, model akan mengetahui kosakata dan distribusi IDF dari test set, menghasilkan evaluasi overoptimistik palsu."
    },
    {
        "kind": "split",
        "title": "Model Klasifikasi Teks: Naive Bayes vs Logistic Regression",
        "body": "### Multinomial Naive Bayes\n\n- Berbasis Teorema Bayes dengan asumsi independensi fitur bersyarat: $P(y|X) \\propto P(y) \\prod P(x_i|y)$.\n- Menghitung frekuensi relatif kata dengan *Laplace Smoothing* ($\\alpha=1.0$).\n- Sangat cepat, efisien memori, tangguh pada dataset berukuran kecil.\n\n---\n\n### Logistic Regression\n\n- Memodelkan probabilitas log-odds menggunakan fungsi aktivasi Sigmoid: $\\sigma(\\mathbf{w}^T \\mathbf{x} + b)$.\n- Menemukan bobot optimal melalui optimasi gradien loss Cross-Entropy.\n- Mampu mempelajari interaksi fitur negatif dan positif secara fleksibel melalui regularisasi $L_1$ / $L_2$.",
        "notes": "Jelaskan bahwa kedua model ini adalah baseline industri yang sangat kuat dan sering kali sulit dikalahkan pada dataset tabular teks pendek."
    },
    {
        "kind": "content",
        "title": "Representasi Vektor Padat: Word Embeddings",
        "body": "Keterbatasan utama representasi One-Hot dan BoW adalah sifatnya yang *sparse*, berdimensi sangat besar, dan **buta makna semantik** (vektor 'kucing' dan 'anjing' ortogonal tegak lurus dengan dot product 0).\n\n- **Distributed Representation**: Setiap kata diproyeksikan ke dalam ruang vektor padat (*dense vector*) berdimensi rendah ($D \\in [50, 300]$).\n- **Prinsip Kontekstual (Harris & Firth)**: *'You shall know a word by the company it keeps.'* Kata-kata yang sering muncul dalam konteks kalimat serupa akan memiliki koordinat spasial yang berdekatan.\n- **Model Fondasi Klasik**: Word2Vec (Skip-gram & CBOW), GloVe (matriks ko-okurensi global), dan FastText (representasi berbasis n-gram subkata).",
        "notes": "Tekankan lompatan konseptual: dari representasi simbolik diskret ke representasi kontinu di mana jarak geometri mencerminkan relasi semantik."
    },
    {
        "kind": "content",
        "title": "Mekanisme Operasi Embedding Lookup",
        "body": "Dalam komputasi neural, lapisan embedding direpresentasikan sebagai matriks bobot terpelajari $E \\in \\mathbb{R}^{V \\times D}$:\n\n- $V$: Ukuran kosakata (*vocabulary size*).\n- $D$: Dimensi embedding (*embedding dimension*).\n- **Operasi Lookup**: Mengambil vektor representasi dari token berindeks $k$ setara dengan mengalikan vektor one-hot $\\mathbf{e}_k$ terhadap matriks $E$, yang dalam praktiknya diimplementasikan secara efisien sebagai seleksi baris $E[k]$:\n\n$$\\mathbf{v}_k = \\text{Lookup}(E, k) = E[k, :]$$\n\n- **Efisiensi**: Menghindari perkalian matriks besar yang boros komputasi via pengindeksan array langsung.",
        "notes": "Tunjukkan bahwa PyTorch nn.Embedding pada dasarnya adalah tabel lookup bobot teroptimasi yang dapat diperbarui gradiennya selama pelatihan."
    },
    {
        "kind": "content",
        "title": "Mengukur Kesamaan Semantik: Cosine Similarity",
        "body": "Dalam ruang vektor padat, derajat kesamaan antara dua representasi kata diukur melalui sudut kosinus antar-vektor:\n\n$$\\text{Cosine Similarity}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2} = \\frac{\\sum_{i=1}^D u_i v_i}{\\sqrt{\\sum_{i=1}^D u_i^2} \\sqrt{\\sum_{i=1}^D v_i^2}}$$\n\n- Rentang Nilai: $[-1.0, +1.0]$.\n  - $+1.0$: Vektor berorientasi sejajar searah (kesamaan semantik tinggi).\n  - $0.0$: Vektor ortogonal saling tegak lurus (independen tanpa korelasi).\n  - $-1.0$: Vektor berorientasi berlawanan arah.\n- **Invarian terhadap Magnitudo**: Mengukur arah orientasi murni, tidak terpengaruh oleh panjang absolut vektor.",
        "notes": "Ingatkan kembali bahwa cosine similarity mengukur geometri ruang vektor terdefinisi dan tidak boleh dianggap sebagai bukti kebenaran ontologis absolut."
    },
    {
        "kind": "content",
        "title": "Keterbatasan Representasi Statis & Pentingnya Urutan Kata",
        "body": "Model representasi statis (BoW, TF-IDF, Word2Vec statis) memiliki batas kapabilitas mendasar:\n\n1. **Polisemi Tak Terpecahkan**: Kata 'bisa' memiliki satu vektor tunggal yang sama persis pada kalimat 'saya bisa makan' dan 'ular itu berbisa'.\n2. **Kehancuran Struktur Temporal**: Makna bahasa sangat bergantung pada urutan kata (*word order*). Perhatikan perbedaan makna dua kalimat berikut:\n   - *'Anjing menggigit orang.'* (Kejadian lumrah)\n   - *'Orang menggigit anjing.'* (Kejadian janggal dan sensasional)\n3. **Solusi Arsitektur**: Kita membutuhkan model jaringan saraf yang memiliki memori internal untuk memproses sekuens data langkah demi langkah: **Recurrent Neural Networks (RNN)**.",
        "notes": "Slide ini menjadi jembatan logis transisi dari pemrosesan teks tradisional menuju pemodelan sekuens rekuren (Sequence Modeling)."
    },
    {
        "kind": "content",
        "title": "Arsitektur Dasar Simple Recurrent Neural Network (RNN)",
        "body": "Simple RNN (Elman RNN) memperkenalkan sambungan umpan balik siklis (*recurrent loop*) yang meneruskan informasi waktu lalu ke masa kini:\n\n$$\\mathbf{h}_t = \\tanh(\\mathbf{W}_{xh} \\mathbf{x}_t + \\mathbf{W}_{hh} \\mathbf{h}_{t-1} + \\mathbf{b}_h)$$\n\n$$\\hat{\\mathbf{y}}_t = \\mathbf{W}_{hy} \\mathbf{h}_t + \\mathbf{b}_y$$\n\n- $\\mathbf{x}_t \\in \\mathbb{R}^D$: Vektor masukan token pada langkah waktu $t$.\n- $\\mathbf{h}_{t-1} \\in \\mathbb{R}^H$: Vektor keadaan tersembunyi (*hidden state*) dari langkah waktu sebelumnya yang berfungsi sebagai memori konteks historis.\n- $\\mathbf{h}_t \\in \\mathbb{R}^H$: Keadaan tersembunyi terbarukan pada waktu $t$.\n- $\\mathbf{W}_{hh} \\in \\mathbb{R}^{H \\times H}$: Matriks bobot transisi rekuren internal.",
        "notes": "Soroti pembagian bobot (weight sharing): matriks bobot W_xh dan W_hh digunakan berulang kali pada setiap langkah waktu t."
    },
    {
        "kind": "content",
        "title": "Representasi Bentangan Waktu (Unrolled RNN Computational Graph)",
        "body": "Untuk memahami alur komputasi RNN, jaringan dapat dibentangkan (*unrolled*) melintasi seluruh langkah waktu $t = 1, 2, \\dots, T$:\n\n```\n   x_1          x_2          x_3                 x_T\n    │            │            │                   │\n    ▼            ▼            ▼                   ▼\n  ┌───┐  h_1   ┌───┐  h_2   ┌───┐        h_T-1  ┌───┐\nh0│RNN│ ─────► │RNN│ ─────► │RNN│ ──► ... ────► │RNN│ ──► h_T\n  └───┘        └───┘        └───┘               └───┘\n    │            │            │                   │\n    ▼            ▼            ▼                   ▼\n   y_1          y_2          y_3                 y_T\n```\n\n- Parameter $(\\mathbf{W}_{xh}, \\mathbf{W}_{hh}, \\mathbf{W}_{hy})$ **dibagikan secara identik (*weight sharing*)** di seluruh bentangan waktu.\n- Pola Pemrosesan: *Sequence-to-One* (misal klasifikasi teks dari $h_T$) atau *Sequence-to-Sequence* (misal penandaan POS token per waktu).",
        "notes": "Tekankan konsep unrolling sebagai dasar dari algoritma Backpropagation Through Time (BPTT)."
    },
    {
        "kind": "content",
        "title": "Pelatihan RNN: Backpropagation Through Time (BPTT)",
        "body": "Optimasi parameter RNN dilakukan dengan merambatkan galat mundur melintasi waktu:\n\n- Total fungsi rugi adalah akumulasi rugi pada setiap langkah waktu: $\\mathcal{L} = \\sum_{t=1}^T \\mathcal{L}_t$.\n- Melalui aturan rantai (*chain rule*), gradien fungsi rugi terhadap bobot rekuren $\\mathbf{W}_{hh}$ melibatkan perkalian matriks Jacobian berulang:\n\n$$\\frac{\\partial \\mathcal{L}_T}{\\partial \\mathbf{h}_1} = \\frac{\\partial \\mathcal{L}_T}{\\partial \\mathbf{h}_T} \\prod_{j=2}^T \\frac{\\partial \\mathbf{h}_j}{\\partial \\mathbf{h}_{j-1}}$$\n\n- Di mana matriks transisi turunan bernilai:\n\n$$\\frac{\\partial \\mathbf{h}_j}{\\partial \\mathbf{h}_{j-1}} = \\operatorname{diag}(1 - \\tanh^2(\\mathbf{z}_j)) \\mathbf{W}_{hh}^T$$",
        "notes": "Jelaskan bahwa faktor perkalian berantai T-1 kali inilah yang memicu masalah numerik fundamental pada sekuens panjang."
    },
    {
        "kind": "case",
        "title": "Dilema Fundamental: Vanishing & Exploding Gradients",
        "body": "### Mengapa Simple RNN Gagal Mengingat Sekuens Panjang?\n\nPerkalian matriks turunan berantai sebanyak $T$ kali menimbulkan dua kondisi ekstrem:\n\n1. **Vanishing Gradients (Paling Sering Terjadi)**:\n   Jika nilai eigen terbesar dari $\\mathbf{W}_{hh}$ bernilai $< 1.0$ dan turunan $\\tanh \\le 1.0$, sinyal gradien teredam secara eksponensial menuju nol ($0.9^{50} \\approx 0.0051$). Akibatnya, bobot tidak dapat diperbarui berdasarkan informasi kata di awal kalimat panjang!\n2. **Exploding Gradients**:\n   Jika nilai eigen $> 1.0$, gradien meledak menjadi bilangan raksasa (`NaN`), menyebabkan ketidakstabilan pelatihan (dapat dimitigasi dengan *Gradient Clipping*).\n3. **Dampak Praktis**: Simple RNN kehilangan memori ketergantungan jarak jauh (*long-range dependencies*) setelah 10–15 langkah waktu.",
        "notes": "Hubungkan slide ini secara erat dengan pembahasan kalkulus dan optimasi gradien pada Bab 22."
    },
    {
        "kind": "content",
        "title": "Solusi Arsitektur Gated: Long Short-Term Memory (LSTM)",
        "body": "Hochreiter & Schmidhuber (1997) merancang **LSTM** untuk mengatasi vanishing gradients secara struktural:\n\n- **Cell State (Jalur Pintas $c_t$)**: Berfungsi seperti jalan tol konveyor (*conveyor belt*) linier yang mengalirkan informasi melintasi sekuens dengan degradasi gradien minimal.\n- **Gerbang Pengatur Aliran (*Gating Mechanism*)**: Tiga gerbang berbasis fungsi aktivasi Sigmoid $\\sigma \\in [0, 1]$ yang secara selektif menentukan berapa banyak informasi yang dilupakan, ditambahkan, dan dikeluarkan:\n  1. **Forget Gate ($f_t$)**: Menentukan memori lampau yang harus dibuang.\n  2. **Input Gate ($i_t$)**: Menentukan informasi baru yang layak disimpan ke cell state.\n  3. **Output Gate ($o_t$)**: Menentukan bagian cell state yang diproyeksikan ke hidden state.",
        "notes": "Analogikan Cell State sebagai buku catatan permanen, sedangkan gerbang-gerbang adalah penyunting yang menentukan tulisan mana yang dihapus, dicatat, dan dibaca."
    },
    {
        "kind": "content",
        "title": "Formulasi Matematis 6 Persamaan Inti LSTM",
        "body": "Pada setiap langkah waktu $t$, unit sel LSTM menghitung secara deterministik:\n\n1. **Forget Gate**: $\\mathbf{f}_t = \\sigma(\\mathbf{W}_f \\mathbf{x}_t + \\mathbf{U}_f \\mathbf{h}_{t-1} + \\mathbf{b}_f)$\n2. **Input Gate**: $\\mathbf{i}_t = \\sigma(\\mathbf{W}_i \\mathbf{x}_t + \\mathbf{U}_i \\mathbf{h}_{t-1} + \\mathbf{b}_i)$\n3. **Kandidat Cell State**: $\\tilde{\\mathbf{c}}_t = \\tanh(\\mathbf{W}_c \\mathbf{x}_t + \\mathbf{U}_c \\mathbf{h}_{t-1} + \\mathbf{b}_c)$\n4. **Pembaruan Cell State**: $\\mathbf{c}_t = \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{c}}_t$\n5. **Output Gate**: $\\mathbf{o}_t = \\sigma(\\mathbf{W}_o \\mathbf{x}_t + \\mathbf{U}_o \\mathbf{h}_{t-1} + \\mathbf{b}_o)$\n6. **Pembaruan Hidden State**: $\\mathbf{h}_t = \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t)$\n\n*Simbol $\\odot$ menyatakan perkalian elemen-demi-elemen (Hadamard product).*",
        "notes": "Tunjukkan bahwa persamaan 4 adalah operasi penambahan linier (bukan perkalian matriks), sehingga gradien dapat mengalir mundur tanpa pelemahan eksponensial."
    },
    {
        "kind": "content",
        "title": "Penyederhanaan Efisien: Gated Recurrent Unit (GRU)",
        "body": "Cho et al. (2014) mengusulkan variasi penyederhanaan LSTM yang lebih ramping:\n\n- Menggabungkan *Cell State* dan *Hidden State* menjadi satu keadaan tunggal $\\mathbf{h}_t$.\n- Menyederhanakan mekanisme kontrol menjadi **dua gerbang saja**:\n  1. **Update Gate ($\\mathbf{z}_t$)**: Mengatur keseimbangan antara memori lama dan kandidat baru (berfungsi ganda menggantikan forget dan input gate):\n     $$\\mathbf{z}_t = \\sigma(\\mathbf{W}_z \\mathbf{x}_t + \\mathbf{U}_z \\mathbf{h}_{t-1} + \\mathbf{b}_z)$$\n  2. **Reset Gate ($\\mathbf{r}_t$)**: Menentukan seberapa besar keadaan masa lalu dilupakan sebelum menghitung kandidat keadaan baru:\n     $$\\mathbf{r}_t = \\sigma(\\mathbf{W}_r \\mathbf{x}_t + \\mathbf{U}_r \\mathbf{h}_{t-1} + \\mathbf{b}_r)$$\n- **Pembaruan Keadaan**: $\\mathbf{h}_t = (1 - \\mathbf{z}_t) \\odot \\mathbf{h}_{t-1} + \\mathbf{z}_t \\odot \\tilde{\\mathbf{h}}_t$.",
        "notes": "Jelaskan bahwa GRU menghemat waktu pelatihan dan memori, menjadikannya sangat populer pada perangkat edge dan dataset menengah."
    },
    {
        "kind": "split",
        "title": "Komparasi Arsitektur: Simple RNN vs LSTM vs GRU",
        "body": "### Dimensi & Kompleksitas Komputasi\n\n- **Simple RNN**: 1 persamaan keadaan, 1 fungsi aktivasi $\\tanh$. Jumlah parameter: $(D \\cdot H + H^2 + 2H)$.\n- **LSTM**: 4 gerbang aktif, pemisahan cell state. Jumlah parameter: $4 \\times (D \\cdot H + H^2 + 2H)$.\n- **GRU**: 2 gerbang aktif, keadaan tunggal. Jumlah parameter: $3 \\times (D \\cdot H + H^2 + 2H)$.\n\n---\n\n### Karakteristik Performa Praktis\n\n- **Kecepatan Komputasi**: Simple RNN $>$ GRU $>$ LSTM.\n- **Kapasitas Memori Jangka Panjang**: LSTM $\\approx$ GRU $\\gg$ Simple RNN.\n- **Efisiensi Parameter**: GRU menghemat tepat $25\\%$ parameter bobot gerbang dibandingkan LSTM pada dimensi terkalibrasi.",
        "notes": "Gunakan perbandingan tabular ini untuk merangkum efisiensi struktural ketiga model."
    },
    {
        "kind": "content",
        "title": "Hasil Eksperimen Komparasi Empiris (Hasil Uji CPU Nyata)",
        "body": "Pengujian terstandar pada dataset sekuensial teks sintetis (120 sampel, $T=12$, $V=40$, 8 Epoch):\n\n| Model | Jumlah Parameter | Waktu Pelatihan (ms) | Loss Akhir | Akurasi Val | Karakteristik Gradien |\n|---|:---:|:---:|:---:|:---:|:---:|\n| **Simple RNN** | 1.698 | 273.67 ms | 0.1084 | 70.00% | Fluktuatif pada sekuens panjang |\n| **LSTM** | 4.722 | 209.91 ms | 0.2881 | **80.00%** | Stabil melalui jalur Cell State |\n| **GRU** | 3.714 | 254.77 ms | 0.1746 | 67.50% | Konvergensi cepat, parameter hemat |\n\n*Catatan: LSTM membuktikan keunggulan akurasi validasi tertinggi (+10% atas RNN), sedangkan GRU membuktikan efisiensi parameter ~21% lebih hemat dibanding LSTM.*",
        "notes": "Paparkan data empiris nyata dari results_ch27.json untuk menunjukkan sains data berbasis bukti (evidence-based data science)."
    },
    {
        "kind": "content",
        "title": "Keterbatasan Alami Seluruh Model Rekuren (RNN-Family)",
        "body": "Meskipun LSTM dan GRU berhasil memitigasi vanishing gradients, seluruh keluarga model rekuren memiliki **hambatan arsitektural fundamental**:\n\n1. **Ketiadaan Komputasi Paralel (*Non-Parallelizable*)**: Perhitungan langkah waktu $t$ mutlak harus menunggu hasil waktu $t-1$. Pemrosesan sekuensial ini tidak dapat dimaksimalkan pada arsitektur perangkat keras modern berorientasi masif paralel (GPU/TPU).\n2. **Bottleneck Vektor Konteks Statis**: Seluruh informasi dari ratusan token masa lalu dipaksa dikompresi ke dalam satu vektor $\\mathbf{h}_T$ berdimensi tetap.\n3. **Titik Tolak Menuju Bab 28**: Keterbatasan inilah yang memicu revolusi **Attention Mechanism** dan arsitektur **Transformer** yang membuang rekursi secara utuh (*Attention Is All You Need*).",
        "notes": "Ini adalah jembatan intelektual paling penting yang mengantarkan mahasiswa ke Chapter 28 (Attention & Transformers)."
    },
    {
        "kind": "content",
        "title": "Responsible NLP: Bias Algoritmik & Keadilan Representasi",
        "body": "Penerapan model pemrosesan bahasa alami di ruang publik menuntut etika rekayasa yang ketat:\n\n- **Bias Gender & Stereotipe Sosial**: Model embedding statis yang dilatih pada korpus internet mentah kerap merefleksikan asosiasi timpang (misal analogi historis 'dokter' $\\leftrightarrow$ pria, 'perawat' $\\leftrightarrow$ wanita).\n- **Bias Geografis & Dialek**: Performa model sering kali mendegradasi saat memproses ragam bahasa non-standar, bahasa daerah, atau dialek lokal.\n- **Kerahasiaan Privasi Teks**: Teks masukan pengguna (nama lengkap, nomor induk kependudukan, rekam medis) wajib di-anonimkan sebelum digunakan sebagai data pelatihan model.",
        "notes": "Hubungkan dengan materi AI Ethics pada Bab 09 mengenai Responsible AI dan kerangka kerja keadilan algoritma."
    },
    {
        "kind": "content",
        "title": "Demonstrasi Praktik Python Chapter 27",
        "body": "Ekosistem praktik mandiri telah tersedia lengkap di repositori:\n\n- `text_preprocessing.py`: Pembersihan teks dan pelestarian kata negasi kritis.\n- `tokenization.py`: Pembangunan kosakata dan padding sekuens integer.\n- `bag_of_words.py` & `tfidf_representation.py`: Vektorisasi klasik terverifikasi numerik.\n- `ngram_features.py`: Penangkapan konteks frasa multi-kata.\n- `text_classification.py`: Pipeline sentimen bebas kebocoran data.\n- `word_embeddings.py` & `cosine_similarity.py`: Lookup vektor padat dan geometri kosinus.\n- `simple_rnn.py`, `lstm_sequence_model.py`, `gru_sequence_model.py`: Model neural rekuren PyTorch.\n- `rnn_lstm_gru_comparison.py`: Benchmarking komparatif otomatis tersimpan dalam JSON.",
        "notes": "Ajak mahasiswa membuka terminal dan menjalankan skrip demonstrasi satu per satu untuk melihat log eksekusi nyata."
    },
    {
        "kind": "content",
        "title": "Diskusi & Pertanyaan Reflektif Kelas",
        "body": "Diskusikan bersama rekan sekelas pertanyaan-pertanyaan berikut:\n\n1. Mengapa pada tugas analisis sentimen e-commerce, model sederhana TF-IDF + Logistic Regression sering kali menghasilkan akurasi yang kompetitif dibandingkan model LSTM yang kompleks?\n2. Dalam skenario apa operasi stemming bahasa Indonesia justru merusak akurasi pemahaman kata berimbuhan?\n3. Bagaimana hubungan antara ukuran batch (*batch size*) dengan kebutuhan token padding `<pad>` pada pemrosesan sekuens heterogen?\n4. Jika GRU menghemat parameter, pada kondisi komputasi dan data seperti apa LSTM tetap menjadi pilihan yang lebih superior?",
        "notes": "Gunakan pertanyaan reflektif ini untuk menguji pemahaman kritis mahasiswa terhadap kompromi rekayasa (engineering trade-offs)."
    },
    {
        "kind": "summary",
        "title": "Ringkasan Bab 27: Intisari Pembelajaran",
        "body": "### Poin Kunci yang Harus Diingat:\n\n1. **Data Preprocessing**: Wajib disesuaikan dengan domain; pelihara kata negasi untuk analisis sentimen.\n2. **TF-IDF**: Menggabungkan frekuensi lokal (TF) dengan kelangkaan global (IDF) disertai normalisasi L2.\n3. **Word Embeddings**: Memproyeksikan token ke ruang vektor padat kontinu di mana kedekatan geometris mencerminkan kemiripan kontekstual.\n4. **Recurrent Models**: Simple RNN memproses teks sekuensial namun terhambat vanishing gradients; LSTM memecahkannya via Cell State linier dan 3 gerbang; GRU menyederhanakannya menjadi 2 gerbang efisien.\n5. **Arah Selanjutnya**: Kebutuhan paralelisasi komputasi dan pemodelan dependensi tanpa batas jarak membawa kita ke arsitektur **Attention & Transformer** pada Bab 28.",
        "notes": "Rangkum seluruh intisari bab secara lugas sebelum menutup sesi presentasi."
    },
    {
        "kind": "content",
        "title": "Referensi Akademik & Bacaan Lanjutan",
        "body": "Daftar pustaka primer terverifikasi untuk pendalaman materi:\n\n1. **Jurafsky, D., & Martin, J. H.** (2024). *Speech and Language Processing* (3rd ed. draft). Pearson / Stanford University.\n2. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning* (Chapter 10: Sequence Modeling). MIT Press.\n3. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.\n4. **Cho, K., et al.** (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP 2014*, 1724–1734.\n5. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval* (Chapter 6: Scoring, Term Weighting and the Vector Space Model). Cambridge University Press.",
        "notes": "Tutup sesi dengan memotivasi mahasiswa untuk membaca bab 10 buku Goodfellow et al. dan draf terbaru Jurafsky & Martin."
    }
]

out_dir = Path(__file__).resolve().parent
out_file = out_dir / "deck.json"

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(slides, f, indent=2, ensure_ascii=False)

print(f"Berhasil menghasilkan {len(slides)} slide Reveal.js di: {out_file}")

