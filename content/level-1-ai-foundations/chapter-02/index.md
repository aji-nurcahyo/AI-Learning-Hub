# Chapter 02: History, Evolution & AI Paradigms

> **Tingkat Keahlian**: Level 1 — AI Foundations  
> **Jalur Spesialisasi**: Core Curriculum  
> **Prasyarat**: Chapter 01 (Introduction to Artificial Intelligence)  
> **Format Pembelajaran**: Modul Bacaan Teori, Latihan Analisis, dan Tugas Proyek Komparatif  

---

## 1. Mengapa Sejarah AI Penting Dipelajari

Perkembangan Artificial Intelligence (AI) sering kali dipahami secara keliru sebagai lompatan teknologi yang tiba-tiba terjadi dalam satu dekade terakhir. Banyak praktisi pemula menganggap era AI baru dimulai saat *Deep Learning* mendominasi pengenalan citra pada tahun 2012 atau ketika *Large Language Models* (LLM) diperkenalkan ke publik secara luas pada tahun 2022. Pemahaman yang terputus dari konteks sejarah ini menimbulkan ilusi bahwa pendekatan berbasis data (*data-driven*) adalah satu-satunya metode yang valid, serta mengabaikan fondasi teoretis yang telah dibangun selama lebih dari tujuh dekade.

Mempelajari sejarah dan evolusi paradigma AI memberikan empat manfaat strategis bagi ilmuwan komputer dan praktisi sistem cerdas:

1. **Memahami Alasan di Balik Pilihan Pendekatan**: Setiap paradigma komputasi cerdas lahir sebagai respons langsung terhadap kegagalan atau keterbatasan paradigma pendahulunya. Mengetahui kelemahan masa lalu mencegah kita mengulangi kesalahan metodologis yang sama.
2. **Menghindari Jebakan Siklus Hype dan Kekecewaan (*AI Winter*)**: Sejarah AI diwarnai oleh gelombang optimisme berlebihan (*hype*) yang disusul oleh pemotongan pendanaan drastis akibat janji teknologi yang tidak terbukti. Pemahaman historis membentuk ekspektasi realistis mengenai kemampuan aktual suatu sistem.
3. **Mengidentifikasi Masalah yang Tepat untuk Alat yang Tepat**: Tidak semua masalah di dunia nyata membutuhkan *Deep Learning* yang rakus komputasi. Masalah dengan aturan regulasi ketat dan logika pasti sering kali jauh lebih efisien dan terverifikasi bila diselesaikan menggunakan sistem berbasis logika formal (*Symbolic AI*).
4. **Membaca Arah Evolusi Masa Depan**: Tren modern menuju pendekatan hibrida (*Neuro-Symbolic AI*) pada dasarnya merupakan sintesis rekonsiliasi antara kekuatan penalaran simbolik klasik dengan ketangguhan persepsi pemelajaran mendalam modern.

Perkembangan AI bukanlah garis lurus tunggal yang linear di mana satu teknologi memusnahkan teknologi sebelumnya, melainkan sebuah dialektika yang melibatkan pergeseran paradigma, kompetisi metodologis, dan konvergensi teknologi.

---

## 2. Fondasi Awal Komputasi dan Gagasan tentang Intelligence

Gagasan tentang mesin yang mampu bernalar dan meniru kecerdasan manusia telah berakar jauh sebelum komputer elektronik pertama diciptakan. Tiga pilar utama yang menjadi landasan filosofis dan matematis AI adalah:

- **Filsafat Budi dan Logika Deduktif**: Berawal dari konsep silogisme Aristoteles yang merumuskan aturan penarikan kesimpulan yang valid secara mekanis, berkembang ke gagasan Thomas Hobbes bahwa *"reasoning is but reckoning"* (menalar tidak lain adalah mengkalkulasi), hingga cita-cita Gottfried Wilhelm Leibniz tentang *Characteristica Universalis*—bahasa formal universal yang mampu menyelesaikan perselisihan rasional layaknya perhitungan aritmatika.
- **Formalisasi Logika Matematika**: Pada abad ke-19 dan awal abad ke-20, George Boole memperkenalkan aljabar biner (*Boolean Algebra*), disusul oleh Gottlob Frege yang membangun logika predikat tingkat pertama (*First-Order Predicate Logic*). Formalisasi ini memungkinkan representasi pengetahuan eksplisit dalam bentuk proposisi yang dapat dimanipulasi secara algoritmik.
- **Teori Komputabilitas Formal (1936)**: Alan Turing merumuskan konsep *Turing Machine*, membuktikan secara matematis bahwa sebuah mesin manipulasi simbol abstrak sederhana mampu mengeksekusi algoritma apa pun yang dapat dihitung (*computable function*). Ini meletakkan dasar bahwa komputasi tidak bergantung pada materi biologis, melainkan pada struktur pemrosesan informasi.

Pada ranah neurologi komputasional, Warren McCulloch dan Walter Pitts (1943) mempublikasikan model matematika pertama untuk jaringan saraf tiruan (*Artificial Neural Networks*). Mereka membuktikan bahwa neuron biologis sederhana yang diidealisasikan sebagai sakelar ambang biner (*threshold logic unit*) dapat mengombinasikan logika biner AND, OR, dan NOT. Hal ini membuktikan bahwa mekanisme penalar logis dapat diimplementasikan di atas struktur yang meniru jaringan biologis.

---

## 3. Alan Turing dan Pertanyaan tentang Machine Intelligence

Pada tahun 1950, Alan Mathison Turing mempublikasikan makalah seminal di jurnal filosofis *Mind* yang berjudul **"Computing Machinery and Intelligence"**. Turing membuka tulisannya dengan pertanyaan provokatif yang menjadi tonggak kelahiran kecerdasan mesin:

$$\text{"I propose to consider the question, 'Can machines think?'"}$$

Menyadari bahwa istilah *"think"* (berpikir) dan *"machine"* (mesin) memiliki definisi bahasa alami yang kabur dan sarat bias antroposentris, Turing mengganti pertanyaan filosofis tersebut dengan pengujian operasional terukur yang dikenal sebagai **The Imitation Game** (kini populer sebagai **Turing Test**).

```
+-------------------------------------------------------------+
|                        TURING TEST                          |
|                                                             |
|   +-----------------------+     +-----------------------+   |
|   |   Entitas A (Mesin)   |     |  Entitas B (Manusia)  |   |
|   +-----------------------+     +-----------------------+   |
|               ^                             ^               |
|               |  Saluran Teks Terisolasi    |               |
|               v                             v               |
|   +-----------------------------------------------------+   |
|   |              Interogator C (Manusia)                |   |
|   |         "Apakah A atau B yang merupakan mesin?"     |   |
|   +-----------------------------------------------------+   |
+-------------------------------------------------------------+
```

### Karakteristik dan Kritik Turing Test

1. **Uji Perilaku Eksternal (*Behavioral Equivalence*)**: Turing Test tidak memedulikan apakah mesin memiliki kesadaran subjektif (*consciousness* atau *qualia*), melainkan apakah keluaran perilakunya (*indistinguishable input-output behavior*) dapat dibedakan dari manusia oleh pengamat rasional.
2. **Keterbatasan Operasional**: Tes ini mengukur kemampuan meniru respons manusia, termasuk kelemahan manusia (seperti berbohong, salah mengetik, atau keterlambatan berhitung). Sebaliknya, sebuah mesin yang mampu menghitung triliunan operasi per detik secara sempurna justru dapat gagal dalam Turing Test karena terlalu tidak manusiawi.
3. **Argumen Chinese Room (John Searle, 1980)**: Filsuf John Searle mengajukan eksperimen pikiran *Chinese Room* untuk mengkritik Turing Test. Searle berargumen bahwa manipulasi simbol berbasis aturan sintaksis (meskipun menghasilkan respons bahasa Mandarin yang fasih) sama sekali tidak membuktikan adanya pemahaman semantik (*intentionality* atau *genuine understanding*).

---

## 4. Dartmouth Summer Research Project 1956

AI secara resmi diakui sebagai disiplin ilmiah mandiri pada musim panas tahun 1956 dalam sebuah lokakarya intensif di Dartmouth College, Hanover, New Hampshire. Proposal lokakarya ini diajukan pada tahun 1955 oleh empat tokoh pelopor:

1. **John McCarthy** (Dartmouth College, kemudian Stanford University) — pencipta istilah *"Artificial Intelligence"* dan bahasa pemrograman Lisp.
2. **Marvin L. Minsky** (Harvard University, kemudian MIT) — pionir komputasi kognitif dan perseptual.
3. **Nathaniel Rochester** (IBM) — arsitek komputer komersial IBM 701.
4. **Claude E. Shannon** (Bell Telephone Laboratories) — bapak teori informasi matematis (*Information Theory*).

Konjektur fundamental yang mendasari proposal Dartmouth adalah:

> *"Every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it."*

Lokakarya ini mempertemukan tokoh-tokoh terkemuka lainnya, termasuk Allen Newell dan Herbert Simon dari Carnegie Mellon University, yang membawa perangkat lunak **Logic Theorist**. Program ini sering disebut sebagai program AI pertama di dunia karena mampu membuktikan 38 dari 52 teorema logika formal dalam buku *Principia Mathematica* karya Alfred North Whitehead dan Bertrand Russell.

---

## 5. Perkembangan Symbolic AI (1956–1974)

Dua dekade pertama setelah lokakarya Dartmouth didominasi oleh paradigma **Symbolic AI**, yang kelak dijuluki oleh John Haugeland sebagai **GOFAI** (*Good Old-Fashioned Artificial Intelligence*). 

Paradigma ini beroperasi di atas premis bahwa kecerdasan diwujudkan melalui manipulasi simbol eksplisit berdasarkan aturan formal logika. Allen Newell dan Herbert Simon (1976) merumuskan fondasi ini dalam **Physical Symbol System Hypothesis (PSSH)**:

$$\text{"A physical symbol system has the necessary and sufficient means for general intelligent action."}$$

### Karakteristik Symbolic AI
- **Representasi Eksplisit**: Fakta dan hubungan dinyatakan secara simbolik (misalnya: `Manusia(Socrates)`, `Fana(x) :- Manusia(x)`).
- **Proses Berbasis Pencarian dan Inferensi**: Masalah dirumuskan sebagai pencarian jalur (*search problem*) di dalam ruang keadaan (*state space*) menggunakan pohon keputusan atau pembuktian teorema deduktif.
- **Keterpisahan Data dan Logika**: Program memisahkan basis pengetahuan (*knowledge base*) dari mesin inferensi (*inference engine*).

Pencapaian utama pada era ini meliputi:
- **General Problem Solver (GPS)** (Newell & Simon, 1959): Pendekatan pemecahan masalah umum menggunakan analisis tujuan-akhir (*Means-Ends Analysis*).
- **ELIZA** (Joseph Weizenbaum, 1966): Program pemrosesan bahasa alami awal yang mensimulasikan psikoterapis Rogerian menggunakan pencocokan pola teks (*pattern matching*) sederhana.
- **SHRDLU** (Terry Winograd, 1970): Agen simulasi mikrodunia balok (*blocks world*) yang mampu memahami perintah bahasa alami, bernalar tentang posisi spasial, dan memanipulasi balok secara logis.

---

## 6. Logic-Based Systems dan Knowledge Representation

Keberhasilan dalam mikrodunia (*toy problems*) mendorong para peneliti untuk mengembangkan formalisme representasi pengetahuan yang lebih tangguh guna menghadapi dunia nyata. Representasi pengetahuan (*Knowledge Representation*) menjadi jantung penelitian Symbolic AI.

```
+-------------------------------------------------------------------+
|               KNOWLEDGE REPRESENTATION SPECTRUM                   |
|                                                                   |
|   Logika Formal          Jaringan Semantik           Frame &      |
|   (First-Order Logic)    (Semantic Networks)         Ontologi     |
|                                                                   |
|   ∀x (Bird(x) ∧          [Canary] --is-a--> [Bird]   Class: Bird  |
|       ¬Penguin(x)              \              /       HasFeathers: |
|       → CanFly(x))              --color--> [Yellow]    Default=True|
+-------------------------------------------------------------------+
```

Tiga paradigma utama representasi pengetahuan meliputi:

1. **Logika Predikat Tingkat Pertama (*First-Order Logic*)**: Menyediakan sintaksis dan semantik matematis yang ketat untuk menyatakan objek, predikat, fungsi, dan kuantor ($\forall, \exists$). Inferensi dijalankan menggunakan prinsip resolusi Robinson (*resolution theorem proving*).
2. **Jaringan Semantik (*Semantic Networks*)**: Representasi graf berarah di mana node melambangkan konsep atau objek, dan edge melambangkan relasi biner (misalnya relasi taksonomi `is-a` atau komposisi `part-of`), memungkinkan mekanisme pewarisan sifat (*property inheritance*).
3. **Sistem Frame dan Ontologi Formal**: Diperkenalkan oleh Marvin Minsky (1975), *frame* mengelompokkan pengetahuan terkait suatu konsep ke dalam struktur data berbasis slot dan nilai default. Konsep ini menjadi cikal bakal pemrograman berorientasi objek dan ontologi web semantik (OWL).

---

## 7. Expert Systems (1970–1980an)

Pada akhir dekade 1970-an, terjadi pergeseran filosofis besar dari pencarian metode umum (*general-purpose problem solvers*) menuju sistem berbasis domain pengetahuan spesifik (*domain-specific knowledge*). Edward Feigenbaum mempelopori pendekatan bahwa:

$$\text{"Knowledge is power: Knowledge, not clever inference algorithms, produces high performance."}$$

Lahirnya **Expert Systems** (Sistem Pakar) menandai era komersialisasi pertama AI dalam sejarah. Sistem pakar adalah perangkat lunak komputer cerdas yang memanfaatkan pengetahuan dan prosedur inferensi untuk memecahkan masalah kompleks yang biasanya membutuhkan keahlian manusia tingkat tinggi.

### Arsitektur Sistem Pakar Klasik

```
+-------------------------------------------------------------+
|                 ARSITEKTUR EXPERT SYSTEM                    |
|                                                             |
|   +--------------------+          +---------------------+   |
|   |    Pakar Manusia   |          |    Pengguna Akhir   |   |
|   +--------------------+          +---------------------+   |
|             |                                ^              |
|             v                                | (Pertanyaan &|
|   +--------------------+                     |  Rekomendasi)|
|   | Knowledge Engineer |                     v              |
|   +--------------------+          +---------------------+   |
|             |                     |   User Interface    |   |
|             v                     +---------------------+   |
|   +--------------------+                     ^              |
|   |   Knowledge Base   | <----------------+  |              |
|   | (Kaidah IF-THEN)   |                  |  v              |
|   +--------------------+          +---------------------+   |
|                                   |  Inference Engine   |   |
|                                   | (Forward / Backward)|   |
|                                   +---------------------+   |
|                                              ^              |
|                                              v              |
|                                   +---------------------+   |
|                                   | Explanation Facility|   |
|                                   +---------------------+   |
+-------------------------------------------------------------+
```

### Studi Kasus Sistem Pakar Sukses
- **DENDRAL** (1965–1970an, Stanford): Mengidentifikasi struktur molekul kimia tak dikenal berdasarkan spektrometri massa.
- **MYCIN** (1972–1976, Stanford): Mendiagnosis infeksi bakteri darah meningitis dan merekomendasikan antibiotik dengan faktor kepastian (*certainty factors*). Evaluasi empiris menunjukkan akurasinya setara atau melampaui dokter spesialis junior.
- **R1 / XCON** (1978–1980an, DEC): Sistem konfigurasi pesanan komputer mainframe VAX komersial yang menghemat jutaan dolar per tahun bagi Digital Equipment Corporation.

---

## 8. Keterbatasan Pendekatan Symbolic

Meskipun sukses dalam domain tertutup dengan aturan eksplisit, Symbolic AI menghadapi tembok keterbatasan mendasar saat dihadapkan pada ketidakpastian dunia nyata:

1. **Knowledge Acquisition Bottleneck**: Proses mengekstraksi pengetahuan tacit dari pakar manusia dan memformulasikannya menjadi ratusan ribu kaidah `IF-THEN` manual membutuhkan waktu sangat panjang, rentan konflik inkonsistensi kaidah, dan sulit dipelihara.
2. **Kerapuhan (*Brittleness*)**: Sistem berbasis kaidah sangat rapuh. Jika sebuah masukan berada sedikit saja di luar cakupan kaidah yang tertulis, sistem tidak mengalami degradasi performa yang halus (*graceful degradation*), melainkan langsung gagal total (*catastrophic failure*).
3. **Ketidakmampuan Menangani Persepsi Kontinu**: Logika diskret sangat buruk dalam memproses sinyal fisik kontinu berderau (*noisy analog signals*), seperti gelombang suara manusia atau matriks piksel kamera video.
4. **Moravec's Paradox (1988)**: Ditemukan oleh Hans Moravec, Rodney Brooks, dan Marvin Minsky, paradoks ini menyatakan:
   > *"Sangat mudah membuat komputer menunjukkan kemampuan berpikir tingkat tinggi pada level orang dewasa (seperti bermain catur atau membuktikan logika), tetapi hampir mustahil memberi komputer kemampuan persepsi dan motorik anak balita berumur satu tahun (seperti mengenali wajah ibu, berjalan tanpa jatuh, atau membedakan suara kucing)."*

---

## 9. AI Winter dan Faktor Penyebabnya

Sejarah AI mencatat dua periode kemunduran besar yang dikenal sebagai **AI Winter**, di mana antusiasme meredup, liputan media menjadi skeptis, dan lembaga pendanaan menghentikan hibah penelitian secara drastis.

```
   Ekspektasi & Pendanaan
      ^
      |     Golden Era I          Golden Era II (Expert Systems)     Deep Learning Era
      |       (1956-1973)                  (1980-1987)                   (2012-kini)
      |         /\                            /\                              /\
      |        /  \                          /  \                            /
      |       /    \                        /    \                          /
      |      /      \                      /      \                        /
      |     /        \                    /        \                      /
      |    /          \                  /          \                    /
      |   /   AI Winter I               /   AI Winter II                /
      |  /    (1974-1980)              /    (1987-1993)                /
      | /     (Lighthill Report)      /     (Lisp Market Collapse)    /
      +------------------------------------------------------------------------> Waktu
```

### AI Winter I (1974–1980)
- **Pemicu Ilmiah**: Terbitnya buku *Perceptrons* karya Marvin Minsky dan Seymour Papert (1969) yang membuktikan keterbatasan perceptron lapis tunggal yang tidak mampu menyelesaikan fungsi logika XOR.
- **Pemicu Kebijakan & Pendanaan**: Laporan **Lighthill Report** (1973) di Inggris Raya oleh Sir James Lighthill yang menyimpulkan bahwa AI gagal mencapai tujuan spektakulernya dan hanya berkutat pada *combinatorial explosion*. Di Amerika Serikat, DARPA memangkas pendanaan riset dasar murni setelah pengesahan Mansfield Amendment.

### AI Winter II (1987–1993)
- **Runtuhnya Pasar Perangkat Keras Khusus**: Industri sistem pakar sangat bergantung pada komputer khusus berharga mahal (seperti *Lisp Machines* dari Symbolics dan LMI). Munculnya workstation umum berbasis prosesor komersial murah dari Sun Microsystems dan IBM PC membuyarkan model bisnis vendor Lisp machine.
- **Kegagalan Proyek Ambisius**: Proyek *Fifth Generation Computer Systems* (FGCS) di Jepang yang menargetkan komputasi pemrosesan pengetahuan paralel masif berbasis logika Prolog gagal mencapai target komersial setelah menghabiskan ratusan juta dolar.

---

## 10. Perkembangan Statistical AI dan Machine Learning (1990–2000an)

Keluarnya AI dari musim dingin kedua ditandai oleh perubahan paradigma epistemologis yang radikal: **beralih dari merekayasa pengetahuan aturan secara manual (*knowledge engineering*) menjadi memelajari pola secara otomatis dari data (*statistical learning*)**.

Alih-alih menyuruh pakar menulis aturan manual, ilmuwan komputer merumuskan AI sebagai masalah inferensi probabilistik dan optimasi matematis:

$$P(Y \mid X) = \frac{P(X \mid Y) P(Y)}{P(X)}$$

Karakteristik Statistical AI meliputi:
- **Pemanfaatan Teori Probabilitas**: Menggantikan logika benar/salah biner dengan kalkulus ketidakpastian (*Bayesian Networks*, *Hidden Markov Models*).
- **Pendekatan Induktif**: Menggunakan algoritma pemelajaran untuk mengekstrak fungsi pemetaan $f: X \to Y$ dari sampel data historis.
- **Algoritma Machine Learning Klasik yang Kokoh**: Era ini melahirkan algoritma yang memiliki landasan teoretis kuat (*statistical learning theory* oleh Vladimir Vapnik):
  - *Support Vector Machines* (SVM) untuk klasifikasi margin maksimum.
  - *Random Forest* dan ensemble methods (*AdaBoost*, *Gradient Boosting*).
  - *Naive Bayes* untuk penyaringan email spam dan klasifikasi teks.
- **Standar Evaluasi Empiris yang Ketat**: Munculnya pemisahan dataset menjadi *training*, *validation*, dan *test set*, serta metrik performa objektif (Accuracy, Precision, Recall, ROC-AUC).

---

## 11. Peran Dataset, Computational Power, dan Hardware Acceleration

Transisi menuju era AI modern tidak didorong oleh penemuan algoritma baru semata, melainkan oleh konvergensi harmonis dari tiga faktor pendorong (*The Holy Trinity of Modern AI*):

```
                       +-----------------------+
                       |      BIG DATA         |
                       |  (ImageNet, Web Text) |
                       +-----------------------+
                                   ^
                                   |
                                   v
+-----------------------+                    +-----------------------+
|  ALGORITMA MODERN     | <================> | HARDWARE ACCELERATION |
| (Backprop, ReLU, Adam)|                    |  (NVIDIA GPU, TPU)    |
+-----------------------+                    +-----------------------+
```

1. **Big Data dan Kurasi Dataset Skala Besar**: Sebelum tahun 2000-an, dataset visual hanya berisi ratusan gambar beresolusi rendah. Pada tahun 2009, Fei-Fei Li dan timnya di Princeton/Stanford merilis **ImageNet**, basis data anotasi visual raksasa berisi lebih dari 14 juta gambar yang dikategorikan ke dalam 20.000 lebih kelas WordNet. ImageNet Large Scale Visual Recognition Challenge (ILSVRC) menjadi tolak ukur benchmark kompetitif dunia.
2. **Perangkat Keras Terakselerasi Paralel (GPU)**: Algoritma *Neural Network* pada dasarnya adalah komputasi perkalian matriks densitas tinggi (*dense matrix multiplication*). *Graphics Processing Units* (GPU) yang awalnya dirancang untuk merender grafika game 3D ternyata jutaan kali lebih efisien mengeksekusi operasi tensor paralel dibandingkan CPU serial konvensional.
3. **Optimasi Algoritmik Baru**: Penemuan teknik seperti fungsi aktivasi non-saturasi *Rectified Linear Unit* (ReLU), regularisasi *Dropout*, dan teknik *Batch Normalization* berhasil menuntaskan masalah klasik *vanishing gradient* pada jaringan saraf berkedalaman tinggi.

---

## 12. Kebangkitan Neural Networks dan Deep Learning (2012)

Meskipun algoritma *backpropagation* telah dipopulerkan oleh Rumelhart, Hinton, dan Williams pada tahun 1986, jaringan saraf tiruan tertidur selama lebih dari dua dekade karena ketidakcukupan data dan daya komputasi saat itu.

Titik balik historis yang mengubah wajah ilmu komputer terjadi pada bulan Oktober **2012** dalam kompetisi ImageNet (ILSVRC 2012). Model jaringan konvolusi dalam (*Deep Convolutional Neural Network*) bernama **AlexNet**, yang dirancang oleh Alex Krizhevsky, Ilya Sutskever, dan Geoffrey Hinton dari University of Toronto, memenangkan kompetisi dengan margin spektakuler.

AlexNet mencatat *top-5 error rate* sebesar **15.3%**, memangkas lebih dari 10.8 poin persentase dibandingkan metode *state-of-the-art* berbasis rekayasa fitur manual (*hand-crafted features* seperti SIFT/HOG + SVM) yang berada di angka 26.2%.

Kemenangan ini menandai runtuhnya dominasi rekayasa fitur manual dan meresmikan dimulainya era **Deep Learning**: model yang mampu memelajari representasi hierarkis fitur secara otomatis dari data mentah (*end-to-end representation learning*), mulai dari tepi piksel tingkat rendah, tekstur, bagian objek, hingga representasi semantik tingkat tinggi.

---

## 13. Perkembangan Computer Vision dan Natural Language Processing

Pasca-2012, *Deep Learning* menyapu bersih berbagai subbidang kecerdasan buatan:

### Evolusi Computer Vision (CV)
- **Arsitektur CNN Bertingkat**: Dari AlexNet (8 lapis, 2012), VGGNet (16–19 lapis, 2014), GoogLeNet/Inception (22 lapis, 2014), hingga **ResNet** (Kaiming He et al., 2015) yang memperkenalkan *Residual Connections* (*skip connections*) sehingga memungkinkan pelatihan jaringan saraf hingga 152 lapis tanpa degradasi gradien. ResNet mencatatkan error 3.57%, melampaui kemampuan estimasi kesalahan manusia (~5%).
- **Object Detection & Segmentation**: Berkembangnya arsitektur seperti Faster R-CNN, YOLO (*You Only Look Once*), dan Mask R-CNN yang mampu mengenali lokasi koordinat *bounding box* dan segmentasi piksel secara *real-time*.

### Evolusi Natural Language Processing (NLP)
- **Representasi Vektor Terdistribusi**: Tomas Mikolov et al. (2013) di Google merilis **Word2Vec**, membuktikan bahwa hubungan semantik kata dapat direpresentasikan secara geometris dalam ruang vektor kontinu ($\text{vector}(\text{"King"}) - \text{vector}(\text{"Man"}) + \text{vector}(\text{"Woman"}) \approx \text{vector}(\text{"Queen"})$).
- **Jaringan Rekuren untuk Data Sekuensial**: Dominasi arsitektur *Recurrent Neural Networks* (RNN) dan *Long Short-Term Memory* (LSTM) untuk memproses teks dan deret waktu (*time series*), meskipun menghadapi kendala lambatnya komputasi serial dan hilangnya konteks kalimat panjang (*bottleneck problem*).

---

## 14. Transformer dan Large Language Models (2017–kini)

Pada bulan Juni 2017, tim peneliti Google Brain mempublikasikan paper monumental berjudul **"Attention Is All You Need"** (Vaswani et al., 2017). Paper ini memperkenalkan arsitektur **Transformer** yang sepenuhnya membuang rekurensi dan konvolusi.

```
+-------------------------------------------------------------+
|               TRANSFORMER: SELF-ATTENTION                   |
|                                                             |
|   "Bank sentral menaikkan suku bunga untuk menahan inflasi" |
|        |                                          |         |
|        +------------------------------------------+         |
|              Koneksi Perhatian Langsung                     |
|                                                             |
|   - Menghitung bobot relevansi antarkata secara paralel.    |
|   - Kompleksitas jarak jalur informasi: O(1).               |
|   - Skalabilitas komputasi paralel masif pada GPU cluster.  |
+-------------------------------------------------------------+
```

Transformer mengatasi dua kelemahan fundamental RNN:
1. **Pemrosesan Paralel Penuh**: Seluruh token dalam kalimat diproses serentak, bukan satu demi satu, memungkinkan pelatihan pada miliaran korpus teks web menggunakan kluster komputasi terdistribusi.
2. **Mekanisme Self-Attention**: Menghitung bobot interaksi langsung antarsepasang token tanpa memedulikan jarak fisik posisinya dalam dokumen, mempertahankan ketergantungan konteks jarak jauh (*long-range dependencies*).

Inovasi ini memicu ledakan model bahasa:
- **BERT** (Google, 2018): *Bidirectional Encoder Representations from Transformers* untuk pemahaman semantik teks (*masked language modeling*).
- **GPT Series** (OpenAI, 2018–kini): Arsitektur *autoregressive decoder-only* berbasis *causal language modeling* yang mampu menghasilkan teks koheren tingkat tinggi.

---

## 15. Generative AI dan Foundation Models

Pada tahun 2021, Stanford Institute for Human-Centered Artificial Intelligence (HAI) mencetuskan istilah **Foundation Models** untuk mendeskripsikan model AI berskala masif yang dilatih pada korpus data yang sangat luas dan beragam menggunakan metode nirpengawasan (*self-supervised learning*), yang kemudian dapat diadaptasi (*fine-tuned*) untuk ribuan tugas hilir yang berbeda.

```
                            RAW UNLABELED DATA
                     (Teks Web, Buku, Kode, Gambar)
                                   |
                                   v
                    +-----------------------------+
                    |      FOUNDATION MODEL       |
                    | (GPT-4, Gemini, Claude, LLaMA|
                    +-----------------------------+
                                   |
           +-----------------------+-----------------------+
           |                       |                       |
           v                       v                       v
     Penerjemahan             Penulisan Kode           Analisis Medis
```

### Paradigma Pemelajaran Modern
1. **Self-Supervised Pre-Training**: Model memprediksi token berikutnya (*next-token prediction*) atau mengisi bagian yang disembunyikan (*denoising objective*) tanpa memerlukan label manual manusia.
2. **Alignment & Human Preference**: Proses penyelarasan perilaku model agar aman, jujur, dan membantu melalui teknik *Instruction Tuning* dan *Reinforcement Learning from Human Feedback* (RLHF).
3. **Emergent Abilities**: Fenomena di mana kemampuan kognitif tingkat lanjut (seperti penalaran multi-langkah *Chain-of-Thought*, pemecahan teka-teki logika, dan penerjemahan antarbahasa asing) muncul secara spontan saat parameter model diskalakan melampaui ambang batas tertentu (*Scaling Laws*).
4. **Multimodalitas**: Integrasi penglihatan (*vision*), suara (*audio*), teks, dan kode ke dalam satu arsitektur terpadu.

---

## 16. Perkembangan Pendekatan Hybrid dan Neuro-Symbolic AI

Meskipun *Foundation Models* memperlihatkan kefasihan bahasa yang luar biasa, model berbasis probabilitas statistik murni memiliki kelemahan mendasar:
- **Halusinasi (*Hallucination*)**: Menghasilkan pernyataan yang terdengar meyakinkan tetapi secara faktual salah atau tidak logis.
- **Ketidakmampuan Verifikasi Ketat**: Tidak adanya jaminan formal bahwa keluaran model mematuhi aturan hukum, etika, atau batasan matematis 100%.
- **Opasitas (*Black-Box Nature*)**: Sulitnya melacak jalur penalaran secara pasti (*lack of interpretability*).

Sebagai solusi terhadap dikotomi ini, komunitas riset modern kembali melirik integrasi antara kedua mazhab besar melalui **Neuro-Symbolic AI**:

$$\text{Neuro-Symbolic AI} = \underbrace{\text{Neural Networks}}_{\text{Persepsi, Pembelajaran dari Data, Ketahanan Noise}} + \underbrace{\text{Symbolic Reasoning}}_{\text{Logika Eksplisit, Pengetahuan Terstruktur, Verifikasi Formal}}$$

Contoh penerapan nyata mencakup:
- **Knowledge-Graph Augmented Generation (KAG / RAG)**: Mengaitkan LLM dengan basis data grafik ontologi formal untuk mencegah halusinasi data faktual.
- **Sistem Pembuktian Teorema Otomatis (misal: AlphaGeometry)**: Menggunakan model saraf untuk memandu pencarian langkah kreatif dan mesin deduksi simbolik untuk mengeksekusi bukti matematika geometris tanpa cela.
- **Safe Autonomous Vehicles**: Menggunakan deep neural network untuk persepsi visual objek di jalan, tetapi menggunakan verifikasi aturan formal logis (*safety envelope*) untuk mengeksekusi rem darurat.

---

## 17. Perbandingan Paradigma AI

Tabel berikut merangkum karakteristik utama empat paradigma besar dalam sejarah kecerdasan buatan:

| Kriteria | Symbolic AI (GOFAI) | Statistical Machine Learning | Deep Learning | Neuro-Symbolic AI |
|---|---|---|---|---|
| **Era Dominan** | 1956–1980an | 1990an–2010an | 2012–kini | Kontemporer (Masa Depan) |
| **Sumber Utama Pengetahuan** | Aturan eksplisit pakar manusia | Pola induktif dari fitur tabular | Representasi berjenjang dari data mentah | Sinergi data mentah dan basis ontologi formal |
| **Representasi Internal** | Simbol diskret, predikat logika, pohon kaidah | Bobot matematis, vektor fitur berdimensi tetap | Tensor berdimensi sangat tinggi, bobot terdistribusi | Graf pengetahuan + representasi vektor laten |
| **Penanganan Noise / Data Kontinu** | Sangat rentan, mudah mengalami kegagalan | Cukup baik dengan teori probabilitas | Sangat tangguh dan tahan derau sinyal | Sangat tangguh dengan proteksi kaidah |
| **Kebutuhan Data Latih** | Rendah (cukup beberapa pakar) | Sedang (ribuan data tabular terlabel) | Ekstrem tinggi (jutaan hingga triliunan token) | Efisien (pengetahuan mengurangi data latih) |
| **Kebutuhan Komputasi** | Rendah (CPU standar) | Sedang (CPU / server tunggal) | Masif (Kluster GPU/TPU intensif) | Fleksibel (kombinasi modul saraf & inferensi) |
| **Tingkat Keterjelasan (*Explainability*)** | Sangat tinggi (*White-box*, jejak inferensi jelas) | Cukup tinggi (koefisien bobot, signifikansi statistik) | Sangat rendah (*Black-box*, opasitas representasi) | Tinggi pada modul logika, terpantau pada modul saraf |
| **Kasus Penggunaan Terbaik** | Sistem audit pajak, verifikasi protokol keamanan | Penilaian kelayakan kredit, deteksi churn nasabah | Klasifikasi citra medis, terjemahan mesin, dialog teks | Diagnostik medis kritis, robotika otonom bersertifikasi |

---

## 18. Pelajaran Berharga dari Sejarah Perkembangan AI

Tujuh dekade perjalanan kecerdasan buatan menyisakan pelajaran fundamental bagi akademisi dan praktisi:

1. **Klaim Berlebihan Selalu Membawa Bencana**: Prediksi prematur seperti *"komputer akan menggantikan semua pekerjaan manusia dalam 10 tahun"* yang pernah digaungkan pada era 1960-an dan era sistem pakar 1980-an terbukti merusak kredibilitas bidang keilmuan. Kejujuran intelektual mengenai batasan teknologi saat ini adalah benteng pertahanan terbaik melawan AI Winter baru.
2. **Tidak Ada Paradigma Tunggal yang Sempurna**: Mengagungkan satu paradigma sebagai solusi untuk seluruh jenis permasalahan adalah kesesatan metodologis (*Law of the Instrument / Maslow's Hammer*).
3. **Data dan Komputasi Adalah Pengubah Permainan**: Banyak ide algoritma brilian di masa lalu (seperti *perceptron multi-layer* atau *recurrent networks*) gagal bukan karena konsepnya cacat, melainkan karena mereka lahir terlalu dini sebelum ketersediaan data masif dan prosesor paralel yang memadai.
4. **Kecerdasan Memerlukan Persepsi dan Penalaran Sekaligus**: Manusia tidak hanya memproses dunia melalui kalkulasi probabilitas data visual (persepsi mirip *Deep Learning*), tetapi juga mampu merenungkan hukum sebab-akibat, mematuhi norma sosial tertulis, dan melakukan koreksi penalaran logis (penalaran mirip *Symbolic AI*).

---

## 19. Ringkasan Modul

- **Sejarah AI** bermula dari formalisasi logika matematis dan konseptualisasi mesin komputasi oleh Alan Turing (1950), sebelum resmi ditegakkan sebagai disiplin akademis mandiri pada **Dartmouth Workshop 1956**.
- **Symbolic AI (GOFAI)** mendominasi periode 1956–1980-an dengan pendekatan rekayasa aturan manual (*Expert Systems*), namun terhambat oleh *Knowledge Acquisition Bottleneck*, kerapuhan (*brittleness*), dan ketidakmampuan memproses persepsi dunia nyata (*Moravec's Paradox*).
- Dua periode **AI Winter** (1974–1980 dan 1987–1993) terjadi akibat jurang antara janji ekspektasi yang digelembungkan dengan realitas performa teknologi, berujung pada penghentian pendanaan masif.
- **Statistical Machine Learning** (1990-an) memulihkan reputasi AI dengan landasan probabilitas formal, pengujian berbasis dataset empiris, dan algoritma induktif.
- **Revolusi Deep Learning** (2012) meledak berkat konvergensi **Big Data (ImageNet)**, **Hardware Acceleration (GPU)**, dan terobosan optimasi algoritmik, menggantikan rekayasa fitur manual.
- Arsitektur **Transformer** (2017) dan **Foundation Models** melahirkan era *Generative AI*, mengubah cara perangkat lunak cerdas dibangun.
- Tantangan kontemporer seperti halusinasi dan opasitas model mendorong lahirnya **Neuro-Symbolic AI**, memadukan penalaran logis yang dapat dijelaskan dengan ketangguhan persepsi mendalam.

---

## 20. Latihan Terstruktur

### Latihan Tingkat Dasar (Konseptual & Historis)
1. Jelaskan perbedaan mendasar antara definisi operasional kecerdasan menurut Alan Turing (*Imitation Game*) dengan konsep pemikiran subjektif (*consciousness*)!
2. Mengapa peristiwa *Dartmouth Summer Research Project 1956* disepakati sebagai kelahiran resmi disiplin Artificial Intelligence, dan siapa saja tokoh kunci penggagasnya?
3. Sebutkan dan jelaskan dua faktor utama yang memicu terjadinya *AI Winter Pertama* pada pertengahan dekade 1970-an!

### Latihan Tingkat Menengah (Analisis Komparatif)
4. Jelaskan fenomena *Moravec's Paradox*! Berikan contoh konkret dua aktivitas: satu yang mudah bagi komputer era 1970-an tetapi sulit bagi balita, dan satu yang mudah bagi balita tetapi sangat sulit bagi komputer era tersebut!
5. Bandingkan arsitektur sistem pakar (*Expert System*) dengan sistem pemelajaran mesin berbasis *Supervised Learning* dalam hal:
   - Sumber aturan pengambilan keputusan.
   - Perilaku ketika menerima data di luar jangkauan pengetahuan awal (*out-of-distribution*).
   - Keterjelasan proses inferensi (*interpretability*).

### Latihan Tingkat Lanjut (Evaluasi Kritis & Sintesis)
6. Sebuah bank nasional ingin membangun sistem otomatis untuk memeriksa kepatuhan transaksi keuangan terhadap regulasi anti-pencucian uang (*Anti-Money Laundering* / AML) yang diatur secara ketat oleh hukum perbankan nasional.
   - Evaluasi trade-off jika bank tersebut memutuskan menggunakan:
     a) *Rule-Based Expert System* murni.
     b) *Deep Learning* murni.
     c) Pendekatan *Neuro-Symbolic / Hybrid*.
   - Pendekatan mana yang Anda rekomendasikan? Sertakan argumen terkait akuntabilitas hukum dan auditabilitas!

---

## 21. Pembahasan dan Panduan Jawaban Latihan

### Pembahasan Latihan Tingkat Dasar
1. **Perbedaan Turing Test vs Consciousness**: Turing mengalihkan perdebatan metafisik yang subjektif mengenai *"apakah mesin benar-benar sadar"* menjadi uji perilaku fungsional eksternal yang terukur secara empiris. Jika sebuah mesin mampu merespons pertanyaan manusia melalui kanal teks sedemikian rupa sehingga pengamat rasional tidak dapat membedakannya dari manusia sungguhan, maka mesin tersebut dianggap lulus uji kecerdasan operasional, terlepas dari apakah ada proses batiniah atau kesadaran subjektif (*qualia*).
2. **Kelahiran AI & Tokoh Kunci**: Dartmouth Workshop 1956 adalah lokakarya pertama di mana istilah *"Artificial Intelligence"* diresmikan dan dipisahkan dari payung keilmuan sibernetika atau teori otomata. Empat tokoh penggagas utamanya adalah John McCarthy, Marvin Minsky, Nathaniel Rochester, dan Claude Shannon, serta dihadiri pelopor lain seperti Allen Newell dan Herbert Simon.
3. **Penyebab AI Winter I**: 
   - Kritik teoretis buku *Perceptrons* (Minsky & Papert, 1969) yang membuktikan jaringan perceptron lapis tunggal tidak mampu menyelesaikan fungsi XOR, mematikan antusiasme riset koneksionisme.
   - Evaluasi kegagalan pendanaan dalam laporan *Lighthill Report* (Inggris, 1973) dan pengetatan anggaran DARPA di bawah Mansfield Amendment, karena janji ambisius penerjemahan bahasa otomatis dan robot serbaguna gagal diwujudkan.

### Pembahasan Latihan Tingkat Menengah
4. **Moravec's Paradox**:
   - *Mudah bagi komputer, sulit bagi balita*: Menghitung perkalian matriks besar, bermain catur dengan mengevaluasi ribuan langkah logis, atau membuktikan teorema matematika (Logic Theorist).
   - *Mudah bagi balita, sulit bagi komputer klasik*: Mengenali wajah ibu dari berbagai sudut pencahayaan, menjaga keseimbangan tubuh saat berjalan di atas karpet bergelombang, atau membedakan suara kucing mengeong dari derau kipas angin.
   - *Akar masalah*: Kemampuan penalaran logis tingkat tinggi adalah penemuan evolusi manusia yang baru berumur ribuan tahun sehingga aturannya eksplisit dan mudah dikodekan, sedangkan persepsi sensori-motorik telah berevolusi selama ratusan juta tahun sehingga proses komputasinya sangat terdistribusi dan bekerja di bawah sadar.
5. **Perbandingan Expert System vs Supervised Learning**:
   - *Sumber aturan*: Expert System mengambil kaidah langsung dari formalisasi wawancara pakar manusia; Supervised Learning mengekstrak bobot kaidah secara induktif dari anotasi label data historis.
   - *Out-of-distribution*: Expert System mengalami *brittleness* (berhenti atau gagal seketika); Supervised Learning cenderung memaksakan prediksi interpolatif yang sering kali salah tanpa indikasi ketidakpastian (*overconfidence*).
   - *Interpretability*: Expert System bersifat *white-box* dengan fasilitas pelacakan kaidah `IF-THEN`; Supervised Learning modern (terutama Deep Learning) cenderung *black-box* karena pengetahuan tersimpan pada jutaan bobot matriks non-linear.

### Pembahasan Latihan Tingkat Lanjut
6. **Evaluasi Kasus AML Perbankan**:
   - *Rule-Based System*: Sangat unggul dalam kepatuhan hukum karena setiap aturan regulasi (misal: "transaksi tunai > 100 juta wajib dilaporkan") dipetakan 100% presisi dan dapat diaudit oleh regulator. Namun, gagal mendeteksi pola pencucian uang baru (*novel money-laundering schemes*) yang sengaja menghindari ambang batas kaidah tertulis (*smurfing*).
   - *Deep Learning murni*: Mampu mendeteksi anomali pola transaksi kompleks yang tersembunyi. Namun, menjadi bencana kepatuhan hukum karena tidak dapat memberikan justifikasi deterministik saat transaksi nasabah diblokir secara sepihak (*right to explanation* di bawah regulasi finansial).
   - *Rekomendasi (Pendekatan Hybrid / Neuro-Symbolic)*: Menggunakan modul *Statistical Machine Learning / Deep Learning* untuk memindai graf transaksi dan mendeteksi anomali pola tersembunyi yang berisiko tinggi (*risk scoring*), kemudian mengumpankannya ke *Rule-Based Decision Engine* yang memastikan bahwa tindakan pembekuan rekening hanya dieksekusi jika melanggar klausul aturan hukum formal yang transparan dan dapat diaudit secara yuridis.

---

## 22. Tugas Mini Project: Analisis Evolusi Pendekatan AI pada Satu Domain Aplikasi

### Deskripsi Proyek
Mahasiswa diminta menyusun sebuah kajian analisis kritis mengenai bagaimana paradigma Artificial Intelligence berevolusi dalam menyelesaikan satu domain permasalahan praktis tertentu. Melalui tugas ini, mahasiswa membuktikan pemahaman bahwa evolusi teknologi AI tidak terjadi di ruang hampa, melainkan dipicu oleh keterbatasan metode lama, ketersediaan data baru, dan tuntutan keandalan aplikasi di dunia nyata.

### Pilihan Domain Aplikasi (Pilih Salah Satu)
1. **Computer Vision**: Dari deteksi tepi berbasis filter matematika (Sobel/Canny) dan rekayasa fitur manual (SIFT/HOG) menuju Deep CNN dan Vision Transformer (ViT).
2. **Natural Language Processing**: Dari tata bahasa berbasis aturan formal (Chomsky Grammars) dan kamus kaidah menuju n-gram statistik, word embeddings, hingga arsitektur Transformer dan LLM.
3. **Medical Diagnosis**: Dari sistem pakar kaidah inferensi klinis (MYCIN) menuju klasifikasi citra radiologi berbasis deep neural network dan model multimodal diagnostic assistant.
4. **Game Playing**: Dari pencarian pohon keadaan heuristik minimax (Deep Blue pada catur) menuju integrasi deep reinforcement learning dan Monte Carlo Tree Search (AlphaGo).
5. **Autonomous Navigation**: Dari perencanaan graf ruang keadaan diskret berbasis peta mikrodunia (Shakey the Robot) menuju end-to-end vision-based deep neural networks dan safety-envelope hybrid controller.

### Ruang Lingkup dan 7 Deliverables yang Dinilai
Laporan disusun dalam format laporan teknis akademik (4–6 halaman, PDF/Markdown) dengan memuat tujuh komponen:
1. **Deskripsi Domain & Karakteristik Masalah**: Perumusan tantangan inti domain tersebut, jenis input (data terstruktur, audio, teks, visual), dan tolok ukur keberhasilan.
2. **Garis Waktu (*Timeline*) Perkembangan Historis**: Urutan kronologis tonggak inovasi penting lengkap dengan tahun, nama sistem/algoritma pelopor, dan institusi pengembang.
3. **Kajian Tiga Pendekatan Utama**:
   - *Pendekatan 1 (Era Awal / Symbolic / Rule-Based)*.
   - *Pendekatan 2 (Era Menengah / Statistical ML)*.
   - *Pendekatan 3 (Era Modern / Deep Learning / Foundation Models)*.
4. **Analisis Komparatif Mendalam**: Tabel perbandingan antarketiga pendekatan mencakup kebutuhan data, kebutuhan komputasi, fleksibilitas terhadap noise, dan interpretabilitas.
5. **Faktor Pendorong Perubahan Paradigma**: Analisis kritis mengapa pendekatan lama ditinggalkan atau ditransformasikan (identifikasi batasan spesifik data, algoritma, atau hardware).
6. **Rekomendasi Arsitektur Masa Kini**: Rekomendasi solusi teknis terbaik untuk permasalahan domain tersebut di industri saat ini, termasuk evaluasi apakah pendekatan hibrida (*Neuro-Symbolic*) relevan diterapkan.
7. **Daftar Pustaka Akademik**: Minimal 3 rujukan primer bereputasi (paper seminal konferensi/jurnal internasional bereputasi atau buku teks standar).

---

## 23. Rubrik Penilaian Proyek (Total 100 Poin)

| Kriteria Penilaian | Bobot | Kinerja Unggul (Penuh) | Kinerja Cukup (Parsial) | Kinerja Kurang (Tidak Terpenuhi) |
|---|---:|---|---|---|
| **Akurasi Sejarah & Konteks** | **25 Poin** | Kronologi waktu akurat, tokoh dan sistem pelopor relevan, tidak ada anakronisme sejarah. (21–25) | Terdapat 1–2 ketidakakuratan tahun atau konteks sejarah yang terabaikan. (11–20) | Kronologi sejarah kabur, banyak fakta tidak akurat atau halusinasi. (0–10) |
| **Pemahaman Paradigma AI** | **25 Poin** | Mampu membedakan secara tajam logika internal Symbolic, Statistical, dan Deep Learning. (21–25) | Memahami konsep umum namun terjadi pencampuran konsep antarparadigma. (11–20) | Tidak memahami perbedaan filosofis dan teknis antarparadigma. (0–10) |
| **Kualitas Perbandingan Komparatif** | **20 Poin** | Tabel perbandingan komprehensif, multi-dimensi (data, komputasi, interpretabilitas, fleksibilitas). (17–20) | Tabel perbandingan bersifat dangkal atau hanya meninjau dari satu sudut pandang (misal akurasi saja). (9–16) | Tidak ada tabel perbandingan yang sistematis. (0–8) |
| **Ketepatan Analisis & Rekomendasi** | **20 Poin** | Menganalisis faktor pendorong secara mendalam dan memberikan rekomendasi solusi realistis. (17–20) | Analisis faktor pendorong terlalu umum; rekomendasi mengabaikan batasan praktis. (9–16) | Tidak mampu menjelaskan alasan perubahan paradigma; rekomendasi tidak masuk akal. (0–8) |
| **Komunikasi & Referensi Akademik** | **10 Poin** | Penulisan profesional, tata bahasa baku, mengutip minimal 3 referensi primer terverifikasi. (9–10) | Format cukup rapi, rujukan akademik kurang dari 3 atau format sitasi tidak konsisten. (5–8) | Tidak ada rujukan primer terverifikasi, plagiarisme terdeteksi, atau tulisan berantakan. (0–4) |

---

## 24. Referensi dan Sumber Belajar Primer

1. **Russell, S., & Norvig, P. (2020)**. *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson. (Bab 1: *Introduction and History of Artificial Intelligence*). URL: [https://aima.cs.berkeley.edu/](https://aima.cs.berkeley.edu/)
2. **Turing, A. M. (1950)**. *Computing Machinery and Intelligence*. Mind, 59(236), 433–460. DOI: [https://doi.org/10.1093/mind/LIX.236.433](https://doi.org/10.1093/mind/LIX.236.433)
3. **McCarthy, J., Minsky, M. L., Rochester, N., & Shannon, C. E. (1955)**. *A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence*. Makalah proposal seminal lokakarya Dartmouth. URL: [https://ojs.aaai.org/index.php/aimagazine/article/view/1904](https://ojs.aaai.org/index.php/aimagazine/article/view/1904)
4. **ACM / IEEE-CS / AAAI (2024)**. *Computer Science Curricula 2023 (CS2023): Artificial Intelligence Knowledge Area*. ACM Press. URL: [https://csed.acm.org/](https://csed.acm.org/)
5. **Newell, A., & Simon, H. A. (1976)**. *Computer Science as Empirical Inquiry: Symbols and Search*. Communications of the ACM, 19(3), 113–126.
6. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017)**. *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017).

