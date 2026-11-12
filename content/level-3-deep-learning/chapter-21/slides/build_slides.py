"""
Chapter 21 - Neural Network Fundamentals
Script Pembangkit Slide Deck Reveal.js (deck.json)
Menghasilkan JSON array slide deck yang memenuhi skema validator AI Learning Hub.
"""

import json
from pathlib import Path

slides = [
    {
        "kind": "cover",
        "title": "Neural Network Fundamentals",
        "body": "### Arsitektur Jaringan Saraf Tiruan, Fungsi Aktivasi & Mekanisme Propagasi Maju\n\n**Level 3 — Deep Learning & Generative AI | Chapter 21**\n\nArtificial Intelligence Learning Hub — Institut Teknologi Sumatera (ITERA)\n*Created & Developed by Aji Pamungkas Tri Nurcahyo*",
        "notes": "Selamat datang di Level 3 Deep Learning and Generative AI. Hari ini kita membuka Bab 21 mengenai Neural Network Fundamentals: fondasi arsitektur jaringan saraf tiruan, matematika neuron buatan, dan propagasi maju."
    },
    {
        "kind": "content",
        "title": "Mengapa Jaringan Saraf Tiruan?",
        "body": "- **Keterbatasan Feature Engineering Klasik**: Model machine learning tradisional sangat bergantung pada perancangan fitur manual yang membutuhkan kepakaran domain intensif.\n- **Representation Learning**: Jaringan saraf tiruan memiliki kemampuan mempelajari representasi fitur hierarkis secara otomatis langsung dari data mentah.\n- **Kapasitas Pemodelan Non-Linier**: Melalui komposisi lapis banyak dan aktivasi non-linier, jaringan saraf mampu memetakan batas keputusan arbitrer yang sangat kompleks.\n- **Skalabilitas Data Besar**: Berbeda dari algoritma klasik yang mengalami saturasi performa, model deep learning terus berkembang seiring bertambahnya volume data.",
        "notes": "Jelaskan transisi paradigma dari feature engineering manual (Level 2) menuju representational learning otomatis di Level 3."
    },
    {
        "kind": "content",
        "title": "Tujuan Pembelajaran Bab 21",
        "body": "Setelah menyelesaikan perkuliahan ini, mahasiswa diharapkan mampu:\n\n- **Menjelaskan** konsep Artificial Neural Networks dan perbandingannya dengan neuron biologis.\n- **Memahami** keterbatasan Perceptron tunggal pada masalah non-linier (XOR problem).\n- **Menghitung** kombinasi linier terbobot (*weighted sum*) dan peran parameter bias.\n- **Menggunakan** berbagai fungsi aktivasi non-linier (Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax).\n- **Menjelaskan** arsitektur Multilayer Perceptron (MLP) dan representasi matriksnya.\n- **Melakukan** propagasi maju (*forward propagation*) langkah demi langkah.\n- **Memilih** fungsi aktivasi output dan fungsi rugi (*loss function*) yang sesuai.",
        "notes": "Tujuan pembelajaran ini dipetakan berdasarkan Taksonomi Bloom level analisis dan sintesis komputasional."
    },
    {
        "kind": "split",
        "title": "AI vs Machine Learning vs Deep Learning",
        "body": "Hubungan inklusif tiga domain kecerdasan komputasional:\n\n### Artificial Intelligence (Payung Utama)\n- Disiplin ilmu luas yang mencakup seluruh teknik untuk merekayasa mesin cerdas, termasuk sistem pakar (*expert systems*) dan logika simbolik.\n\n### Machine Learning (Sub-bidang AI)\n- Pendekatan pembelajaran berbasis data di mana algoritma mengoptimalkan performa secara empiris tanpa aturan terprogram eksplisit.\n\n### Deep Learning (Sub-bidang ML)\n- Pembelajaran mesin yang memanfaatkan arsitektur jaringan saraf tiruan dengan kedalaman banyak lapisan (*deep representation layers*).",
        "notes": "Gambarkan visualisasi diagram konsentris: AI mencakup ML, dan ML mencakup Deep Learning."
    },
    {
        "kind": "content",
        "title": "Inspirasi Biologis vs Model Matematis",
        "body": "- **Soma (Badan Sel)**: Dalam model matematis direpresentasikan sebagai operasi penjumlahan kombinasi linier terbobot ($\\sum w_i x_i + b$).\n- **Dendrit**: Berperan sebagai kanal masukan fitur ($x_1, x_2, \\dots, x_n$).\n- **Sinapsis**: Kekuatan sambungan antar neuron direpresentasikan sebagai parameter bobot matematis ($w_i$).\n- **Akson & Potensial Aksi**: Transmisi sinyal listrik luaran direpresentasikan sebagai fungsi aktivasi non-linier $a = f(z)$.\n- **Catatan Penting**: Artificial Neural Network HANYA terinspirasi secara abstrak dan BUKAN merupakan simulasi biologis otak manusia.",
        "notes": "Tekankan batasan ilmiah: jangan melebih-lebihkan analogi biologis. Neuron buatan adalah fungsi aproksimasi matematis."
    },
    {
        "kind": "split",
        "title": "Model Matematis Neuron Buatan",
        "body": "Formula kombinasi linier dan transformasi aktivasi:\n\n### 1. Kombinasi Linier Terbobot (Weighted Sum)\n$$z = \\sum_{i=1}^{n} w_i x_i + b = \\mathbf{w}^T \\mathbf{x} + b$$\n- $\\mathbf{w}$: Vektor bobot sinaptik berdimensi $(n, 1)$.\n- $\\mathbf{x}$: Vektor fitur masukan berdimensi $(n, 1)$.\n- $b$: Skalar bias pergeseran ambang.\n\n### 2. Transformasi Aktivasi Non-Linier\n$$a = f(z) = f(\\mathbf{w}^T \\mathbf{x} + b)$$\n- $f(\\cdot)$: Fungsi aktivasi yang memetakan nilai $z$ ke rentang luaran tertentu.",
        "notes": "Tunjukkan bahwa operasi fundamental setiap neuron terdiri dari dua tahap berurutan: transformasi linier diikuti pemetaan non-linier."
    },
    {
        "kind": "content",
        "title": "Peran Krusial Parameter Bias",
        "body": "- **Mengapa Kita Butuh Bias?**: Tanpa bias ($b=0$), garis batas keputusan atau hyperplane matematis selalu dipaksa melewati titik pusat koordinat (*origin* $(0,0)$).\n- **Kebebasan Geometris**: Parameter bias bertindak seperti konstanta intersep pada persamaan garis lurus ($y = mx + c$), memungkinkan batas keputusan bergeser secara bebas di ruang fitur.\n- **Ambang Aktivasi**: Bias merepresentasikan seberapa mudah atau sulit sebuah neuron melepaskan sinyal aktivasi saat menerima masukan nol.",
        "notes": "Jelaskan dengan analogi tuas saklar: bias menentukan posisi awal saklar sebelum menerima tekanan masukan."
    },
    {
        "kind": "split",
        "title": "Rosenblatt Perceptron (1958)",
        "body": "Arsitektur awal neuron pembelajaran terawasi:\n\n### Karakteristik Perceptron\n- Memiliki masukan biner atau kontinu $\\mathbf{x}$.\n- Menggunakan fungsi aktivasi tangga Heaviside (*Unit Step Function*):\n$$f(z) = \\begin{cases} 1 & \\text{jika } z \\ge 0 \\\\ 0 & \\text{jika } z < 0 \\end{cases}$$\n\n### Batas Keputusan\n- Menghasilkan garis pemisah linier: $w_1 x_1 + w_2 x_2 + b = 0$.\n- Sisi positif ($z \\ge 0$) diprediksi sebagai Kelas 1.\n- Sisi negatif ($z < 0$) diprediksi sebagai Kelas 0.",
        "notes": "Perceptron merupakan batu loncatan historis penting yang digagas oleh Frank Rosenblatt pada tahun 1958."
    },
    {
        "kind": "content",
        "title": "Gerbang Logika Linier (AND, OR, NAND)",
        "body": "- **Gerbang AND**: Memerlukan kedua masukan aktif ($x_1=1, x_2=1$).\n  - Bobot: $w_1 = 1.0, w_2 = 1.0$, Bias: $b = -1.5$.\n  - $z(1,1) = 1.0(1) + 1.0(1) - 1.5 = +0.5 \\ge 0 \\implies 1$.\n  - $z(1,0) = 1.0(1) + 1.0(0) - 1.5 = -0.5 < 0 \\implies 0$.\n- **Gerbang OR**: Cukup salah satu masukan aktif.\n  - Bobot: $w_1 = 1.0, w_2 = 1.0$, Bias: $b = -0.5$.\n- **Keterpisahan Linier (*Linear Separability*)**: Titik data kelas 0 dan 1 dapat dipisahkan secara sempurna oleh satu garis lurus.",
        "notes": "Tunjukkan bahwa logika boolean dasar dapat diselesaikan sepenuhnya menggunakan Perceptron tunggal."
    },
    {
        "kind": "split",
        "title": "Keterbatasan Perceptron: Masalah XOR",
        "body": "Tantangan non-linier yang memicu AI Winter pertama (1969):\n\n### Tabel Kebenaran XOR\n- $(0,0) \\to 0$, $(0,1) \\to 1$, $(1,0) \\to 1$, $(1,1) \\to 0$.\n\n### Kontradiksi Sistem Pertidaksamaan\n1. $b < 0$ (dari input $0,0$)\n2. $w_2 + b \\ge 0$ (dari input $0,1$)\n3. $w_1 + b \\ge 0$ (dari input $1,0$)\n4. $w_1 + w_2 + b < 0$ (dari input $1,1$)\n\nJumlahkan (2) & (3) $\\implies w_1 + w_2 + b > 0$, yang berkontradiksi langsung dengan (4)!",
        "notes": "Minsky dan Papert (1969) membuktikan secara matematis bahwa Perceptron lapis tunggal tidak mungkin menyelesaikan XOR."
    },
    {
        "kind": "content",
        "title": "Solusi XOR: Jaringan Lapis Banyak (MLP)",
        "body": "- **Dekomposisi Logika**: Masalah XOR dapat dipecahkan dengan menggabungkan gerbang dasar: $\\text{XOR}(x_1, x_2) = (x_1 \\lor x_2) \\land \\neg(x_1 \\land x_2)$.\n- **Arsitektur 2-Layer**:\n  - *Hidden Layer*: Neuron 1 memetakan gerbang OR, Neuron 2 memetakan gerbang NAND.\n  - *Output Layer*: Neuron 3 memetakan gerbang AND dari luaran Neuron 1 dan 2.\n- **Transformasi Ruang Fitur**: Lapisan tersembunyi mentransformasikan titik data ke ruang representasi baru di mana kelas menjadi *linearly separable*.",
        "notes": "Inilah kekuatan utama jaringan multi-lapisan: lapisan tersembunyi merekayasa representasi fitur baru."
    },
    {
        "kind": "content",
        "title": "Mengapa Fungsi Aktivasi Non-Linier Wajib?",
        "body": "- **Bahaya Komposisi Linier**: Tanpa aktivasi non-linier, jaringan saraf 100 lapisan sekalipun secara matematis HANYA setara dengan regresi linier satu lapisan tunggal!\n  - Bukti: $\\mathbf{W}_2 (\\mathbf{W}_1 \\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2 = (\\mathbf{W}_2 \\mathbf{W}_1) \\mathbf{x} + (\\mathbf{W}_2 \\mathbf{b}_1 + \\mathbf{b}_2) = \\mathbf{W}_{\\text{net}} \\mathbf{x} + \\mathbf{b}_{\\text{net}}$.\n- **Pemecah Linearitas**: Fungsi aktivasi non-linier menyuntikkan kelengkungan matematis, memungkinkan aproksimasi fungsi kontinu arbitrer.",
        "notes": "Tekankan bukti formal: komposisi fungsi linier selalu menghasilkan fungsi linier lain."
    },
    {
        "kind": "split",
        "title": "Fungsi Aktivasi: Sigmoid (Logistik)",
        "body": "Pemetaan kontinu ke rentang probabilitas $(0, 1)$:\n\n### Formula Matematis\n$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$\n\n### Turunan Pertama\n$$\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$$\n\n### Kelebihan & Keterbatasan\n- (+) Ideal untuk output klasifikasi biner.\n- (-) **Saturasi & Vanishing Gradient**: Saat $|z| > 4$, turunan mendekati nol, menghentikan aliran pembelajaran gradien.",
        "notes": "Sigmoid banyak digunakan di masa lalu, namun kini jarang dipakai pada hidden layers karena masalah vanishing gradient."
    },
    {
        "kind": "split",
        "title": "Fungsi Aktivasi: Tangen Hiperbolik (Tanh)",
        "body": "Pemetaan kontinu zero-centered ke rentang $(-1, 1)$:\n\n### Formula Matematis\n$$\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}$$\n\n### Turunan Pertama\n$$\\tanh'(z) = 1 - \\tanh^2(z)$$\n\n### Kelebihan & Keterbatasan\n- (+) Luaran berpusat di nol (*zero-centered*), mempercepat konvergensi gradien.\n- (-) Masih mengalami masalah saturasi gradien pada nilai input ekstrem.",
        "notes": "Tanh secara umum lebih unggul dibanding Sigmoid pada hidden layer karena sifat zero-centerednya."
    },
    {
        "kind": "split",
        "title": "Fungsi Aktivasi: ReLU (Rectified Linear Unit)",
        "body": "Standar emas aktivasi jaringan saraf modern:\n\n### Formula Matematis\n$$f(z) = \\max(0, z)$$\n\n### Turunan Pertama\n$$f'(z) = \\begin{cases} 1 & \\text{jika } z > 0 \\\\ 0 & \\text{jika } z < 0 \\end{cases}$$\n\n### Kelebihan & Keterbatasan\n- (+) Sangat cepat dihitung (tanpa operasi eksponensial mahal).\n- (+) Tidak jenuh pada sisi positif ($z > 0$).\n- (-) **Dying ReLU**: Jika neuron menerima gradien negatif terus-menerus, ia mati permanen.",
        "notes": "ReLU adalah aktivasi default paling populer di deep learning modern sejak diperkenalkan oleh Nair & Hinton (2010)."
    },
    {
        "kind": "split",
        "title": "Fungsi Aktivasi: Leaky ReLU",
        "body": "Modifikasi pencegah fenomena Dying ReLU:\n\n### Formula Matematis\n$$f(z) = \\max(\\alpha z, z) = \\begin{cases} z & \\text{jika } z > 0 \\\\ \\alpha z & \\text{jika } z \\le 0 \\end{cases}$$\n- Biasanya $\\alpha = 0.01$.\n\n### Turunan Pertama\n$$f'(z) = \\begin{cases} 1 & \\text{jika } z > 0 \\\\ \\alpha & \\text{jika } z \\le 0 \\end{cases}$$\n\n### Karakteristik Utama\n- Mempertahankan gradien kecil $\\alpha$ pada wilayah negatif sehingga neuron tetap dapat pulih selama proses training.",
        "notes": "Leaky ReLU memberikan jalur penyelamatan bagi neuron agar tidak mengalami 'kematian' permanen."
    },
    {
        "kind": "split",
        "title": "Fungsi Aktivasi: Softmax",
        "body": "Distribusi probabilitas untuk klasifikasi multikelas:\n\n### Formula Matematis\n$$\\text{Softmax}(z_i) = \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}}$$\n\n### Karakteristik Distribusi\n- $\\text{Softmax}(z_i) \\in (0, 1)$ untuk setiap kelas $i$.\n- $\\sum_{i=1}^{K} \\text{Softmax}(z_i) = 1.0$ tepat.\n\n### Trik Stabilitas Numerik (*Subtract Max*)\n$$\\text{Softmax}(z_i) = \\frac{e^{z_i - \\max(\\mathbf{z})}}{\\sum_j e^{z_j - \\max(\\mathbf{z})}}$$\n- Mencegah bahaya komputasi floating-point overflow pada logits bernilai besar.",
        "notes": "Softmax mengubah vektor logits riil tak terbatas menjadi distribusi probabilitas yang valid dan terkalibrasi."
    },
    {
        "kind": "content",
        "title": "Arsitektur Multilayer Perceptron (MLP)",
        "body": "- **Input Layer (Lapisan Masukan)**: Menerima fitur masukan $\\mathbf{x} \\in \\mathbb{R}^{D_{\\text{in}}}$. Tidak memiliki bobot parameter internal.\n- **Hidden Layers (Lapisan Tersembunyi)**: Lapisan perantara yang mengekstraksi representasi fitur non-linier melalui kombinasi bobot dan aktivasi.\n- **Output Layer (Lapisan Luaran)**: Menghasilkan prediksi akhir sesuai tipe masalah (regresi kontinu, biner, atau multikelas).\n- **Fully Connected (Dense)**: Setiap neuron pada lapisan $l$ terhubung ke seluruh neuron pada lapisan $l-1$.",
        "notes": "Jelaskan topologi standar feedforward network di mana aliran informasi hanya bergerak maju dari kiri ke kanan."
    },
    {
        "kind": "content",
        "title": "Notasi & Dimensi Matriks Jaringan",
        "body": "- Misalkan jaringan memiliki dimensi lapis: $[n_0, n_1, n_2, \\dots, n_L]$.\n- **Matriks Bobot $\\mathbf{W}^{[l]}$**: Menghubungkan layer $l-1$ ke layer $l$.\n  - Dimensi konvensi batch ($X W$): berukuran $(n_{l-1}, n_l)$.\n- **Vektor Bias $\\mathbf{b}^{[l]}$**: Berdimensi $(1, n_l)$.\n- **Total Parameter Lapisan $l$**: $(n_{l-1} \\times n_l) + n_l$.\n- **Universal Approximation Theorem**: MLP dengan minimal satu hidden layer dan aktivasi non-linier dapat mengaproksimasi fungsi kontinu apa pun dengan derajat presisi arbitrer (Cybenko 1989, Hornik 1991).",
        "notes": "Ingatkan bahwa teorema aproksimasi universal tidak menjamin bahwa bobot optimal dapat ditemukan dengan mudah oleh algoritma optimasi."
    },
    {
        "kind": "split",
        "title": "Mekanisme Propagasi Maju (Forward Pass)",
        "body": "Kalkulasi rekursif layer demi layer:\n\n### Persamaan Komputasi Umum (Layer $l$)\n$$\\mathbf{Z}^{[l]} = \\mathbf{A}^{[l-1]} \\mathbf{W}^{[l]} + \\mathbf{b}^{[l]}$$\n$$\\mathbf{A}^{[l]} = f^{[l]}(\\mathbf{Z}^{[l]})$$\n\n### Inisialisasi & Terminasi\n- $\\mathbf{A}^{[0]} = \\mathbf{X}$ (Matriks masukan batch data).\n- $\\hat{\\mathbf{Y}} = \\mathbf{A}^{[L]}$ (Aktivasi lapisan luaran akhir).\n- Semua nilai intermediate $\\mathbf{Z}^{[l]}$ dan $\\mathbf{A}^{[l]}$ disimpan dalam *cache* untuk evaluasi gradien saat backpropagation.",
        "notes": "Forward pass adalah pemetaan deterministik dari fitur masukan menjadi vektor prediksi akhir."
    },
    {
        "kind": "content",
        "title": "Worked Numerical Example (Arsitektur 2 -> 3 -> 1)",
        "body": "- **Masukan**: $\\mathbf{x} = [1.5, 2.0]$.\n- **Layer 1 (Hidden, ReLU)**:\n  - $\\mathbf{W}^{[1]} = \\begin{bmatrix} 0.5 & -0.2 & 0.8 \\\\ 0.4 & 0.7 & -0.5 \\end{bmatrix}$, $\\mathbf{b}^{[1]} = [0.1, -0.3, 0.2]$.\n  - $z_1 = 1.5(0.5) + 2.0(0.4) + 0.1 = 1.65 \\implies a_1 = 1.65$.\n  - $z_2 = 1.5(-0.2) + 2.0(0.7) - 0.3 = 0.80 \\implies a_2 = 0.80$.\n  - $z_3 = 1.5(0.8) + 2.0(-0.5) + 0.2 = 0.40 \\implies a_3 = 0.40$.\n- **Layer 2 (Output, Sigmoid)**:\n  - $\\mathbf{W}^{[2]} = [0.6, -0.8, 0.3]^T$, $b^{[2]} = -0.1$.\n  - $z^{[2]} = 1.65(0.6) + 0.80(-0.8) + 0.40(0.3) - 0.1 = 0.370$.\n  - $a^{[2]} = \\sigma(0.370) = \\frac{1}{1 + e^{-0.370}} = \\mathbf{0.5915}$.",
        "notes": "Jalankan perhitungan manual ini langkah demi langkah di papan tulis agar mahasiswa memahami asal-usul setiap angka."
    },
    {
        "kind": "split",
        "title": "Pemrosesan Batch Tervektorisasi",
        "body": "Memproses $N$ sampel data secara simultan:\n\n### Dimensi Matriks Batch\n- Matriks Data $\\mathbf{X}$: Berukuran $(N, D_{\\text{in}})$.\n- Hidden Layer $\\mathbf{Z}^{[1]}$: $(N, D_{\\text{in}}) \\times (D_{\\text{in}}, D_h) + (1, D_h) \\implies (N, D_h)$.\n- Output Layer $\\mathbf{Z}^{[2]}$: $(N, D_h) \\times (D_h, D_{\\text{out}}) + (1, D_{\\text{out}}) \\implies (N, D_{\\text{out}})$.\n\n### Manfaat Vektorisasi\n- Mengeksploitasi pustaka BLAS (Basic Linear Algebra Subprograms) berkecepatan tinggi.\n- Jauh lebih efisien dibanding perulangan `for` pada sampel individual.",
        "notes": "Vektorisasi matriks adalah kunci performa tinggi komputasi deep learning modern."
    },
    {
        "kind": "content",
        "title": "Pemilihan Layer Output & Pasangan Loss",
        "body": "- **Regresi Kontinu**:\n  - Aktivasi Output: Linear ($a = z$).\n  - Fungsi Rugi: Mean Squared Error (MSE) atau Huber Loss.\n- **Klasifikasi Biner**:\n  - Aktivasi Output: Sigmoid ($a = \\sigma(z) \\in (0,1)$).\n  - Fungsi Rugi: Binary Cross-Entropy (BCE).\n- **Klasifikasi Multikelas ($K$ Kelas Mutually Exclusive)**:\n  - Aktivasi Output: Softmax (distribusi probabilitas $\\sum p_k = 1$).\n  - Fungsi Rugi: Categorical Cross-Entropy (CCE).\n- **Prinsip**: Pasangan aktivasi dan loss harus sesuai secara probabilitas matematis.",
        "notes": "Kombinasi yang salah (misal Softmax dengan MSE) akan menghasilkan permukaan optimasi yang sangat buruk dan lambat konvergen."
    },
    {
        "kind": "split",
        "title": "Mean Squared Error (MSE)",
        "body": "Fungsi rugi standar estimasi nilai kontinu:\n\n### Formula Matematis\n$$\\mathcal{L}_{\\text{MSE}}(\\mathbf{y}, \\hat{\\mathbf{y}}) = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$$\n\n### Sifat & Karakteristik\n- Menghitung rata-rata kuadrat deviasi antara target riil dan prediksi.\n- Memberikan penalti kuadratik yang sangat berat terhadap kesalahan besar (*outliers*).\n- Turunan terhadap prediksi bersifat linier dan elegan: $\\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}_i} = \\frac{2}{N}(\\hat{y}_i - y_i)$.",
        "notes": "MSE adalah fungsi loss paling intuitif untuk masalah aproksimasi fungsi dan regresi."
    },
    {
        "kind": "split",
        "title": "Binary Cross-Entropy (BCE)",
        "body": "Fungsi rugi berbasis Maximum Likelihood untuk biner:\n\n### Formula Matematis\n$$\\mathcal{L}_{\\text{BCE}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\left[ y_i \\ln(\\hat{y}_i) + (1 - y_i) \\ln(1 - \\hat{y}_i) \\right]$$\n\n### Penalti Logaritmik Asimetris\n- Jika $y = 1$ dan $\\hat{y} \\to 0$, loss $-\\ln(\\hat{y}) \\to +\\infty$.\n- Jika $y = 0$ dan $\\hat{y} \\to 1$, loss $-\\ln(1 - \\hat{y}) \\to +\\infty$.\n- Menghukum model secara sangat keras jika membuat prediksi salah dengan tingkat keyakinan tinggi (*confident wrong*).",
        "notes": "BCE memastikan model tidak hanya akurat tetapi juga terkalibrasi secara probabilitas."
    },
    {
        "kind": "content",
        "title": "Categorical Cross-Entropy (CCE)",
        "body": "- **Formula One-Hot Encoding**: $\\mathcal{L}_{\\text{CCE}} = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{k=1}^{K} y_{i,k} \\ln(\\hat{y}_{i,k})$.\n- Karena label one-hot hanya bernilai 1 pada kelas target benar $c$, formula menyederhanakan menjadi: $\\mathcal{L}_i = -\\ln(\\hat{y}_{i, c})$.\n- **Stabilisasi Epsilon Clipping**: Di dunia nyata, evaluasi $\\ln(0)$ menghasilkan nilai $-\\infty$ atau `NaN` pada floating-point komputasi.\n- **Praktik Standar**: Klip nilai prediksi ke dalam batas aman: $\\hat{y} = \\text{clip}(\\hat{y}, \\epsilon, 1 - \\epsilon)$ dengan $\\epsilon = 10^{-15}$.",
        "notes": "Jelaskan pentingnya stabilitas numerik dalam implementasi kode deep learning profesional."
    },
    {
        "kind": "content",
        "title": "Kapan Jaringan Saraf BUKAN Pilihan Tepat?",
        "body": "- **Volume Data Kecil ($N < 1,000$)**: Algoritma klasik seperti Random Forest atau SVM umumnya mengungguli neural network karena terhindar dari overfitting parah.\n- **Data Tabular Terstruktur**: GBDT (XGBoost/LightGBM) secara konsisten terbukti lebih efisien dan akurat pada sebagian besar benchmark data tabel industri.\n- **Kebutuhan Transparansi Mutlak**: Dalam domain regulasi ketat (hukum pidana, kredit perbankan murni), model *white-box* (Decision Tree, Linear Regression) lebih diutamakan.\n- **Keterbatasan Komputasi**: Model deep learning menuntut biaya pelatihan dan inferensi yang jauh lebih besar.",
        "notes": "Latih pemikiran kritis mahasiswa: Deep Learning bukan peluru perak (silver bullet) untuk semua masalah data."
    },
    {
        "kind": "split",
        "title": "Inisialisasi Bobot: He vs Xavier",
        "body": "Menghindari ledakan atau kematian sinyal aktivasi:\n\n### Inisialisasi Nol (Dilarang Keras!)\n- Jika seluruh $W = 0$, seluruh neuron pada hidden layer menghitung nilai identik (*symmetry problem*) dan tidak dapat belajar.\n\n### Xavier / Glorot Initialization\n$$\\text{Var}(W) = \\frac{2}{n_{\\text{in}} + n_{\\text{out}}}$$\n- Optimal untuk fungsi aktivasi Tanh dan Sigmoid.\n\n### He Normal Initialization\n$$\\text{Var}(W) = \\frac{2}{n_{\\text{in}}}$$\n- Optimal khusus untuk fungsi aktivasi ReLU dan variannya.",
        "notes": "Inisialisasi bobot yang tepat menjamin varians aktivasi tetap stabil di seluruh kedalaman lapisan."
    },
    {
        "kind": "content",
        "title": "Ringkasan Alur Komputasi Forward Pass",
        "body": "1. **Masukan**: Terima tensor $\\mathbf{X}$ berdimensi $(N, D_{\\text{in}})$.\n2. **Transformasi Linier Hidden**: Hitung $\\mathbf{Z}^{[1]} = \\mathbf{X} \\mathbf{W}^{[1]} + \\mathbf{b}^{[1]}$.\n3. **Transformasi Non-Linier Hidden**: Hitung $\\mathbf{A}^{[1]} = \\text{ReLU}(\\mathbf{Z}^{[1]})$.\n4. **Transformasi Linier Output**: Hitung $\\mathbf{Z}^{[2]} = \\mathbf{A}^{[1]} \\mathbf{W}^{[2]} + \\mathbf{b}^{[2]}$.\n5. **Aktivasi Output**: Hitung $\\hat{\\mathbf{Y}} = \\sigma(\\mathbf{Z}^{[2]})$ atau $\\text{Softmax}(\\mathbf{Z}^{[2]})$.\n6. **Evaluasi Loss**: Bandingkan prediksi $\\hat{\\mathbf{Y}}$ dengan target aktual $\\mathbf{Y}$ via fungsi rugi $\\mathcal{L}$.",
        "notes": "Rangkum seluruh alur propagasi maju menjadi satu kesatuan alur kerja yang mudah diingat."
    },
    {
        "kind": "summary",
        "title": "Kesimpulan & Jembatan ke Chapter 22",
        "body": "- **Fondasi Kokoh**: Kita telah memahami bagaimana neuron buatan menyusun arsitektur MLP dan memetakan masukan menjadi prediksi melalui Propagasi Maju.\n- **Pertanyaan Terbuka**: Setelah nilai Loss $\\mathcal{L}$ terhitung, bagaimana jaringan menyesuaikan ribuan bobot $\\mathbf{W}$ dan bias $\\mathbf{b}$ secara otomatis agar nilai loss mengecil?\n- **Langkah Berikutnya (Chapter 22)**: Kita akan mempelajari **Backpropagation** (aturan rantai kalkulus untuk menghitung gradien galat) dan **Optimization** (Gradient Descent, Momentum, RMSprop, Adam)!",
        "notes": "Tutup sesi perkuliahan Chapter 21 dan bangun antusiasme mahasiswa menyongsong materi inti Backpropagation di Chapter 22."
    }
]

def main():
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "deck.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    print(f"Berhasil menulis {len(slides)} slide ke {out_file}")

if __name__ == "__main__":
    main()

