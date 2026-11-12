"""Pembangkit Berkas Slide Reveal.js (deck.json) untuk Chapter 15:
Decision Trees & Ensemble Learning (34 Slide Terstruktur)
"""

import json
import pathlib

slides = [
    {
        "kind": "cover",
        "title": "Decision Trees & Ensemble Learning",
        "body": "### Pohon Keputusan, CART, Pruning, Bagging, Random Forest, AdaBoost & Gradient Boosting\n\n**Level 2 — Machine Learning | Chapter 15**\n\nArtificial Intelligence Learning Hub — Institut Teknologi Sumatera (ITERA)\n*Created & Developed by Aji Pamungkas Tri Nurcahyo*",
        "notes": "Selamat datang di perkuliahan Chapter 15. Hari ini kita membahas keluarga model pohon keputusan, mulai dari struktur dasar CART, kriteria impurity, teknik pruning, hingga ensemble modern seperti Random Forest dan Gradient Boosting."
    },
    {
        "kind": "content",
        "title": "Why Decision Trees?",
        "body": "Mengapa model berbasis pohon sangat populer dalam Machine Learning?\n\n- **Tingkat Keterbukaan Tinggi (White-Box)**: Keputusan dapat ditelusuri secara visual menjadi rangkaian aturan IF-THEN.\n- **Invarian terhadap Skala Fitur**: Tidak memerlukan standardisasi atau normalisasi numerik yang ketat.\n- **Mampu Menangkap Relasi Non-Linier**: Membagi ruang fitur secara ortogonal tanpa asumsi linearitas.\n- **Fondasi Ensemble Modern**: Menjadi fondasi bagi algoritma terkuat pada data tabular (Random Forest, XGBoost, LightGBM).",
        "notes": "Jelaskan bahwa pohon keputusan menjembatani pemodelan berbasis aturan dari AI simbolik dengan pembelajaran statistik modern."
    },
    {
        "kind": "content",
        "title": "Learning Objectives",
        "body": "Setelah menyelesaikan perkuliahan ini, mahasiswa diharapkan mampu:\n\n- **Menjelaskan** konsep Decision Tree untuk klasifikasi dan regresi.\n- **Menginterpretasikan** anatomi pohon: root node, internal nodes, branches, dan leaf nodes.\n- **Menghitung** Shannon Entropy, Gini Impurity, dan Information Gain secara matematis.\n- **Menganalisis** fenomena overfitting pada pohon dan teknik pemangkasan (*pruning*).\n- **Membedakan** mekanisme Bagging (Bootstrap Aggregating) dan Boosting sekuensial.\n- **Mengimplementasikan** Random Forest, AdaBoost, dan Gradient Boosting menggunakan Python.\n- **Mengevaluasi** performa ensemble menggunakan Out-of-Bag (OOB) dan validasi silang berstrata.",
        "notes": "Sampaikan capaian pembelajaran kognitif terstandarisasi untuk bab ini."
    },
    {
        "kind": "split",
        "title": "Decision Tree Concept",
        "body": "Prinsip dasar pohon keputusan:\n\n### Pemartisian Rekursif\n- Membagi ruang fitur berdimensi $D$ menjadi wilayah-wilayah hiper-persegi panjang (*hyper-rectangles*).\n- Setiap pembagian (*split*) bertujuan memaksimalkan kemurnian target pada node anak.\n\n### Hubungan dengan Aturan Logika\n- Setiap jalur dari *Root Node* ke *Leaf Node* ekuivalen dengan konjungsi logika IF $X_1 > \theta_1$ AND $X_2 \le \theta_2$ THEN $Y = c$.\n- Aturan dipelajari secara otomatis dari data, bukan didefinisikan manual oleh pakar.",
        "notes": "Tekankan bahwa pohon menyederhanakan ruang probabilitas kompleks menjadi partisi bertingkat."
    },
    {
        "kind": "split",
        "title": "Classification vs Regression Trees",
        "body": "Pohon keputusan mendukung dua paradigma pemodelan:\n\n### Classification Trees\n- Target bersifat kategorikal diskret ($y \\in \\{0, 1, \\dots, K-1\\}$).\n- Daun memprediksi label kelas mayoritas atau distribusi probabilitas kelas.\n- Kriteria split: Gini Impurity atau Shannon Entropy.\n\n### Regression Trees\n- Target bersifat numerik kontinu ($y \\in \\mathbb{R}$).\n- Daun memprediksi nilai rata-rata sampel lokal (*piecewise constant*).\n- Kriteria split: Penurunan varians atau Mean Squared Error (MSE).",
        "notes": "Bandingkan prediksi diskret versus fungsi tangga konstan sepotong-sepotong pada regresi pohon."
    },
    {
        "kind": "content",
        "title": "Tree Anatomy",
        "body": "Elemen-elemen struktural dari pohon keputusan:\n\n- **Root Node (Node Akar)**: Titik masuk utama seluruh populasi data latih; tidak memiliki cabang masuk.\n- **Internal / Decision Node**: Node yang mengevaluasi kondisi fitur ($X_j \\le \\theta$) dan memecah data menjadi dua atau lebih cabang.\n- **Branch (Cabang)**: Jalur transisi keputusan berdasarkan hasil evaluasi kondisi (Ya / Tidak).\n- **Leaf / Terminal Node**: Ujung pohon yang tidak memiliki anak; menghasilkan estimasi target final.",
        "notes": "Gambarkan pohon terbalik di mana akar berada di paling atas dan daun berada di paling bawah."
    },
    {
        "kind": "content",
        "title": "IF-THEN Decision Rules",
        "body": "Pohon keputusan mengekstrak pengetahuan terstruktur dari data:\n\n```text\nJIKA Suhu > 84.0 °C MAKA\n    Prediksi: FAULT (Status Kritis)\nJIKA Suhu <= 84.0 °C DAN Vibrasi > 5.2 mm/s MAKA\n    Prediksi: FAULT (Kerusakan Bantalan)\nJIKA Suhu <= 84.0 °C DAN Vibrasi <= 3.6 mm/s MAKA\n    Prediksi: NORMAL (Operasi Prima)\n```\n\nSifat transparansi ini sangat penting dalam domain berisiko tinggi seperti industri, medis, dan otomotif.",
        "notes": "Bandingkan dengan model black-box seperti Deep Neural Network yang sulit diaudit langkah logikanya."
    },
    {
        "kind": "split",
        "title": "Recursive Partitioning",
        "body": "Mekanisme algoritma pembentukan pohon (Greedy Approach):\n\n### Algoritma Greedy Top-Down\n- Pada setiap langkah, algoritma mencari fitur $j$ dan ambang batas $\\theta$ terbaik yang memaksimalkan reduksi impurity.\n- Tidak melakukan pencarian ulang (*backtracking*); keputusan split lokal diambil secara sekuensial.\n\n### Kriteria Penghentian (Stopping Criteria)\n- Semua sampel di node memiliki kelas yang sama (*purity 100%*).\n- Mencapai batas kedalaman maksimum (*max depth*).\n- Jumlah sampel node kurang dari ambang batas minimum (*min samples split*).",
        "notes": "Jelaskan mengapa pendekatan greedy dipilih karena pencarian pohon global optimal adalah masalah NP-Complete."
    },
    {
        "kind": "content",
        "title": "Shannon Entropy",
        "body": "Ukuran ketidakpastian informasi dalam suatu himpunan data:\n\n- **Formulasi Matematis**:\n  $$H(S) = -\\sum_{i=1}^{K} p_i \\log_2(p_i)$$\n  di mana $p_i$ adalah proporsi sampel kelas ke-$i$ pada node $S$.\n- **Karakteristik Skala**:\n  - Untuk kasus 2-kelas: $H(S) = 0$ (murni total, seluruh sampel 1 kelas).\n  - Nilai maksimum $H(S) = 1.0$ bit ketika kelas terdistribusi imbang sempurna ($50\\% : 50\\%$).\n- Digunakan secara historis pada algoritma ID3 dan C4.5.",
        "notes": "Jelaskan interpretasi entropy sebagai jumlah bit minimum yang diperlukan untuk mengkodekan kelas sampel."
    },
    {
        "kind": "content",
        "title": "Gini Impurity",
        "body": "Ukuran probabilitas salah klasifikasi jika sampel dilabeli acak menurut distribusi kelas:\n\n- **Formulasi Matematis**:\n  $$I_G(S) = 1 - \\sum_{i=1}^{K} p_i^2$$\n- **Karakteristik Komputasi**:\n  - Untuk kasus 2-kelas: $I_G(S) = 0$ (murni) hingga maksimum $0.50$ (tersebar merata $50\\% : 50\\%$).\n  - Lebih cepat dihitung daripada Entropy karena tidak melibatkan operasi logaritma $\\log_2$.\n- Merupakan kriteria pemisahan default pada algoritma CART dan Scikit-learn.",
        "notes": "Tunjukkan bahwa secara empiris Gini dan Entropy menghasilkan pohon dengan performa yang hampir identik pada 98% kasus."
    },
    {
        "kind": "split",
        "title": "Information Gain",
        "body": "Penurunan tingkat ketidakmurnian setelah pemisahan node:\n\n### Formulasi Matematika\n$$IG(S, A) = \\text{Imp}(S) - \\sum_{v \\in \\text{Children}} \\frac{|S_v|}{|S|} \\text{Imp}(S_v)$$\n\n### Bobot Proporsi Sampel\n- Ketidakmurnian node anak dibobotkan berdasarkan proporsi jumlah data $\\frac{|S_v|}{|S|}$.\n- Fitur dengan nilai $IG$ terbesar dipilih sebagai pemisah node aktif saat ini.",
        "notes": "Jelaskan bahwa Information Gain mengukur seberapa banyak informasi baru yang kita dapatkan dari memfilter data dengan fitur A."
    },
    {
        "kind": "case",
        "title": "Splitting Calculation Example",
        "body": "Studi kasus perhitungan manual node sensor (10 sampel: 4 Normal, 6 Fault):\n\n- **Node Induk ($S$)**: $H(S) = - (0.4 \\log_2 0.4 + 0.6 \\log_2 0.6) = 0.9710$ bits.\n- **Kandidat Split (Suhu > 75 °C)**:\n  - Cabang Kiri (Suhu $\\le 75$ °C): 5 sampel (4 Normal, 1 Fault) -> $H(S_L) = 0.7219$ bits.\n  - Cabang Kanan (Suhu $> 75$ °C): 5 sampel (0 Normal, 5 Fault) -> $H(S_R) = 0$ bits (Murni!).\n- **Information Gain**:\n  $$IG = 0.9710 - \\left( \\frac{5}{10}(0.7219) + \\frac{5}{10}(0) \\right) = 0.6100 \\text{ bits}$$\n  Pemisahan ini menghasilkan penurunan entropi yang masif.",
        "notes": "Ajak mahasiswa memperhatikan bagaimana node kanan yang murni membuat nilai entropi anak menjadi nol."
    },
    {
        "kind": "content",
        "title": "CART Algorithm",
        "body": "Classification and Regression Trees (Breiman et al., 1984):\n\n- **Pemisahan Biner Murni (*Strictly Binary Splits*)**: Setiap node induk selalu menghasilkan tepat dua cabang anak ($X_j \\le \\theta$ vs $X_j > \\theta$).\n- **Evaluasi Fitur Numerik Kontinu**: Fitur kontinu diurutkan, lalu setiap titik tengah antar-nilai berurutan diuji sebagai kandidat ambang $\\theta$.\n- **Kriteria Objektif**:\n  - Klasifikasi: Meminimalkan rata-rata tertimbang Gini Impurity.\n  - Regresi: Meminimalkan Mean Squared Error (MSE) / varians lokal.",
        "notes": "Jelaskan bahwa implementasi DecisionTreeClassifier di Scikit-learn mengadopsi algoritma CART yang dioptimalkan."
    },
    {
        "kind": "split",
        "title": "Decision Boundaries",
        "body": "Karakteristik batas keputusan pohon keputusan:\n\n### Berorientasi Sumbu (*Axis-Aligned*)\n- Garis batas pemisah selalu tegak lurus terhadap sumbu fitur individual ($X_j = \\theta$).\n- Tidak menghasilkan garis miring atau diagonal secara langsung.\n\n### Sensitivitas terhadap Rotasi Data\n- Jika pola sejati data berbentuk diagonal linear, pohon memerlukan struktur tangga bertingkat banyak (*staircase partition*) untuk mendekatinya.\n- Menyebabkan pohon rentan terhadap variasi orientasi koordinat data.",
        "notes": "Tunjukkan ilustrasi bagaimana pohon membagi bidang 2D menjadi petak-petak persegi panjang."
    },
    {
        "kind": "split",
        "title": "Tree Depth & Model Complexity",
        "body": "Pengaruh kedalaman pohon (*Tree Depth*):\n\n### Pohon Dangkal (Shallow Tree, $d \\le 2$)\n- Memiliki bias tinggi dan varians rendah (*Underfitting*).\n- Aturan sangat sederhana, terkadang gagal menangkap interaksi multi-fitur yang halus.\n\n### Pohon Dalam (Deep Tree, $d \\ge 10$)\n- Memiliki bias rendah dan varians sangat tinggi (*Overfitting*).\n- Pohon menghafal noise spesifik pada data latih; setiap daun hanya menampung 1 sampel.",
        "notes": "Bandingkan bias-variance trade-off pohon dengan derajat polinomial dari Bab 13."
    },
    {
        "kind": "content",
        "title": "Overfitting in Decision Trees",
        "body": "Mengapa pohon keputusan tanpa batasan sangat rentan terhadap overfitting?\n\n- **Kapasitas Pemisahan Tak Terbatas**: Pohon dapat terus bercabang hingga setiap daun mencapai kemurnian 100% ($I_G = 0$).\n- **Sensitivitas terhadap Outlier**: Outlier tunggal dapat menciptakan cabang buatan yang sangat sempit dan tidak relevan di dunia nyata.\n- **Gejala Klinis**: Akurasi latih mencapai 100%, namun akurasi uji anjlok drastis (*generalization error* tinggi).\n- **Solusi Rekayasa**: Menerapkan pemangkasan (*pruning*) dan regularisasi kompleksitas.",
        "notes": "Tunjukkan contoh empiris dari praktikum di mana unpruned tree memiliki gap akurasi 10% antara train dan test."
    },
    {
        "kind": "split",
        "title": "Tree Pruning Techniques",
        "body": "Dua strategi utama pemangkasan pohon:\n\n### Pre-Pruning (Early Stopping)\n- Menghentikan pertumbuhan pohon sebelum menjadi terlalu kompleks.\n- Hyperparameter: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`.\n\n### Post-Pruning (Cost-Complexity Pruning)\n- Membiarkan pohon tumbuh penuh, lalu memangkas cabang yang kurang signifikan dari bawah ke atas.\n- Fungsi Biaya Minimal: $R_\\alpha(T) = R(T) + \\alpha |T|$, di mana $|T|$ adalah jumlah daun.",
        "notes": "Jelaskan bahwa post-pruning sering kali lebih superior karena tidak melewatkan interaksi fitur penting di cabang bawah."
    },
    {
        "kind": "content",
        "title": "Ensemble Learning Foundations",
        "body": "Kekuatan agregasi model (*Wisdom of the Crowds*):\n\n- **Konsep Dasar**: Menggabungkan prediksi dari banyak model dasar (*base learners*) untuk menghasilkan model gabungan yang jauh lebih akurat dan stabil.\n- **Syarat Efektivitas Ensemble**:\n  1. *Model Diversity*: Setiap model dasar harus membuat kesalahan pada sampel yang berbeda (de-korelasi).\n  2. *Better than Random Guessing*: Setiap learner harus memiliki performa lebih baik dari tebakan acak ($> 50\\%$).\n- **Dua Pilar Utama**: Bagging (reduksi varians) dan Boosting (reduksi bias).",
        "notes": "Analogi teorem Condorcet's Jury Theorem: kelompok pemilih independen lebih akurat daripada satu orang ahli."
    },
    {
        "kind": "split",
        "title": "Bagging (Bootstrap Aggregating)",
        "body": "Prinsip kerja Bagging (Breiman, 1996):\n\n### Pelatihan Paralel Mandiri\n- Membuat $B$ subset data latih independen menggunakan teknik *Bootstrap Sampling*.\n- Setiap model dasar (misalnya Decision Tree penuh) dilatih secara independen secara paralel.\n\n### Agregasi Prediksi Final\n- **Klasifikasi**: Mayoritas suara (*Hard Voting*) atau rata-rata probabilitas (*Soft Voting*).\n- **Regresi**: Rata-rata aritmatika dari seluruh prediksi pohon: $\\hat{y} = \\frac{1}{B} \\sum_{b=1}^{B} f_b(x)$.",
        "notes": "Jelaskan bahwa Bagging secara matematis membagi varians model dengan faktor mendekati B jika model independen."
    },
    {
        "kind": "content",
        "title": "Bootstrap Sampling",
        "body": "Pengambilan sampel acak dengan pengembalian (*sampling with replacement*):\n\n- **Mekanisme**: Dari dataset berukuran $N$, ambil $N$ sampel secara acak dengan pengembalian.\n- **Peluang Sampel Tidak Terpilih**:\n  $$P(\\text{tidak terpilih}) = \\left(1 - \\frac{1}{N}\\right)^N \\xrightarrow[N \\to \\infty]{} e^{-1} \\approx 0.368$$\n- **Implikasi**: Setiap set bootstrap rata-rata hanya memuat $\\approx 63.2\\%$ sampel unik dari data asli; sisanya $\\approx 36.8\\%$ disebut sebagai sampel **Out-of-Bag (OOB)**.",
        "notes": "Tunjukkan pembuktian matematis limit euler yang mendasari keajaiban evaluasi OOB."
    },
    {
        "kind": "content",
        "title": "Random Forest Architecture",
        "body": "Penyempurnaan Bagging oleh Breiman (2001):\n\n- **Dua Sumber Keacakan (Dual Randomness)**:\n  1. *Row Sampling*: Bootstrap sampling pada baris data latih.\n  2. *Feature Subsampling*: Pada setiap split node, hanya $m \\approx \\sqrt{p}$ fitur yang dipilih secara acak sebagai kandidat.\n- **Tujuan De-korelasi**: Mencegah fitur yang sangat dominan terpilih di root node seluruh pohon; memastikan setiap pohon mempelajari pola yang berbeda.\n- **Hasil Akhir**: Varians ensemble turun drastis tanpa meningkatkan bias secara signifikan.",
        "notes": "Tekankan perbedaan vital antara Bagging biasa (semua fitur dievaluasi) vs Random Forest (fitur disub-sampel)."
    },
    {
        "kind": "split",
        "title": "Random Feature Selection",
        "body": "Aturan praktis pemilihan jumlah fitur per split ($m$):\n\n### Kasus Klasifikasi\n- Nilai default standar: $m = \\lfloor \\sqrt{p} \\rfloor$.\n- Jika terdapat $p = 16$ fitur sensor, maka hanya 4 fitur acak yang diuji di setiap titik percabangan.\n\n### Kasus Regresi\n- Nilai default standar: $m = \\lfloor \\frac{p}{3} \\rfloor$.\n- Memberikan trade-off optimal antara kekuatan individual pohon dan keragaman ensemble.",
        "notes": "Jelaskan bahwa di Scikit-learn parameter ini dikontrol melalui argumen max_features."
    },
    {
        "kind": "content",
        "title": "Out-of-Bag (OOB) Evaluation",
        "body": "Validasi internal gratis tanpa memerlukan validation set terpisah:\n\n- **Cara Kerja**: Untuk setiap sampel data $x_i$, evaluasi hanya dilakukan menggunakan kumpulan pohon yang **tidak menyertakan** $x_i$ dalam set bootstrap-nya.\n- **Keunggulan Metodologis**:\n  - Menghemat data: Seluruh dataset dapat digunakan untuk pelatihan tanpa harus memotong validation set.\n  - Sangat mendekati skor k-Fold Cross Validation tanpa biaya komputasi tambahan.\n- Diaktifkan di Scikit-learn dengan parameter `oob_score=True`.",
        "notes": "Buktikan dengan hasil eksperimen praktikum di mana skor OOB (93.7%) hampir identik dengan 5-fold CV (94.8%)."
    },
    {
        "kind": "content",
        "title": "Boosting Principles",
        "body": "Pembelajaran ensemble sekuensial yang adaptif:\n\n- **Filosofi Dasar**: Membangun model secara berurutan; setiap model baru bertugas memperbaiki kesalahan (*residual error*) yang dibuat oleh model sebelumnya.\n- **Base Learner**: Umumnya menggunakan *Weak Learners* (model sederhana dengan bias agak tinggi, seperti *Decision Stump* berkedalaman 1).\n- **Fokus Pelatihan**: Mengubah pembobotan data atau menyesuaikan gradien fungsi loss ke arah sampel yang sulit diprediksi.\n- **Tujuan Utama**: Mereduksi bias model secara dramatis.",
        "notes": "Kontraskan secara tegas: Bagging melatih model paralel independen, sedangkan Boosting melatih model sekuensial dependensial."
    },
    {
        "kind": "split",
        "title": "AdaBoost (Adaptive Boosting)",
        "body": "Mekanisme kerja AdaBoost (Freund & Schapire, 1997):\n\n### Pembobotan Ulang Sampel\n- Awalnya semua sampel memiliki bobot sama $w_i = \\frac{1}{N}$.\n- Sampel yang salah diklasifikasikan oleh learner ke-$m$ dinaikkan bobotnya; sampel benar diturunkan bobotnya.\n\n### Agregasi Tertimbang\n- Model dasar dengan akurasi tinggi diberikan suara (*weight*) $\\alpha_m$ lebih besar dalam agregasi final:\n  $$H(x) = \\text{sign}\\left( \\sum_{m=1}^{M} \\alpha_m h_m(x) \\right)$$",
        "notes": "Jelaskan mengapa AdaBoost sangat sensitif terhadap label noise atau outlier ekstrem karena bobotnya akan terus dilipatgandakan."
    },
    {
        "kind": "split",
        "title": "Gradient Boosting",
        "body": "Optimasi loss berbasis penurunan gradien (Friedman, 2001):\n\n### Fitting pada Pseudo-Residuals\n- Pohon baru tidak memprediksi label asli, melainkan memprediksi residu negatif gradien: $r_{im} = -\\left[\\frac{\\partial L(y_i, F(x_i))}{\\partial F(x_i)}\\right]$.\n\n### Regularisasi Shrinkage\n- Setiap pohon baru diskalakan dengan *Learning Rate* ($\\eta \\in (0, 1]$):\n  $$F_m(x) = F_{m-1}(x) + \\eta \\cdot h_m(x)$$\n  Mencegah model bergerak terlalu agresif menuju overfitting.",
        "notes": "Jelaskan bahwa Gradient Boosting adalah analogi dari Gradient Descent di ruang fungsi."
    },
    {
        "kind": "content",
        "title": "Bagging vs Boosting",
        "body": "Perbandingan komprehensif dua paradigma ensemble utama:\n\n| Dimensi Karakteristik | Bagging (Random Forest) | Boosting (Gradient Boosting) |\n|---|---|---|\n| **Arsitektur Latih** | Paralel independen | Sekuensial bertahap |\n| **Tujuan Reduksi** | Menurunkan **Varians** | Menurunkan **Bias** (lalu varians) |\n| **Model Dasar** | Pohon dalam (*deep trees*) | Pohon dangkal (*shallow trees*) |\n| **Sensitivitas Outlier** | Cenderung tahan / robust | Cenderung sensitif / rentan overfit |\n| **Kebutuhan Tuning** | Sedikit (relatif mudah) | Tinggi (`learning_rate`, `n_est`, `depth`) |",
        "notes": "Tabel ini adalah ringkasan konsep esensial yang sering muncul dalam ujian dan wawancara teknis AI/ML."
    },
    {
        "kind": "content",
        "title": "Feature Importance Analysis",
        "body": "Mengukur kontribusi fitur dalam model ensemble berbasis pohon:\n\n- **Mean Decrease in Impurity (MDI)**: Menghitung total penurunan Gini Impurity yang diatribusikan pada fitur tertentu di seluruh pohon dalam forest.\n- **Keterbatasan Kritis MDI**:\n  1. Bias terhadap fitur berkardinalitas tinggi (misal ID atau bilangan acak kontinu).\n  2. Mengaburkan kontribusi fitur yang saling berkorelasi (*collinear features*).\n- **Rekomendasi Modern**: Gunakan **Permutation Feature Importance** atau **SHAP Values** untuk validasi interpretasi yang andal.",
        "notes": "Ingatkan mahasiswa agar tidak langsung mempercayai skor MDI tanpa memeriksa sifat fitur data."
    },
    {
        "kind": "content",
        "title": "Model Comparison Benchmark",
        "body": "Hasil evaluasi empiris pada dataset sensor industri (5-Fold Stratified CV):\n\n| Arsitektur Model | CV Accuracy | CV Macro F1 | Test Accuracy | Fit Time |\n|---|---|---|---|---|\n| **Dummy Baseline** | 41.48% | 19.54% | 41.11% | 0.6 ms |\n| **Single Tree (Pruned)** | 91.48% | 91.20% | 90.00% | 2.6 ms |\n| **Bagging (50 Trees)** | 95.19% | 95.32% | 91.11% | 75.7 ms |\n| **Random Forest (100 Trees)** | 94.81% | 94.78% | 90.00% | 98.9 ms |\n| **Gradient Boosting (100 Trees)** | **96.30%** | **96.34%** | **93.33%** | 223.2 ms |\n\nGradient Boosting mencapai performa generalisasi tertinggi dengan F1 Macro 96.34%.",
        "notes": "Sampaikan bahwa peningkatan akurasi ensemble dibayar dengan peningkatan waktu komputasi pelatihan dan latensi inferensi."
    },
    {
        "kind": "code",
        "title": "Python Demonstration: Random Forest",
        "body": "Implementasi Scikit-learn Pipeline bebas kebocoran:\n\n```python\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import classification_report\n\n# Inisialisasi Random Forest dengan OOB Score\nrf = RandomForestClassifier(\n    n_estimators=100,\n    max_features='sqrt',\n    oob_score=True,\n    random_state=42\n)\nrf.fit(X_train, y_train)\n\nprint(f\"Skor OOB: {rf.oob_score_ * 100:.2f}%\")\nprint(f\"Akurasi Uji: {rf.score(X_test, y_test) * 100:.2f}%\")\n```",
        "notes": "Tunjukkan betapa mudah dan elegannya API Scikit-learn dalam mengonfigurasi forest dengan OOB scoring."
    },
    {
        "kind": "case",
        "title": "Case Study: Predictive Maintenance",
        "body": "Penerapan ensemble pada turbin pembangkit listrik geotermal:\n\n- **Masalah**: Kerusakan bantalan mendadak menyebabkan downtime senilai $100.000 per hari.\n- **Tantangan**: Fitur telemetri memiliki pola non-linier dan interaksi kompleks antar-variabel (suhu pendingin dan frekuensi vibrasi).\n- **Solusi**: Menerapkan Gradient Boosting dengan kalibrasi ambang batas alarm.\n- **Hasil**: Deteksi dini kegagalan bantalan meningkat menjadi 97%, menurunkan unplanned outage sebesar 85% dalam 12 bulan operasi.",
        "notes": "Bahas studi kasus nyata untuk menghubungkan teori matematika pohon dengan dampak ekonomi industri riil."
    },
    {
        "kind": "case",
        "title": "Discussion & Engineering Decisions",
        "body": "Pertanyaan reflektif untuk diskusi kelas:\n\n1. Jika Anda bekerja di sistem mikrokontroler (Edge IoT) dengan memori 64 KB, apakah Anda memilih Single Pruned Tree atau Random Forest?\n2. Mengapa Random Forest tidak memerlukan penanganan missing value yang rumit pada data non-tabular?\n3. Dalam kondisi data seperti apa AdaBoost akan berkinerja jauh lebih buruk daripada Bagging?",
        "notes": "Pandu diskusi mahasiswa mengenai trade-off komputasi memori vs akurasi model di lingkungan terbatas daya."
    },
    {
        "kind": "summary",
        "title": "Summary & Key Takeaways",
        "body": "Poin-poin kesimpulan esensial Chapter 15:\n\n- **Decision Trees** mempartisi ruang fitur secara rekursif berbasis kriteria impurity (Gini / Entropy); transparan namun rentan overfitting.\n- **Pruning** (pre dan post ccp-alpha) membatasi kompleksitas pohon untuk mempertahankan generalisasi.\n- **Bagging & Random Forest** melatih banyak pohon paralel; feature subsampling de-korelasi pohon untuk mereduksi varians.\n- **Boosting (AdaBoost & Gradient Boosting)** melatih weak learners sekuensial untuk mereduksi bias secara bertahap.\n- Evaluasi **Out-of-Bag (OOB)** memberikan estimasi performa generalisasi tanpa biaya k-Fold CV tambahan.",
        "notes": "Ulas kembali seluruh ringkasan konsep sebelum menutup sesi perkuliahan."
    },
    {
        "kind": "content",
        "title": "Academic References",
        "body": "Rujukan ilmiah primer dan dokumentasi teknis:\n\n- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.\n- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press.\n- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 1189-1232.\n- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.\n- Scikit-learn Developers. (2024). *Tree-based Models & Ensemble Methods Guide*.",
        "notes": "Tunjukkan 5 referensi akademik klasik dan modern yang menjadi rujukan materi hari ini."
    }
]

def build_slides():
    out_path = pathlib.Path(__file__).parent.parent / "slides" / "deck.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2)
    print(f"Successfully generated {len(slides)} slides at: {out_path}")

if __name__ == "__main__":
    build_slides()

