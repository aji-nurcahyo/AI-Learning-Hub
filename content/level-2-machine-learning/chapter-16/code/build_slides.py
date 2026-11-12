"""Pembangkit Berkas Slide Reveal.js (deck.json) untuk Chapter 16:
Support Vector Machines & k-Nearest Neighbors (36 Slide Terstruktur)
"""

import json
import pathlib

slides = [
    {
        "kind": "cover",
        "title": "Support Vector Machines & k-Nearest Neighbors",
        "body": "### Margin Maksimum, Variabel Slack, Kernel Trick, Instance-Based Learning & Geometri Ruang Metrik\n\n**Level 2 — Machine Learning | Chapter 16**\n\nArtificial Intelligence Learning Hub — Institut Teknologi Sumatera (ITERA)\n*Created & Developed by Aji Pamungkas Tri Nurcahyo*",
        "notes": "Selamat datang di perkuliahan Chapter 16. Hari ini kita membahas dua algoritma fundamental berbasis geometri dan jarak: Support Vector Machines (SVM) dan k-Nearest Neighbors (KNN)."
    },
    {
        "kind": "content",
        "title": "Why SVM and KNN?",
        "body": "Mengapa kita mempelajari SVM dan k-Nearest Neighbors bersamaan?\n\n- **Perspektif Geometris Bersama**: Keduanya memandang data sebagai titik-titik koordinat dalam ruang vektor berdimensi $D$.\n- **Sensitivitas Terhadap Metrik Jarak**: Keduanya sangat bergantung pada perhitungan jarak dan penskalaan fitur (*feature scaling*).\n- **Filosofi Pembelajaran Berlawanan**:\n  - SVM adalah model parametrik berbasis optimasi batas pemisah global (*eager learner*).\n  - KNN adalah model non-parametrik berbasis memori lokal (*lazy learner*).",
        "notes": "Jelaskan bahwa membandingkan model berbasis optimasi global (SVM) dengan model berbasis tetangga lokal (KNN) memberikan pemahaman mendalam tentang lanskap machine learning."
    },
    {
        "kind": "content",
        "title": "Learning Objectives",
        "body": "Setelah menyelesaikan perkuliahan ini, mahasiswa diharapkan mampu:\n\n- **Menjelaskan** prinsip Maximum Margin Separating Hyperplane pada SVM.\n- **Mengidentifikasi** Support Vectors dan perannya dalam menentukan batas keputusan.\n- **Membedakan** formulasi Hard-Margin dan Soft-Margin SVM berbasis variabel slack.\n- **Menerapkan** trik kernel (Linear, Polynomial, RBF) untuk batas non-linier.\n- **Mengoptimasi** parameter C dan Gamma pada model SVM.\n- **Menjelaskan** prinsip kerja k-Nearest Neighbors dan menghitung metrik jarak ($L_1, L_2$).\n- **Menganalisis** dampak penskalaan fitur (*feature scaling*) dan kutukan dimensi (*curse of dimensionality*).\n- **Membandingkan** kelebihan serta kekurangan SVM vs KNN secara komputasi dan memori.",
        "notes": "Paparkan tujuan pembelajaran kognitif terstandarisasi untuk bab ini."
    },
    {
        "kind": "split",
        "title": "Classification Decision Boundary",
        "body": "Batas pemisah antar-kelas dalam ruang fitur:\n\n### Masalah Batas Tak Hingga\n- Pada data yang terpisah linier, terdapat banyak sekali garis atau hiperbidang yang mampu membagi data dengan akurasi 100%.\n- Namun garis manakah yang paling kokoh dan memiliki kemampuan generalisasi terbaik?\n\n### Pendekatan Intuitif SVM\n- Memilih hiperbidang yang berada tepat di tengah-tengah kedua kelas.\n- Memaksimalkan 'zona aman' atau jarak ke titik data terdekat dari masing-masing kelas.",
        "notes": "Tunjukkan bahwa algoritma perseptron biasa sembarangan memilih garis asal memisahkan data, sedangkan SVM mencari garis terbaik secara matematis."
    },
    {
        "kind": "content",
        "title": "SVM Introduction",
        "body": "Sejarah dan konsep dasar Support Vector Machines (Vapnik & Chervonenkis, 1960-an; Cortes & Vapnik, 1995):\n\n- **Landasan Teori**: Berakar pada *Statistical Learning Theory* dan *Structural Risk Minimization* (SRM).\n- **Prinsip Utama**: Menemukan batas keputusan yang meminimalkan batas atas galat generalisasi (*generalization error bound*), bukan sekadar meminimalkan galat empiris data latih.\n- **Keluaran**: Pengklasifikasi biner atau multikelas (OvR/OvO) yang sangat tangguh terhadap overfitting pada data berdimensi sedang hingga tinggi.",
        "notes": "Kenalkan Vladimir Vapnik sebagai tokoh sentral di balik formulasi elegan SVM."
    },
    {
        "kind": "split",
        "title": "Separating Hyperplane",
        "body": "Definisi matematis hiperbidang pemisah:\n\n### Persamaan Hiperbidang\n$$\\mathbf{w}^T \\mathbf{x} + b = 0$$\n- $\\mathbf{w}$ adalah vektor bobot (*weight vector*) yang tegak lurus (*normal*) terhadap hiperbidang.\n- $b$ adalah bias (*intercept*) yang menentukan pergeseran hiperbidang dari titik asal.\n\n### Aturan Keputusan\n$$\\hat{y} = \\text{sign}(\\mathbf{w}^T \\mathbf{x} + b)$$\n- Sampel dengan $\\mathbf{w}^T \\mathbf{x} + b > 0$ diprediksi sebagai kelas positif ($+1$).\n- Sampel dengan $\\mathbf{w}^T \\mathbf{x} + b < 0$ diprediksi sebagai kelas negatif ($-1$).",
        "notes": "Jelaskan bahwa hiperbidang berdimensi D-1 membagi ruang fitur berdimensi D menjadi dua bagian."
    },
    {
        "kind": "content",
        "title": "Maximum Margin Principle",
        "body": "Mengapa margin terbesar menghasilkan generalisasi terbaik?\n\n- **Margin Geometris**: Jarak tegak lurus terpendek dari hiperbidang ke sampel data latih terdekat.\n- **Lebar Margin Total**:\n  $$\\text{Margin} = \\frac{2}{\\|\\mathbf{w}\\|_2}$$\n- **Tujuan Optimasi**: Memaksimalkan margin $\\frac{2}{\\|\\mathbf{w}\\|}$ ekuivalen dengan meminimalkan norma kuadrat bobot $\\frac{1}{2}\\|\\mathbf{w}\\|^2$:\n  $$\\min_{\\mathbf{w}, b} \\frac{1}{2} \\|\\mathbf{w}\\|^2 \\quad \\text{dengan syarat} \\quad y_i(\\mathbf{w}^T \\mathbf{x}_i + b) \\ge 1, \\quad \\forall i$$",
        "notes": "Tunjukkan bahwa meminimalkan ||w|| secara langsung melebarkan koridor margin pemisah."
    },
    {
        "kind": "content",
        "title": "Support Vectors",
        "body": "Sampel-sampel paling kritis penentu batas keputusan:\n\n- **Definisi**: Titik data latih yang terletak tepat di tepi margin ($y_i(\\mathbf{w}^T \\mathbf{x}_i + b) = 1$) atau melanggar margin.\n- **Karakteristik Unik**: \n  - Hiperbidang pemisah **hanya ditentukan secara eksklusif** oleh Support Vectors.\n  - Menghapus ribuan titik data lain yang berada jauh dari margin tidak akan mengubah orientasi model sama sekali!\n- **Efisiensi Memori**: Model SVM hanya perlu menyimpan subset kecil sampel sebagai support vectors untuk inferensi di masa depan.",
        "notes": "Tunjukkan bahwa support vectors adalah pilar penyangga koridor margin."
    },
    {
        "kind": "split",
        "title": "Hard-Margin SVM",
        "body": "Asumsi data terpisah linier sempurna tanpa derau:\n\n### Syarat Keras Mutlak\n- Seluruh sampel data latih **wajib** berada di sisi luar batas margin yang benar ($y_i(\\mathbf{w}^T \\mathbf{x}_i + b) \\ge 1$).\n- Tidak mengizinkan satu pun sampel melanggar koridor margin atau salah klasifikasi.\n\n### Keterbatasan Fatal di Dunia Nyata\n- Menjadi tidak fisibel (*infeasible problem*) jika ada dua titik kelas berbeda yang saling tumpang tindih.\n- Sangat rentan terhadap outlier tunggal yang dapat memutar arah margin secara dramatis.",
        "notes": "Gambarkan bagaimana satu titik pencilan dapat mempersempit margin hingga nyaris nol pada hard-margin."
    },
    {
        "kind": "split",
        "title": "Soft-Margin SVM",
        "body": "Relaksasi margin untuk data dunia nyata yang ber-derau (Cortes & Vapnik, 1995):\n\n### Toleransi Pelanggaran Margin\n- Mengizinkan beberapa titik data berada di dalam koridor margin atau bahkan salah sisi.\n- Memperkenalkan variabel kelonggaran (*slack variable* $\\xi_i \\ge 0$).\n\n### Kendala Relaksasi Baru\n$$y_i(\\mathbf{w}^T \\mathbf{x}_i + b) \\ge 1 - \\xi_i, \\quad \\xi_i \\ge 0$$\n- Jika $\\xi_i = 0$, sampel berada di luar margin (sempurna).\n- Jika $0 < \\xi_i \\le 1$, sampel berada di dalam koridor margin namun kelasnya benar.\n- Jika $\\xi_i > 1$, sampel salah diklasifikasikan.",
        "notes": "Jelaskan bahwa soft-margin adalah penyelamat yang membuat SVM dapat digunakan pada data riil."
    },
    {
        "kind": "content",
        "title": "Slack Variables & Penalty",
        "body": "Mengukur seberapa jauh suatu titik melanggar margin:\n\n- Nilai $\\xi_i$ merepresentasikan jarak relatif pelanggaran terhadap garis batas margin yang sah.\n- **Fungsi Objektif Soft-Margin**:\n  $$\\min_{\\mathbf{w}, b, \\boldsymbol{\\xi}} \\frac{1}{2} \\|\\mathbf{w}\\|^2 + C \\sum_{i=1}^{N} \\xi_i$$\n  di mana $C > 0$ adalah parameter regularisasi penalti pelanggaran.\n- Mengatur keseimbangan antara dua tujuan yang saling bertentangan: **memperlebar margin** vs **meminimalkan jumlah pelanggaran**.",
        "notes": "Jelaskan bahwa formulasi ini dapat ditransformasikan ke dalam masalah quadratic programming konveks."
    },
    {
        "kind": "split",
        "title": "Regularization Parameter C",
        "body": "Pengatur kompromi bias-varians pada SVM:\n\n### Nilai C Kecil (Low Variance, High Bias)\n- Penalti pelanggaran longgar; model memprioritaskan margin yang lebar.\n- Mengizinkan lebih banyak variabel slack; banyak support vectors.\n- Tahan terhadap noise, namun rentan underfitting jika terlalu kecil.\n\n### Nilai C Besar (High Variance, Low Bias)\n- Penalti pelanggaran sangat berat; model memaksakan sedikit kesalahan.\n- Koridor margin menyempit drastis; sedikit support vectors.\n- Sangat sensitif terhadap outlier lokal; rentan overfitting.",
        "notes": "Bandingkan parameter C pada SVM dengan alpha pada Ridge/Lasso (C berbanding terbalik dengan alpha)."
    },
    {
        "kind": "content",
        "title": "Nonlinear Classification",
        "body": "Bagaimana jika data sama sekali tidak dapat dipisahkan secara linier?\n\n- **Contoh Klasik**: Masalah XOR atau titik konsentris lingkaran (donat) di bidang 2D.\n- **Ide Dasar Cover's Theorem (1965)**:\n  \"Pola data non-linier di ruang berdimensi rendah memiliki probabilitas lebih tinggi untuk terpisah secara linier jika diproyeksikan ke ruang berdimensi lebih tinggi.\"\n- **Tantangan Komputasi**: Menghitung transformasi koordinat eksplisit $\\phi(\\mathbf{x})$ ke ruang berdimensi ribuan atau tak hingga membutuhkan memori dan komputasi masif.",
        "notes": "Visualisasikan bagaimana titik 2D di bidang lingkaran dapat dipisahkan dengan bidang datar setelah diangkat ke dimensi 3 z = x^2 + y^2."
    },
    {
        "kind": "split",
        "title": "The Kernel Trick",
        "body": "Keajaiban aljabar komputasi kernel:\n\n### Dual Problem SVM\n- Formulasi dual Lagrange SVM hanya membutuhkan operasi perkalian titik (*inner product*) antar-sampel: $\\langle \\phi(\\mathbf{x}_i), \\phi(\\mathbf{x}_j) \\rangle$.\n\n### Fungsi Kernel $K(\\mathbf{x}_i, \\mathbf{x}_j)$\n- Menghitung nilai inner product di ruang berdimensi tinggi **secara langsung di ruang fitur asli**, tanpa pernah menghitung koordinat $\\phi(\\mathbf{x})$ secara eksplisit!\n  $$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\langle \\phi(\\mathbf{x}_i), \\phi(\\mathbf{x}_j) \\rangle$$",
        "notes": "Tekankan bahwa trik kernel memungkinkan kita bekerja di ruang berdimensi tak hingga dengan biaya komputasi ruang asli."
    },
    {
        "kind": "content",
        "title": "Linear Kernel",
        "body": "Bentuk kernel paling sederhana dan tercepat:\n\n- **Formula Matematis**:\n  $$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\mathbf{x}_i^T \\mathbf{x}_j$$\n- **Karakteristik Operasional**:\n  - Tidak melakukan transformasi non-linier; murni dot product biasa.\n  - Menghasilkan batas keputusan berupa garis lurus atau bidang datar.\n- **Kapan Digunakan?**:\n  - Ketika jumlah fitur sangat banyak ($D \\gg N$), seperti pada klasifikasi teks TF-IDF atau data genomik.\n  - Memiliki waktu pelatihan dan latensi inferensi tercepat di antara semua kernel.",
        "notes": "Sampaikan bahwa jika fitur sudah sangat banyak, data sering kali sudah terpisah linier sehingga tidak butuh kernel kompleks."
    },
    {
        "kind": "content",
        "title": "Polynomial Kernel",
        "body": "Memodelkan interaksi antar-fitur hingga derajat tertentu:\n\n- **Formula Matematis**:\n  $$K(\\mathbf{x}_i, \\mathbf{x}_j) = (\\gamma \\mathbf{x}_i^T \\mathbf{x}_j + r)^d$$\n  di mana $d$ adalah derajat polinomial, $\\gamma$ adalah skala, dan $r$ adalah koefisien independen.\n- **Karakteristik Operasional**:\n  - Derajat $d=2$ menangkap interaksi kuadratik ($X_1^2, X_1 X_2, X_2^2$).\n  - Derajat $d=3$ menangkap interaksi kubik.\n- **Catatan Praktis**: Sangat sensitif terhadap skala fitur; derajat tinggi ($d > 5$) dapat menyebabkan ledakan nilai numerik (*overflow*).",
        "notes": "Tunjukkan bahwa derajat polinomial harus dibatasi untuk mencegah overfitting dan ledakan komputasi."
    },
    {
        "kind": "content",
        "title": "Radial Basis Function (RBF) Kernel",
        "body": "Kernel non-linier universal paling populer (Gaussian Kernel):\n\n- **Formula Matematis**:\n  $$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp\\left( -\\gamma \\|\\mathbf{x}_i - \\mathbf{x}_j\\|^2 \\right), \\quad \\gamma > 0$$\n- **Ruang Fitur Tersirat**:\n  Secara teoritis ekuivalen dengan memproyeksikan data ke **ruang Hilbert berdimensi tak terhingga** melalui deret Taylor eksponensial.\n- **Karakteristik**: Mengukur kemiripan (*similarity*) dua sampel; bernilai $1.0$ jika titik identik dan mendekati $0$ jika titik berjauhan.",
        "notes": "RBF adalah pisau bedah universal SVM yang mampu memisahkan pola pulau-pulau data paling kompleks sekalipun."
    },
    {
        "kind": "split",
        "title": "Gamma Parameter in RBF",
        "body": "Pengendali radius lonceng Gaussian pada RBF:\n\n### Nilai Gamma Kecil (Lonceng Lebar)\n- Radius pengaruh setiap support vector sangat luas.\n- Kontur batas keputusan menjadi sangat mulus dan mendekati linier.\n- Rentan underfitting jika $\\gamma$ terlalu kecil.\n\n### Nilai Gamma Besar (Lonceng Sempit)\n- Radius pengaruh support vector sangat sempit dan tajam.\n- Setiap titik data membentuk pulau keputusannya sendiri.\n- Menghafal noise secara ekstrem (Overfitting parah).",
        "notes": "Jelaskan analogi gamma seperti mengatur fokus sorotan lampu senter: lebar (difus) vs sempit (fokus tajam)."
    },
    {
        "kind": "code",
        "title": "SVM Pipeline Implementation",
        "body": "Praktik terbaik Scikit-learn Pipeline bebas kebocoran data:\n\n```python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.svm import SVC\n\n# Membangun pipeline SVM RBF\npipe_svm = Pipeline([\n    ('scaler', StandardScaler()),\n    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))\n])\n\npipe_svm.fit(X_train, y_train)\nprint(f\"Akurasi Uji: {pipe_svm.score(X_test, y_test)*100:.2f}%\")\nprint(f\"Total Support Vectors: {sum(pipe_svm.named_steps['svm'].n_support_)}\")\n```",
        "notes": "Tunjukkan bahwa StandardScaler harus berada di dalam Pipeline agar parameter mean dan std data uji tidak bocor."
    },
    {
        "kind": "content",
        "title": "KNN Introduction",
        "body": "Pengenalan k-Nearest Neighbors (Fix & Hodges, 1951; Cover & Hart, 1967):\n\n- **Filosofi Intuitif**: *\"Tunjukkan siapa tetangga terdekatmu, maka aku akan tahu siapa dirimu.\"*\n- **Karakteristik Pembelajaran**:\n  - **Non-Parametrik**: Tidak membuat asumsi eksplisit tentang distribusi probabilitas data atau bentuk fungsional batas pemisah.\n  - **Instance-Based / Memory-Based**: Tidak mempelajari model matematis kompak selama pelatihan; model adalah data itu sendiri.\n  - **Lazy Learning**: Menunda proses inferensi hingga ada data uji baru yang masuk.",
        "notes": "Bandingkan KNN dengan SVM: SVM membuat model di awal (eager), sedangkan KNN menunda komputasi hingga waktu prediksi (lazy)."
    },
    {
        "kind": "split",
        "title": "Instance-Based vs Eager Learning",
        "body": "Dua filosofi komputasi yang bertolak belakang:\n\n### Eager Learning (SVM, Trees, Regression)\n- **Waktu Pelatihan (Fit)**: Lambat / intensif komputasi (memecahkan optimasi matematis).\n- **Waktu Prediksi (Inferensi)**: Sangat cepat $O(D)$ atau $O(\\text{depth})$.\n- **Jejak Memori**: Ringkas; hanya menyimpan parameter bobot $\\mathbf{w}$ atau pohon.\n\n### Lazy Learning (KNN)\n- **Waktu Pelatihan (Fit)**: Nyaris instan ($0$ ms, hanya menyimpan data ke RAM).\n- **Waktu Prediksi (Inferensi)**: Lambat $O(N \\times D)$, harus menghitung jarak ke seluruh data!\n- **Jejak Memori**: Boros; wajib menyimpan seluruh data latih di RAM.",
        "notes": "Tabel perbandingan ini sangat fundamental dalam pemilihan arsitektur sistem cerdas di industri IoT."
    },
    {
        "kind": "content",
        "title": "Distance Metrics in KNN",
        "body": "Metrik pengukuran kedekatan titik dalam ruang berdimensi $D$:\n\n- **Jarak Euclidean ($L_2$ Norm)**: Garis lurus terpendek antar-dua titik:\n  $$d_{\\text{euc}}(\\mathbf{p}, \\mathbf{q}) = \\sqrt{\\sum_{i=1}^{D} (p_i - q_i)^2}$$\n- **Jarak Manhattan ($L_1$ Norm, Taxicab / City Block)**: Jumlah selisih absolut koordinat:\n  $$d_{\\text{man}}(\\mathbf{p}, \\mathbf{q}) = \\sum_{i=1}^{D} |p_i - q_i|$$\n- **Jarak Minkowski ($L_p$ Norm)**: Generalisasi parametrik:\n  $$d_{\\text{mink}}(\\mathbf{p}, \\mathbf{q}) = \\left( \\sum_{i=1}^{D} |p_i - q_i|^p \\right)^{1/p}$$",
        "notes": "Tunjukkan bahwa Manhattan Distance lebih tahan terhadap pencilan pada satu dimensi tertentu dibandingkan Euclidean."
    },
    {
        "kind": "case",
        "title": "Nearest Neighbor Calculation Example",
        "body": "Perhitungan manual jarak sensor uji ke dua data latih historis:\n\n- Titik Uji ($X_{\\text{uji}}$): Sinyal = $-60$ dBm, Latensi = $80$ ms.\n- Data Latih $A$ (Status NORMAL): Sinyal = $-55$ dBm, Latensi = $60$ ms.\n- Data Latih $B$ (Status KRITIS): Sinyal = $-80$ dBm, Latensi = $150$ ms.\n\n- **Jarak Euclidean ke $A$**:\n  $$d(X_{\\text{uji}}, A) = \\sqrt{(-60 - (-55))^2 + (80 - 60)^2} = \\sqrt{25 + 400} = \\sqrt{425} \\approx 20.61$$\n- **Jarak Euclidean ke $B$**:\n  $$d(X_{\\text{uji}}, B) = \\sqrt{(-60 - (-80))^2 + (80 - 150)^2} = \\sqrt{400 + 4900} = \\sqrt{5300} \\approx 72.80$$\nKarena $d(X_{\\text{uji}}, A) < d(X_{\\text{uji}}, B)$, titik uji diprediksi memiliki status NORMAL.",
        "notes": "Ajak mahasiswa memperhatikan bagaimana latensi mendominasi perhitungan jika data tidak distandarisasi."
    },
    {
        "kind": "split",
        "title": "Choosing the k Parameter",
        "body": "Pengatur fleksibilitas batas keputusan KNN:\n\n### Nilai k = 1 (Voronoi Tessellation)\n- Setiap titik data menguasai wilayah poligon terdekatnya.\n- Akurasi latih selalu 100% mutlak, namun sangat rapuh terhadap noise (High Variance).\n\n### Nilai k Terlalu Besar ($k \\approx N$)\n- Batas keputusan menjadi terlalu mulus dan kaku.\n- Prediksi selalu condong ke kelas mayoritas global populasi (High Bias / Underfitting).\n\n### Pemilihan Optimal\n- Melalui k-Fold Cross Validation; disarankan memilih bilangan ganjil untuk menghindari voting seri.",
        "notes": "Jelaskan bahwa memilih k ganjil (3, 5, 7) sangat efektif mencegah situasi dead-lock pada klasifikasi biner."
    },
    {
        "kind": "split",
        "title": "Majority vs Weighted Voting",
        "body": "Dua mekanisme penetapan kelas final dari $k$ tetangga:\n\n### Uniform Majority Voting\n- Setiap tetangga dari $k$ titik terdekat memiliki hak suara yang sama rata ($1$ suara per titik).\n- Berisiko salah jika tetangga ke-$k$ berada sangat jauh di perbatasan kelas.\n\n### Distance-Weighted Voting\n- Memberikan bobot suara berbanding terbalik dengan jarak ($w_i = \\frac{1}{d_i}$ atau $w_i = \\frac{1}{d_i^2}$).\n- Tetangga yang sangat dekat memiliki pengaruh dominan, mereduksi sensitivitas terhadap nilai $k$ yang terlalu besar.",
        "notes": "Tunjukkan hasil eksperimen di mana weighted voting meningkatkan akurasi dari 57.78% ke 62.22%."
    },
    {
        "kind": "content",
        "title": "The Curse of Dimensionality",
        "body": "Tantangan eksponensial di ruang berdimensi tinggi ($D \\gg 10$):\n\n- **Fenomena Ruang Kosong**: Volume ruang hiper-kubus tumbuh secara eksponensial seiring bertambahnya dimensi $D$, sehingga data menjadi sangat jarang (*sparse*).\n- **Jarak Kehilangan Makna**: Perbedaan antara jarak ke tetangga terdekat ($d_{\\min}$) dan tetangga terjauh ($d_{\\max}$) mendekati nol:\n  $$\\lim_{D \\to \\infty} \\frac{d_{\\max} - d_{\\min}}{d_{\\min}} = 0$$\n- **Dampak pada KNN**: Konsep 'tetangga terdekat' menjadi tidak bermakna karena semua titik sama-sama berjarak jauh.\n- **Solusi**: Reduksi dimensi (PCA) atau seleksi fitur relevan sebelum menjalankan KNN.",
        "notes": "Gambarkan konsep bola dalam kubus berdimensi 100 di mana hampir seluruh volume berada di sudut-sudut kubus."
    },
    {
        "kind": "content",
        "title": "The Absolute Need for Feature Scaling",
        "body": "Membuktikan kegagalan fatal ketiadaan standarisasi fitur:\n\n- Misalkan terdapat dua fitur sensor:\n  - $X_1$ (Latensi Jaringan): Rentang $10$ s/d $400$ ms (Varian $\\approx 1800$).\n  - $X_2$ (Tegangan Baterai): Rentang $2.1$ s/d $3.8$ V (Varian $\\approx 0.06$).\n- Dalam rumus Euclidean $\\sqrt{(X_{1a} - X_{1b})^2 + (X_{2a} - X_{2b})^2}$, kuadrat selisih latensi **40.000 kali lebih besar** daripada tegangan!\n- **Akibat**: Model bertindak seolah-olah tegangan baterai tidak ada sama sekali.\n- **Wajib**: Terapkan `StandardScaler()` atau `MinMaxScaler()` sebelum melatih SVM atau KNN.",
        "notes": "Tekankan berkali-kali bahwa scaling adalah hal yang mutlak dan tidak bisa ditawar pada bab ini."
    },
    {
        "kind": "content",
        "title": "KNN Decision Boundary Visualization",
        "body": "Karakteristik visual batas keputusan k-Nearest Neighbors:\n\n- **Bentuk Batas**: Sangat fleksibel, berbentuk kurva bergerigi atau pulau-pulau melengkung non-linier yang menyesuaikan kepadatan sampel lokal.\n- **Transisi Seiring Pertambahan k**:\n  - Pada $k=1$: Membentuk mosaik poligon Voronoi dengan pulau-pulau kecil di sekitar outlier.\n  - Pada $k=5$: Batas menjadi lebih mulus dan menghubungkan kluster sampel secara alami.\n  - Pada $k=25$: Batas merata menjadi garis lengkung lebar berskala makro.",
        "notes": "Visualisasikan diagram Voronoi di kelas untuk menunjukkan bagaimana ruang kartesius dipartisi oleh titik-titik tetangga."
    },
    {
        "kind": "content",
        "title": "SVM vs KNN Comparison Matrix",
        "body": "Perbandingan arsitektural komprehensif dua paradigma:\n\n| Parameter Karakteristik | Support Vector Machines (SVM) | k-Nearest Neighbors (KNN) |\n|---|---|---|\n| **Paradigma Belajar** | Model-Based (Eager Learning) | Instance-Based (Lazy Learning) |\n| **Batas Keputusan** | Optimal Margin Hyperplane | Batas Non-Parametrik Lokal |\n| **Kompleksitas Training** | $O(N^2 D)$ hingga $O(N^3 D)$ | **$O(1)$ (Hanya simpan data)** |\n| **Kompleksitas Prediksi** | **$O(D \\times N_{\\text{SV}})$ (Sangat Cepat)** | $O(N \\times D)$ (Lambat pada data besar) |\n| **Kebutuhan Memori** | Efisien (Hanya simpan Support Vectors)| Boros (Wajib simpan seluruh data) |\n| **Kutukan Dimensi** | Cenderung tahan via Kernel & Margin | Sangat rentan terhadap dimensi tinggi |",
        "notes": "Tabel ini menjadi panduan referensi utama pemilihan model dalam perancangan sistem produksi AI."
    },
    {
        "kind": "content",
        "title": "Model Performance Benchmark",
        "body": "Hasil evaluasi empiris pada dataset sensor nirkabel (5-Fold Stratified CV):\n\n| Arsitektur Model | CV Accuracy | CV Macro F1 | Test Accuracy | Fit Time | Pred Time |\n|---|---|---|---|---|---|\n| **Dummy Baseline** | 38.89% | 18.67% | 38.89% | 0.8 ms | 0.12 ms |\n| **Linear SVM** | 59.26% | 59.10% | 62.22% | 7.1 ms | 1.63 ms |\n| **Polynomial SVM (d=3)** | 53.33% | 53.28% | 60.00% | 5.9 ms | 1.54 ms |\n| **RBF SVM (Optimal)** | **67.78%** | **68.05%** | **70.00%** | 7.0 ms | 2.01 ms |\n| **KNN Uniform (k=5)** | 58.89% | 59.05% | 57.78% | 3.2 ms | 2.47 ms |\n| **KNN Weighted (k=5)**| 59.63% | 59.78% | 62.22% | 3.6 ms | 1.89 ms |\n\nRBF SVM unggul secara konsisten dengan akurasi uji 70.00% dan Macro F1 68.05%.",
        "notes": "Paparkan hasil numerik aktual dari eksperimen praktikum."
    },
    {
        "kind": "code",
        "title": "Python Demonstration: KNN & SVM",
        "body": "Implementasi Scikit-learn Pipeline mandiri:\n\n```python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.metrics import classification_report\n\n# Membangun KNN Pipeline dengan Distance Weighting\nknn_pipe = Pipeline([\n    ('scaler', StandardScaler()),\n    ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))\n])\n\nknn_pipe.fit(X_train, y_train)\ny_pred = knn_pipe.predict(X_test)\nprint(classification_report(y_test, y_pred, digits=4))\n```",
        "notes": "Tunjukkan bahwa kedua model menggunakan antarmuka Estimator API yang konsisten (fit, predict, score)."
    },
    {
        "kind": "case",
        "title": "Case Study: Smart Agriculture WSN",
        "body": "Penerapan pada jaringan sensor nirkabel pemantau perkebunan sawit cerdas:\n\n- **Skenario**: 500 simpul sensor nirkabel tersebar di area 100 hektar mengirimkan telemetri tanah, cuaca, dan daya baterai.\n- **Tantangan**: Simpul mengalami degradasi akibat kelembaban tinggi dan transmisi terhalang kanopi pohon sawit.\n- **Solusi**: Menggunakan RBF SVM terstandarisasi untuk mendeteksi simpul berstatus WARNING/CRITICAL sebelum komunikasi putus total.\n- **Dampak**: Waktu tanggap perbaikan teknisi berkurang dari 3 hari menjadi 4 jam; kehilangan data telemetri berkurang 92%.",
        "notes": "Bahas studi kasus nyata WSN di perkebunan cerdas untuk memberikan konteks dunia industri."
    },
    {
        "kind": "case",
        "title": "Discussion & Engineering Decisions",
        "body": "Pertanyaan reflektif untuk diskusi kelas:\n\n1. Mengapa pada masalah klasifikasi dokumen teks dengan 50.000 fitur kata, Linear SVM jauh lebih disukai daripada RBF SVM atau KNN?\n2. Mengapa KNN tidak dapat memberikan skor Feature Importance secara langsung sebagaimana Random Forest?\n3. Dalam skenario apa KNN dengan $k=1$ dapat mengalahkan algoritma lain yang jauh lebih kompleks?",
        "notes": "Ajak mahasiswa berdiskusi tentang trade-off komputasi dan interpretasi model."
    },
    {
        "kind": "summary",
        "title": "Summary & Key Takeaways",
        "body": "Poin-poin kesimpulan esensial Chapter 16:\n\n- **Support Vector Machines** mencari hiperbidang pemisah dengan margin terbesar; hanya bergantung pada subset kritis bernama **Support Vectors**.\n- **Soft-Margin SVM** menggunakan variabel slack $\\xi_i$ dan penalti $C$ untuk menyeimbangkan lebar margin vs toleransi kesalahan.\n- **Kernel Trick** (RBF, Poly) memetakan data non-linier ke ruang berdimensi tinggi tanpa biaya representasi eksplisit.\n- **k-Nearest Neighbors** adalah model lazy non-parametrik yang memprediksi berdasarkan mayoritas tetangga terdekat.\n- **Feature Scaling Wajib Hukumnya** untuk model geometris dan berbasis jarak agar tidak terjadi distorsi metrik.",
        "notes": "Ulas kembali ringkasan konsep sebelum mengakhiri sesi perkuliahan."
    },
    {
        "kind": "content",
        "title": "Academic References",
        "body": "Rujukan ilmiah primer dan dokumentasi teknis:\n\n- Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273-297.\n- Cover, T., & Hart, P. (1967). *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory, 13(1), 21-27.\n- Vapnik, V. (1998). *Statistical Learning Theory*. John Wiley & Sons.\n- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.\n- Scikit-learn Developers. (2024). *Support Vector Machines & Nearest Neighbors User Guide*.",
        "notes": "Tampilkan 5 referensi akademik klasik dan modern yang menjadi acuan penyusunan materi."
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

