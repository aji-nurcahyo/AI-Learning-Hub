"""
Script builder untuk slides deck Chapter 13: deck.json (30 slide)
Chapter 13: Supervised Learning — Regression Algorithms
"""

import json
from pathlib import Path

slides = [
    {
        "kind": "cover",
        "title": "Regression Algorithms",
        "body": "### Formulasi Matematis, Ordinary Least Squares, Polinomial & Regularisasi Modern\n\n**Level 2 — Machine Learning | Chapter 13**\n\nArtificial Intelligence Learning Hub — Institut Teknologi Sumatera (ITERA)\n*Created & Developed by Aji Pamungkas Tri Nurcahyo*",
        "notes": "Selamat datang di perkuliahan Chapter 13: Supervised Learning — Regression Algorithms. Hari ini kita melangkah ke pemodelan prediktif untuk variabel target kontinu, mencakup OLS, ekspansi polinomial, serta teknik regularisasi Ridge, Lasso, dan ElasticNet."
    },
    {
        "kind": "content",
        "title": "Regression Applications",
        "body": "Penerapan nyata algoritma regresi di industri modern:\n\n- **Smart Grid & Energi**: Memprediksi beban konsumsi daya puncak pabrik (kW) untuk mengoptimasi efisiensi pembangkit.\n- **Estimasi Umur Komponen (RUL)**: Menaksir sisa jam operasional turbin sebelum keausan bantalan memicu kerusakan fatal.\n- **Sensor Virtual (Soft Sensor)**: Mengestimasi variabel kimia atau termal yang sulit diukur langsung berdasarkan kombinasi sensor lain.\n- **Peramalan Finansial & Logistik**: Memperkirakan waktu tempuh pengiriman dan permintaan armada logistik.",
        "notes": "Tunjukkan bahwa regresi adalah salah satu algoritma paling banyak digunakan di seluruh sektor industri kuantitatif."
    },
    {
        "kind": "content",
        "title": "Learning Objectives",
        "body": "Setelah menyelesaikan bab ini, mahasiswa diharapkan mampu:\n\n- **Menjelaskan** konsep Regression dalam Supervised Learning dan perbedaannya dengan Classification.\n- **Membedakan** Simple dan Multiple Linear Regression.\n- **Menjelaskan** metode estimasi Ordinary Least Squares (OLS) dan fungsi biaya MSE.\n- **Menginterpretasikan** koefisien slope, intersep, dan galat residual secara fisik.\n- **Menggunakan** Polynomial Regression dan memahami linearitas terhadap parameter.\n- **Menjelaskan dan membedakan** teknik regularisasi Ridge (L2), Lasso (L1), dan ElasticNet.\n- **Menghitung** metrik MAE, MSE, RMSE, serta menjelaskan fenomena R² bernilai negatif.\n- **Mengevaluasi** asumsi Gauss-Markov melalui analisis diagnostik residual.",
        "notes": "Peta kompetensi yang akan dicapai dalam perkuliahan hari ini."
    },
    {
        "kind": "split",
        "title": "Regression Problem",
        "body": "Memahami karakteristik masalah regresi:\n\n### Ruang Target Kontinu\n- Target keluaran adalah bilangan riil tak terhingga: $y \\in \\mathbb{R}$.\n- Menjawab pertanyaan kuantitatif: *\"Berapa banyak?\"* atau *\"Berapa besarannya?\"*\n\n### Tantangan Metrik\n- Peluang menebak nilai tepat hingga angka desimal tak terhingga adalah nol.\n- Evaluasi berfokus pada meminimalkan jarak galat ($|y - \\hat{y}|$) antara estimasi model dan kenyataan.",
        "notes": "Bandingkan dengan klasifikasi diskrit yang hanya memprediksi kategori."
    },
    {
        "kind": "content",
        "title": "Linear Regression",
        "body": "Prinsip dasar Regresi Linier (Linear Regression):\n\n- **Definisi**: Model matematika yang memodelkan hubungan antara fitur masukan $X$ dan target $y$ menggunakan fungsi linier tertimbang.\n- **Garis Hipotesis**: Membentuk garis lurus pada ruang 2D, bidang datar pada ruang 3D, dan *hyperplane* pada ruang berdimensi tinggi.\n- **Keunggulan**: Sangat cepat dilatih, stabil, dan memberikan interpretasi koefisien yang transparan (*white-box model*).\n- **Keterbatasan**: Rentan underfitting jika fenomena dunia nyata memiliki hubungan non-linear yang kuat.",
        "notes": "Jelaskan analogi menarik garis lurus terbaik di antara sebaran titik data."
    },
    {
        "kind": "split",
        "title": "Simple vs Multiple Regression",
        "body": "Perbandingan dimensi prediktor:\n\n### Simple Linear Regression\n- **Satu Fitur Masukan**: $\\hat{y} = w x + b$.\n- Hubungan univariat sederhana (misal: beban mesin terhadap konsumsi daya).\n\n### Multiple Linear Regression\n- **Banyak Fitur Masukan**: $\\hat{y} = w_1 x_1 + w_2 x_2 + \\dots + w_D x_D + b$.\n- Hubungan multivariat terintegrasi (suhu, putaran RPM, tekanan, dan beban).\n- Mengukur efek marginal tiap variabel dengan asumsi fitur lain konstan.",
        "notes": "Jelaskan konsep ceteris paribus saat menginterpretasikan koefisien pada regresi berganda."
    },
    {
        "kind": "content",
        "title": "Coefficient and Intercept",
        "body": "Arti fisik dari parameter model regresi:\n\n- **Slope Coefficient ($w_j$)**: Besarnya laju perubahan pada target $y$ untuk setiap penambahan 1 satuan pada fitur $x_j$:\n  $$w_j = \\frac{\\partial \\hat{y}}{\\partial x_j}$$\n- **Intercept ($b$)**: Nilai taksiran target ketika seluruh fitur input bernilai nol ($x = 0$).\n- **Contoh Kasus Daya Listrik**:\n  - $w_{\\text{load}} = 2.4640$ kW/% : Tiap kenaikan $1\\%$ beban menaikkan daya sebesar $+2.46$ kW.\n  - $b = 3.3049$ kW : Konsumsi daya dasar sistem saat kondisi siaga murni.",
        "notes": "Tekankan pentingnya memperhatikan satuan fisik saat menginterpretasikan angka koefisien."
    },
    {
        "kind": "split",
        "title": "Residuals",
        "body": "Penyimpangan data aktual terhadap model:\n\n### Definisi Residual\n- Selisih antara nilai nyata dan nilai prediksi: $e_i = y_i - \\hat{y}_i$.\n- Merupakan bagian variasi data yang **gagal ditangkap** oleh model.\n\n### Sifat Residual Ideal\n- Rata-rata galat mendekati nol: $\\mathbb{E}[e] = 0$.\n- Tersebar secara acak tanpa pola terstruktur (*white noise*).\n- Varians seragam di seluruh rentang prediksi (Homoskedastik).",
        "notes": "Tunjukkan bahwa residual positif berarti model meremehkan (underpredict), residual negatif berarti melebih-lebihkan (overpredict)."
    },
    {
        "kind": "content",
        "title": "Ordinary Least Squares (OLS)",
        "body": "Metode estimasi kuadrat terkecil biasa (*Ordinary Least Squares*):\n\n- **Prinsip Dasar**: Memilih parameter bobot $\\mathbf{w}$ yang meminimalkan jumlah kuadrat dari seluruh galat residual:\n  $$\\min_{\\mathbf{w}} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2 = \\min_{\\mathbf{w}} \\|\\mathbf{y} - X\\mathbf{w}\\|_2^2$$\n- **Teorema Gauss-Markov**: Membuktikan bahwa estimator OLS adalah yang terbaik, linier, dan tidak bias (*Best Linear Unbiased Estimator / BLUE*) jika asumsi terpenuhi.\n- **Sensitivitas Outlier**: Karena galat dikuadratkan, OLS sangat sensitif terhadap titik data pencilan ekstrem.",
        "notes": "Jelaskan sejarah penemuan OLS oleh Gauss dan Legendre."
    },
    {
        "kind": "split",
        "title": "Loss Function",
        "body": "Fungsi kerugian dan penalti pada OLS:\n\n### Mengapa Menguadratkan Galat?\n- Menghilangkan tanda negatif (galat $+5$ dan $-5$ tidak saling meniadakan).\n- Menghasilkan fungsi konveks mulus yang memiliki tepat satu titik minimum global.\n\n### Formulasi Biaya Rata-Rata\n$$\\text{MSE}(\\mathbf{w}) = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\mathbf{w}^T \\mathbf{x}_i)^2$$\n$$\\mathcal{L}_{\\text{teori}}(\\mathbf{w}) = \\frac{1}{2N} \\|\\mathbf{y} - X\\mathbf{w}\\|_2^2$$",
        "notes": "Tunjukkan bahwa fungsi biaya MSE berbentuk mangkuk parabola multidimensi."
    },
    {
        "kind": "split",
        "title": "Gradient Descent Review",
        "body": "Dua pendekatan komputasi mencari solusi bobot optimal:\n\n### 1. Normal Equation (Analitis)\n- Rumus: $\\mathbf{w}^* = (X^T X)^{-1} X^T \\mathbf{y}$.\n- Langsung eksak tanpa iterasi; kompleksitas $\\mathcal{O}(D^3)$.\n- Sangat lambat jika jumlah fitur $D > 10.000$.\n\n### 2. Gradient Descent (Iteratif)\n- Pembaruan berulang: $\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\nabla J(\\mathbf{w})$.\n- Skalabel untuk dataset masif berjuta baris dan fitur.",
        "notes": "Ingatkan kembali kaitan dengan Bab 06 tentang Gradient Descent."
    },
    {
        "kind": "content",
        "title": "Regression Assumptions",
        "body": "Empat asumsi klasik Gauss-Markov agar estimator OLS valid:\n\n- **1. Linearitas Parameter**: Hubungan variabel dapat dinyatakan sebagai kombinasi linier koefisien $\\mathbf{w}$.\n- **2. Nilai Harapan Galat Nol**: $\\mathbb{E}[\\mathbf{e} \\mid X] = 0$ (tidak ada bias sistematis).\n- **3. Homoskedastisitas**: Varians residual bersifat konstan di seluruh skala prediksi (tidak berbentuk terompet).\n- **4. Ketiadaan Multikolinearitas Sempurna**: Matriks fitur $X^T X$ dapat dibalik (*full rank* / tidak ada fitur redundan identik).",
        "notes": "Tekankan bahwa jika asumsi ini dilanggar, estimasi koefisien dan p-value menjadi tidak dapat dipercaya."
    },
    {
        "kind": "split",
        "title": "Polynomial Regression",
        "body": "Memodelkan kelengkungan non-linear data:\n\n### Masalah Linier Kaku\n- Beban mesin tinggi memicu pemanasan kuadratik yang tidak bisa ditangkap garis lurus.\n\n### Solusi Ekspansi Polinomial\n- Memperluas fitur: $[x] \\to [x, x^2, x^3]$.\n- Kurva hipotesis melengkung secara elegan mengikuti pola fisik data.\n- **Kunci Matematis**: Tetap model linier terhadap parameter $\\mathbf{w}$!",
        "notes": "Tekankan paradoks: kurva non-linear pada input tetapi model linier pada parameter."
    },
    {
        "kind": "content",
        "title": "Polynomial Features",
        "body": "Transformasi ruang fitur menggunakan Scikit-learn:\n\n```python\npoly = PolynomialFeatures(degree=2, include_bias=False)\nX_poly = poly.fit_transform(X_train)\n```\n\n- **Ekspansi Dimensi**: Dari 3 fitur dasar $[x_1, x_2, x_3]$ berkembang menjadi 9 fitur mencakup suku kuadratik ($x_1^2, x_2^2, x_3^2$) dan interaksi silang ($x_1 x_2, x_1 x_3, x_2 x_3$).\n- **Wajib Pipeline**: `PolynomialFeatures` harus di-fit *strictly* pada data latih agar tidak membocorkan distribusi data uji.",
        "notes": "Peringatkan lonjakan dimensi fitur jika memilih derajat polinomial terlalu tinggi."
    },
    {
        "kind": "split",
        "title": "Underfitting and Overfitting",
        "body": "Dilema kompleksitas derajat polinomial:\n\n### Derajat 1: Underfitting\n- Bias tinggi, varians rendah.\n- Model terlalu kaku (Train RMSE $6.90$ kW, Test RMSE $6.83$ kW).\n\n### Derajat 2: Optimal (Good Fit)\n- Bias rendah, varians rendah (Test RMSE turun ke $5.62$ kW).\n\n### Derajat 4+: Overfitting\n- Bias rendah, varians tinggi.\n- Menghafal derau; kurva berosilasi liar (Test RMSE berbalik naik).",
        "notes": "Hubungkan dengan hasil eksperimen polynomial_regression.py."
    },
    {
        "kind": "content",
        "title": "Regularization",
        "body": "Mengendalikan kompleksitas model melalui penalti regularisasi:\n\n- **Masalah Bobot Besar**: Saat model mulai overfit, nilai koefisien $\\mathbf{w}$ meledak menjadi angka sangat besar untuk saling membatalkan derau.\n- **Prinsip Penalti**: Menambahkan suku biaya kompleksitas pada fungsi objektif:\n  $$\\mathcal{L}_{\\text{reg}}(\\mathbf{w}) = \\text{Loss}(\\mathbf{y}, X\\mathbf{w}) + \\alpha \\cdot \\Omega(\\mathbf{w})$$\n- **Efek Nyata**: Memaksa algoritma memilih bobot yang lebih kecil dan stabil, memangkas varians secara signifikan.",
        "notes": "Analogi: memberi rem pada model agar tidak melaju terlalu liar mengikuti derau."
    },
    {
        "kind": "split",
        "title": "Ridge Regression",
        "body": "Regularisasi L2 (Tikhonov Regularization):\n\n### Formulasi Fungsi Biaya\n$$J_{\\text{Ridge}}(\\mathbf{w}) = \\text{MSE} + \\alpha \\|\\mathbf{w}\\|_2^2 = \\text{MSE} + \\alpha \\sum_{j=1}^D w_j^2$$\n\n### Karakteristik Kunci\n- **Smooth Shrinkage**: Koefisien menyusut mendekati nol secara proporsional.\n- **Tanpa Fitur Nol**: Seluruh fitur dipertahankan (koefisien tidak pernah tepat 0).\n- **Solusi Multikolinearitas**: Menjamin matriks $(X^T X + \\alpha I)$ selalu dapat dibalik.",
        "notes": "Jelaskan bahwa Ridge sangat cocok saat kita meyakini semua fitur berkontribusi pada target."
    },
    {
        "kind": "split",
        "title": "Lasso Regression",
        "body": "Regularisasi L1 & Seleksi Fitur Otomatis:\n\n### Formulasi Fungsi Biaya\n$$J_{\\text{Lasso}}(\\mathbf{w}) = \\frac{1}{2N} \\|\\mathbf{y} - X\\mathbf{w}\\|_2^2 + \\alpha \\|\\mathbf{w}\\|_1 = \\text{Loss} + \\alpha \\sum_{j=1}^D |w_j|$$\n\n### Sifat Sparsity Unik\n- Kontur belah ketupat L1 memiliki sudut tajam pada sumbu koordinat.\n- Memaksa koefisien fitur yang tidak penting menjadi **tepat $0.0000$**.\n- Bertindak sebagai seleksi fitur otomatis (*embedded feature selection*).",
        "notes": "Tunjukkan bagaimana Lasso memangkas fitur tidak relevan pada eksperimen lasso_regression.py."
    },
    {
        "kind": "split",
        "title": "ElasticNet",
        "body": "Jalan tengah terbaik antara L1 dan L2:\n\n### Formulasi Penalti Gabungan\n$$\\text{Penalti} = \\alpha \\cdot \\rho \\|\\mathbf{w}\\|_1 + \\frac{\\alpha (1 - \\rho)}{2} \\|\\mathbf{w}\\|_2^2$$\n\n### Parameter l1_ratio ($\\rho$)\n- $\\rho = 1.0$ : Murni Lasso.\n- $\\rho = 0.0$ : Murni Ridge.\n- Mengatasi kelemahan Lasso pada sekelompok fitur yang saling berkorelasi tinggi.",
        "notes": "Jelaskan mengapa ElasticNet adalah pilihan default yang sangat aman saat menghadapi multikolinearitas."
    },
    {
        "kind": "content",
        "title": "Feature Scaling and Regularization",
        "body": "Penskalaan fitur adalah syarat mutlak sebelum regularisasi:\n\n- **Masalah Skala Fisik**: Penalti regularisasi $\\alpha \\sum w_j^2$ tidak mengenal satuan fisik.\n- **Ketidakadilan Penalti**: Variabel dengan rentang angka kecil akan memiliki bobot koefisien besar, sehingga menerima hukuman penalti yang jauh lebih berat dibanding variabel berangka besar.\n- **Solusi Standar**: Terapkan `StandardScaler` di dalam `Pipeline` Scikit-learn agar transformasi fit murni pada data latih dan diterapkan pada data uji.",
        "notes": "Tegaskan kembali bahwa melupakan StandardScaler sebelum Ridge/Lasso adalah kesalahan fatal."
    },
    {
        "kind": "split",
        "title": "MAE",
        "body": "Mean Absolute Error (Galat Mutlak Rata-rata):\n\n### Formulasi Matematis\n$$\\text{MAE} = \\frac{1}{N} \\sum_{i=1}^N |y_i - \\hat{y}_i|$$\n\n### Keunggulan Praktis\n- Memberikan nilai simpangan rata-rata dalam satuan fisik asli (kW).\n- **Robust Terhadap Outlier**: Memberikan bobot linier pada setiap galat tanpa menguadratkannya.",
        "notes": "Gunakan MAE saat menjelaskan hasil prediksi kepada manajer operasional non-teknis."
    },
    {
        "kind": "split",
        "title": "MSE and RMSE",
        "body": "Pengukuran galat berbasis kuadratik:\n\n### Mean Squared Error (MSE)\n- $\\text{MSE} = \\frac{1}{N} \\sum (y_i - \\hat{y}_i)^2$.\n- Satuan berkuadrat ($\\text{kW}^2$), sensitif terhadap lonjakan galat besar.\n\n### Root Mean Squared Error (RMSE)\n- $\\text{RMSE} = \\sqrt{\\text{MSE}}$.\n- Mengembalikan satuan ke bentuk linier (kW).\n- Semakin besar selisih $\\text{RMSE} - \\text{MAE}$, semakin banyak pencilan.",
        "notes": "Bandingkan satuan MSE vs RMSE."
    },
    {
        "kind": "content",
        "title": "R-squared and Negative R2",
        "body": "Koefisien determinasi ($R^2$) dan fenomena $R^2 < 0$:\n\n- **Formula**: $R^2 = 1 - (\\text{SS}_{\\text{res}} / \\text{SS}_{\\text{tot}})$.\n- **Interpretasi**: Proporsi variansi target yang berhasil dijelaskan oleh model.\n- **Mitos Nilai Negatif**: Pembelajar sering mengira $R^2$ selalu $[0, 1]$ karena ada simbol kuadrat.\n- **Kenyataan Out-of-Sample**: Pada data uji, jika model berkinerja lebih buruk daripada sekadar menebak nilai rata-rata $\\bar{y}$, maka $\\text{SS}_{\\text{res}} > \\text{SS}_{\\text{tot}}$, menghasilkan **$R^2 < 0$**!",
        "notes": "Tunjukkan hasil eksperimen di mana model naif menghasilkan R^2 = -14.20."
    },
    {
        "kind": "content",
        "title": "Residual Analysis",
        "body": "Pemeriksaan kesehatan model melalui diagnostik residual:\n\n- **1. Mean Residual $\\approx 0$**: Memastikan tidak ada bias sistematis terarah pada estimasi model.\n- **2. Uji Homoskedastisitas**: Varians sebaran titik residual harus seragam di beban rendah, sedang, dan tinggi (tidak membentuk corong).\n- **3. Uji Normalitas**: Histogram galat harus simetris (kemencengan *skewness* $\\approx 0$).\n- **4. Deteksi Outlier**: Sampel dengan galat $|e| > 3\\sigma$ patut diinvestigasi sebagai anomali sensor.",
        "notes": "Jelaskan bahwa residual analysis adalah langkah wajib sebelum model diserahkan ke produksi."
    },
    {
        "kind": "content",
        "title": "Cross-Validation for Regression",
        "body": "Validasi silang khusus pada tugas regresi kontinu:\n\n- **Standard K-Fold Cross-Validation**: Mengacak sampel ke dalam $k$ lipatan seragam (`shuffle=True, random_state=42`).\n- **Metrik Scoring Scikit-learn**: Menggunakan konvensi `scoring='neg_root_mean_squared_error'` (nilai negatif karena pengoptimal memaksimalkan skor).\n- **Pelaporan Statistik**: Laporkan rata-rata dan deviasi standar: $\\text{RMSE}_{\\text{CV}} = 6.17 \\pm 0.90$ kW.\n- **Tujuan Utama**: Menjamin model tidak bergantung pada keberuntungan pemisahan satu data uji acak.",
        "notes": "Jelaskan mengapa neg_root_mean_squared_error bertanda negatif di Scikit-learn."
    },
    {
        "kind": "content",
        "title": "Model Comparison",
        "body": "Hasil perbandingan 6 model regresi pada data telemetri daya:\n\n- **Dummy Baseline (Mean)**: RMSE $76.65$ kW | $R^2 = -0.0054$.\n- **Linear Regression (OLS)**: RMSE $5.37$ kW | $R^2 = 0.9951$ (Turun 92.99%).\n- **Polynomial Reg (Deg 2)**: **RMSE $4.12$ kW | $R^2 = 0.9971$ (Pemenang!)**.\n- **Ridge Regression (L2)**: RMSE $6.74$ kW | $R^2 = 0.9922$ (Stabil).\n- **Lasso Regression (L1)**: RMSE $5.51$ kW | $R^2 = 0.9948$ (Model Renggang).\n- **ElasticNet (L1+L2)**: RMSE $9.72$ kW | $R^2 = 0.9838$.",
        "notes": "Rangkum alasan mengapa Polynomial Derajat 2 paling unggul pada dataset ini."
    },
    {
        "kind": "split",
        "title": "Python Demonstration",
        "body": "Implementasi Scikit-learn Pipeline ringkas dan modular:\n\n```python\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler, PolynomialFeatures\nfrom sklearn.linear_model import Ridge\n\npipe = Pipeline([\n    ('poly', PolynomialFeatures(degree=2)),\n    ('scaler', StandardScaler()),\n    ('reg', Ridge(alpha=10.0))\n])\npipe.fit(X_train, y_train)\npreds = pipe.predict(X_test)\n```\n\n- Bebas dari kebocoran data (*zero-leakage*).\n- Siap diekspor sebagai artefak produksi terpadu.",
        "notes": "Tekankan pentingnya membungkus transformasi data ke dalam objek Pipeline."
    },
    {
        "kind": "content",
        "title": "Case Study & Limitations",
        "body": "Studi kasus industri dan keterbatasan yang harus diwaspadai:\n\n- **Korelasi Bukan Kausalitas**: Koefisien getaran positif $+5.28$ kW/mms tidak berarti kita bisa menghemat listrik dengan meredam sensor getaran secara paksa!\n- **Bahaya Ekstrapolasi**: Jangan gunakan model untuk memprediksi di luar rentang batas operasional pelatihan (misal suhu di bawah nol derajat).\n- **Pembusukan Model (Model Drift)**: Keausan mesin bertahap akan menggeser hubungan fisik, menuntut pelatihan ulang berkala.",
        "notes": "Ingatkan bahaya menyimpulkan kausalitas tanpa eksperimen kontrol acak."
    },
    {
        "kind": "split",
        "title": "Discussion & Questions",
        "body": "Pertanyaan reflektif untuk penguasaan materi:\n\n### Diskusi Kelas\n1. Mengapa Polynomial Regression tetap digolongkan sebagai model linier?\n2. Kapan Anda akan memilih Lasso dibandingkan Ridge untuk data sensor industri?\n3. Mengapa nilai $R^2$ tinggi tidak selalu menjamin model Anda bebas dari masalah?",
        "notes": "Buka sesi tanya jawab interaktif dengan mahasiswa."
    },
    {
        "kind": "content",
        "title": "Summary & References",
        "body": "Kesimpulan pembelajaran Chapter 13:\n\n- **Regresi** memodelkan target kontinu melalui minimisasi residual kuadratik OLS.\n- **Polinomial** menangkap kelengkungan fisik tanpa merusak linearitas parameter.\n- **Regularisasi** memangkas varians: Ridge menyusutkan bobot, Lasso memusnahkan fitur tidak relevan ($w=0$), dan ElasticNet memadukan keduanya.\n- **Referensi Otoritatif**:\n  - Géron, A. (2022). *Hands-On Machine Learning*, 3rd Edition.\n  - Hastie et al. (2009). *The Elements of Statistical Learning*, 2nd Edition.\n  - James et al. (2021). *An Introduction to Statistical Learning* (ISLR).\n  - CS2023 ACM/IEEE-CS/AAAI (2024). *Artificial Intelligence Area*.",
        "notes": "Tutup perkuliahan dengan menegaskan transisi ke algoritma klasifikasi diskrit di bab berikutnya."
    }
]


def main():
    deck_path = Path("content/level-2-machine-learning/chapter-13/slides/deck.json")
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deck_path, "w", encoding="utf-8") as f:
        json.dump(slides, f, indent=2, ensure_ascii=False)
    print(f"Successfully generated {len(slides)} slides at: {deck_path}")


if __name__ == "__main__":
    main()

