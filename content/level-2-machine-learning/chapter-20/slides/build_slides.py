"""
Chapter 20 - Machine Learning Capstone
Script Pembangkit Slide Deck Reveal.js (deck.json)
Menghasilkan JSON array slide deck yang memenuhi skema validator AI Learning Hub.
"""

import json
from pathlib import Path

slides = [
    {
        "kind": "cover",
        "title": "Machine Learning Capstone Project",
        "body": "### Integrasi Menyeluruh Siklus Hidup Pembelajaran Mesin Level 2\n\n**Level 2 — Machine Learning | Chapter 20**\n\nArtificial Intelligence Learning Hub — Institut Teknologi Sumatera (ITERA)\n*Created & Developed by Aji Pamungkas Tri Nurcahyo*",
        "notes": "Selamat datang di sesi Capstone Level 2 Machine Learning. Sesi ini merupakan puncak integrasi seluruh konsep dari Bab 11 hingga 19 ke dalam solusi industri dunia nyata."
    },
    {
        "kind": "content",
        "title": "Tujuan Pembelajaran Capstone",
        "body": "- **Merumuskan** masalah bisnis menjadi spesifikasi teknis pembelajaran mesin yang terukur.\n- **Membangun** pipeline prapemrosesan bebas bocor data (*data leakage-free pipeline*).\n- **Menetapkan** baseline terukur dan menyeleksi kandidat model lintas paradigma Level 2.\n- **Menerapkan** optimasi hyperparameter dengan validasi silang berulang yang kredibel.\n- **Mengevaluasi** model pada *untouched test set* dengan metrik relevan industri.\n- **Mendiagnosis** galat, menyetel *decision threshold*, dan mengaudit transparansi model.",
        "notes": "Slide ini memetakan tujuan pembelajaran capstone berdasarkan taksonomi Bloom tingkat tinggi: evaluasi dan kreasi sistem terpadu."
    },
    {
        "kind": "content",
        "title": "8 Tahapan Baku Siklus Proyek Capstone",
        "body": "1. **Formulasi & Baseline**: Definisikan objektif bisnis, batasan operasional, dan metrik acuan awal.\n2. **Isolasi Data**: Pisahkan 20% data uji terisolasi sebelum transformasi apa pun.\n3. **Rekayasa Fitur**: Ekstraksi fitur domain fisik berbasis teori teknik.\n4. **Benchmark Model**: Uji model linier, berbasis pohon, dan kernel SVM.\n5. **Hyperparameter Tuning**: Optimasi sistematis menggunakan GridSearchCV.\n6. **Evaluasi Test Set**: Uji reliabilitas pada data yang belum pernah dilihat model.\n7. **Analisis Galat & Biaya**: Diseksi False Negative/Positive dan optimasi ambang batas.\n8. **Responsible AI**: Audit keadilan performa subkelompok dan penyusunan Model Card.",
        "notes": "Delapan tahapan ini merupakan standar rekayasa pembelajaran mesin profesional yang wajib diikuti oleh setiap peserta capstone."
    },
    {
        "kind": "content",
        "title": "Pilihan Topik Proyek Capstone (5 Jalur)",
        "body": "- **Jalur 1 (Proyek Acuan)**: Pemeliharaan Prediktif Kompresor Sentrifugal Industri.\n- **Jalur 2 (Fintech)**: Sistem Deteksi Penipuan Transaksi Keuangan Real-Time.\n- **Jalur 3 (Energi Berkelanjutan)**: Prediksi Daya Keluaran Pembangkit Listrik Tenaga Surya.\n- **Jalur 4 (Kesehatan Digital)**: Stratifikasi Risiko Kardiometabolik Pasien Rawat Jalan.\n- **Jalur 5 (E-Commerce)**: Prediksi *Customer Churn* dan Estimasi *Lifetime Value*.",
        "notes": "Peserta dapat memilih satu dari lima jalur proyek yang disediakan atau mengikuti jalur proyek acuan industri yang dibahas secara mendalam dalam modul."
    },
    {
        "kind": "content",
        "title": "Studi Kasus Acuan: Telemetri Kompresor Industri",
        "body": "- **Domain**: Pemantauan kesehatan mesin rotasi pabrik petrokimia.\n- **Populasi Data**: 600 sampel telemetri kontinu dari 12 unit pompa kompresor (`PUMP_01` hingga `PUMP_12`).\n- **Target Prediksi**: `machine_failure` (0 = Operasi Normal, 1 = Indikasi Kerusakan Kritis).\n- **Distribusi Kelas**: 243 Normal (40.5%) dan 357 Kerusakan (59.5%).\n- **Tantangan Utama**: Menghindari *catastrophic downtime* akibat *False Negative* yang tidak terdeteksi.",
        "notes": "Proyek acuan menggunakan data telemetri kompresor industri untuk melatih kepekaan mahasiswa terhadap trade-off biaya operasional dunia nyata."
    },
    {
        "kind": "content",
        "title": "Fitur Sensor Fisik Mesin",
        "body": "- `vibration_rms_g`: Akselerasi getaran efektif RMS ($g$) sesuai standar ISO 10816.\n- `bearing_temperature_c`: Suhu bantalan kompresor ($^\\circ\\text{C}$).\n- `rotational_speed_rpm`: Kecepatan putaran poros motor listrik.\n- `hydraulic_pressure_bar`: Tekanan oli pelumasan mekanis.\n- `motor_current_amp`: Konsumsi arus listrik yang ditarik oleh motor penggerak.\n- `oil_acidity_tan`: Nilai total asam (*Total Acid Number*) penanda degradasi pelumas.",
        "notes": "Mengenalkan enam parameter telemetri fisik mesin yang menjadi masukan mentah sebelum dilakukan proses rekayasa fitur."
    },
    {
        "kind": "content",
        "title": "Tahap 1: Formulasi Masalah & Baseline Acuan",
        "body": "- **Dummy Majority Baseline**: Selalu memprediksi kelas mayoritas (Kerusakan).\n  - Akurasi semu: 59.17%, namun ROC-AUC tepat 0.5000 (tidak ada kemampuan diskriminasi).\n- **Simple Logistic Regression Baseline**:\n  - Akurasi: 95.83%, F1-Score: 0.9655, ROC-AUC: 0.9914.\n- **Prinsip Rekayasa**: Model kompleks apa pun yang diajukan WAJIB melampaui performa baseline acuan sederhana ini.",
        "notes": "Menjelaskan mengapa baseline model sangat fundamental dalam proyek ML: untuk menetapkan batas bawah performa sebelum bereksperimen dengan model kompleks."
    },
    {
        "kind": "content",
        "title": "Tahap 2: Partisi Data & Pencegahan Kebocoran",
        "body": "- **Pemisahan Terisolasi**: 80% data latih (480 sampel) dan 20% data uji (120 sampel).\n- **Stratifikasi**: Mempertahankan proporsi kelas target yang identik di kedua partisi.\n- **Golden Rule Data Leakage**:\n  - Data uji tidak boleh disentuh selama rekayasa fitur, penentuan skala, maupun seleksi hyperparameter.\n  - Seluruh transformer wajib di-*fit* HANYA pada data latih di dalam *Scikit-Learn Pipeline*.",
        "notes": "Menekankan pemisahan data terisolasi dan penggunaan pipeline Scikit-Learn untuk mencegah kebocoran informasi dari data uji ke data latih."
    },
    {
        "kind": "content",
        "title": "Tahap 3: Rekayasa Fitur Berbasis Fisika Mesin",
        "body": "- **Rasio Beban Termal-Tekanan**: $\\text{Rasio} = \\frac{\\text{Suhu Bantalan}}{\\text{Tekanan Hidrolik}}$\n  - Menangkap hilangnya daya angkat film pelumas saat suhu meningkat drastis.\n- **Indeks Daya Getaran Mekanis**: $\\text{Indeks} = (\\text{Vibrasi RMS})^2 \\times \\left(\\frac{\\text{RPM}}{1000}\\right)$\n  - Mengukur energi kinetik destruktif yang diserap oleh struktur bantalan.\n- **Stres Pelumasan Listrik**: $\\text{Stres} = \\text{Arus Motor} \\times \\text{Keasaman Minyak}$\n  - Mendeteksi beban gesekan mekanis berlebih akibat degradasi kimiawi pelumas.",
        "notes": "Menjelaskan formulasi tiga fitur baru hasil rekayasa domain yang meningkatkan kapasitas pemisahan model terhadap pola kerusakan non-linier."
    },
    {
        "kind": "content",
        "title": "Tahap 4: Benchmark 4 Model Kandidat Level 2",
        "body": "- Evaluasi 5-Fold Stratified Cross-Validation pada data latih:\n  - **Logistic Regression (L2)**: Akurasi 92.08%, F1 0.9336, ROC-AUC 0.9818\n  - **Support Vector Machine (RBF)**: Akurasi 92.50%, F1 0.9373, ROC-AUC 0.9697\n  - **Random Forest Classifier**: Akurasi 92.29%, F1 0.9350, ROC-AUC 0.9762\n  - **Gradient Boosting Classifier**: Akurasi 91.88%, F1 0.9319, ROC-AUC 0.9747\n- **Keputusan**: Ensemble pohon dipilih untuk tahap penyetelan hyperparameter intensif.",
        "notes": "Membandingkan performa 4 model kandidat dari paradigma linier, kernel margin, bagging, dan boosting pada data latih menggunakan validasi silang 5-lipat."
    },
    {
        "kind": "content",
        "title": "Tahap 5: Hyperparameter Tuning Sistematis",
        "body": "- **Metodologi**: GridSearchCV dengan metrik F1-Score dan 5-Fold CV.\n- **Ruang Pencarian Random Forest**:\n  - `n_estimators`: [50, 100, 150]\n  - `max_depth`: [5, 8, 12, None]\n  - `min_samples_split`: [2, 5, 10]\n- **Konfigurasi Terbaik**: `max_depth=8`, `min_samples_split=10`, `n_estimators=150`.\n- **Hasil CV**: F1-Score meningkat dari 0.9350 menjadi **0.9390**.",
        "notes": "Menjelaskan proses penyetelan hyperparameter pada Random Forest untuk membatasi variansi model dan mengoptimalkan trade-off bias-variansi."
    },
    {
        "kind": "content",
        "title": "Tahap 6: Evaluasi Akhir pada Untouched Test Set",
        "body": "- **Karakteristik Data Uji**: 120 sampel terisolasi (49 Normal, 71 Kerusakan).\n- **Hasil Evaluasi Model Champion (Random Forest)**:\n  - **Akurasi**: 93.33%\n  - **Presisi**: 93.15%\n  - **Recall**: 95.77% (Hanya 3 kasus kerusakan yang lolos)\n  - **F1-Score**: 0.9444\n  - **ROC-AUC**: 0.9859\n- **Analisis**: Model mempertahankan daya generalisasi tinggi tanpa gejala overfitting.",
        "notes": "Menyajikan hasil verifikasi akhir pada data uji yang belum pernah dilihat model sebelumnya, membuktikan stabilitas generalisasi model."
    },
    {
        "kind": "split",
        "title": "Struktur Confusion Matrix pada Data Uji",
        "body": "Analisis terinci 120 sampel uji:\n\n### Kuadran Positif & Negatif\n- **True Negatives (TN = 44)**: Pompa sehat diprediksi beroperasi normal.\n- **False Positives (FP = 5)**: Pompa sehat keliru diprediksi rusak (alarm palsu, biaya ringan $200).\n- **False Negatives (FN = 3)**: Pompa rusak lolos tanpa deteksi (risiko downtime besar $5,000).\n- **True Positives (TP = 68)**: Kerusakan mesin berhasil diantisipasi tepat waktu.\n\n### Recall Kritis\n- **Recall 95.77%**: 68 dari 71 potensi kegagalan mesin fatal berhasil dicegah sebelum merusak fasilitas.",
        "notes": "Membedah matriks konfusi dan mengaitkan setiap kuadran dengan dampak nyata di lantai pabrik industri."
    },
    {
        "kind": "content",
        "title": "Tahap 7: Analisis Galat & Dampak Biaya Operasional",
        "body": "- **Matriks Biaya Industri**:\n  - Biaya False Negative (Kerusakan fatal tak terdeteksi): **$5,000**\n  - Biaya False Positive (Inspeksi teknisi tanpa kerusakan): **$200**\n- **Simulasi Penyetelan Ambang Batas (Threshold Tuning)**:\n  - Threshold default (0.50): 3 FN, 5 FP -> Total Risiko = **$16,000**\n  - Threshold optimal (0.30): 1 FN, 6 FP -> Total Risiko = **$6,200**\n- **Dampak Finansial**: Menurunkan ambang ke 0.30 menghemat risiko kerugian sebesar **$9,800 (61.25%)**.",
        "notes": "Membuktikan bahwa dalam penerapan industri nyata, metrik probabilitas harus dihubungkan dengan matriks biaya bisnis, bukan hanya ambang 0.5 default."
    },
    {
        "kind": "content",
        "title": "Tahap 8: Transparansi Penalaran (Explainability)",
        "body": "- **Permutation Feature Importance**:\n  1. `bearing_temperature_c`: Penurunan F1 rata-rata sebesar 0.1278 (faktor paling kritis)\n  2. `vibration_rms_g`: Penurunan F1 rata-rata sebesar 0.1217\n  3. `oil_acidity_tan`: Penurunan F1 rata-rata sebesar 0.0161\n- **Partial Dependence Profile (PDP)**:\n  - Begitu suhu bantalan melampaui $75^\\circ\\text{C}$, estimasi peluang kerusakan melonjak dari 45% ke >80%.\n- **Validasi Domain**: Temuan model selaras dengan batas termal pelumas ISO VG-46.",
        "notes": "Menjelaskan hasil analisis keterpahaman model yang memvalidasi bahwa model belajar pola fisika nyata, bukan sekadar korelasi palsu."
    },
    {
        "kind": "content",
        "title": "Tahap 9: Responsible AI & Audit Keadilan",
        "body": "- **Audit Antar Unit Mesin**:\n  - Disparitas performa F1 antar unit pompa (`PUMP_01` hingga `PUMP_12`) adalah 0.2000.\n  - Memenuhi batas ambang toleransi sistem (< 0.30).\n- **Batas Tata Kelola Aman (Governance Guardrails)**:\n  - Rekomendasi operasional: Sistem peringatan dini dengan intervensi manusia (*Human-in-the-Loop*).\n  - Larangan operasional: Tidak boleh digunakan sebagai sistem pemutus darurat otomatis (*emergency shutdown*) tanpa sertifikasi keselamatan SIL-3.",
        "notes": "Menjelaskan prinsip Responsible AI dalam proyek Capstone: audit disparitas performa subkelompok dan pendefinisian batasan aman penggunaan model."
    },
    {
        "kind": "content",
        "title": "Penyusunan Model Card Standar Industri",
        "body": "- **Model Details**: Nama, versi, arsitektur Random Forest (150 pohon), tanggal rilis.\n- **Intended Use**: Deteksi dini anomali pompa sentrifugal bertekanan kontinu.\n- **Factors & Subgroups**: Dievaluasi pada 12 unit mesin dan berbagai rezim suhu.\n- **Metrics**: F1-Score, Recall, ROC-AUC, dan Expected Monetary Value (EMV).\n- **Quantitative Analyses**: Evaluasi test set (N=120) dan profil PDP fitur fisik.\n- **Ethical & Operational Limits**: Wajib konfirmasi teknisi; dilarang untuk proteksi nuklir.",
        "notes": "Memperkenalkan artefak Model Card sebagai dokumen tata kelola standar industri untuk pelaporan model yang bertanggung jawab."
    },
    {
        "kind": "content",
        "title": "Rubrik Penilaian Proyek Capstone (100 Poin)",
        "body": "- **Formulasi Masalah & Baseline (15%)**: Kejelasan batasan bisnis dan metrik evaluasi acuan.\n- **Prapemrosesan & Pipeline (20%)**: Eksekusi pipeline bebas kebocoran dan rekayasa fitur domain.\n- **Pemodelan & Tuning (25%)**: Eksplorasi multi-model Level 2 dan validasi silang kredibel.\n- **Evaluasi & Analisis Galat (20%)**: Evaluasi test set terisolasi, diseksi FN/FP, dan optimasi threshold.\n- **Responsible AI & Dokumentasi (20%)**: Transparansi model, Model Card, dan tata kelola etis.",
        "notes": "Memaparkan rubrik penilaian objektif dan transparan yang digunakan dosen atau penilai dalam mengevaluasi laporan dan kode Capstone mahasiswa."
    },
    {
        "kind": "content",
        "title": "Checklist Kesiapan Pengumpulan Proyek",
        "body": "- [ ] Dataset terisolasi secara benar tanpa ada kebocoran informasi (*data leakage*).\n- [ ] Seluruh transformasi data dibungkus rapi dalam Scikit-Learn Pipeline.\n- [ ] Eksperimen membandingkan minimal 3 algoritma Level 2 yang berbeda arsitektur.\n- [ ] Evaluasi akhir menggunakan *test set* terpisah yang belum pernah dilihat model.\n- [ ] Disertai analisis biaya bisnis atau trade-off presisi-recall industri.\n- [ ] Dilengkapi dengan Model Card dan panduan mitigasi risiko kegagalan model.",
        "notes": "Daftar periksa praktis bagi mahasiswa sebelum mengumpulkan artefak kode, notebook, dan laporan akhir Capstone."
    },
    {
        "kind": "content",
        "title": "Panduan Presentasi Sidang Capstone",
        "body": "- **Durasi Standar**: 15 menit presentasi + 10 menit tanya jawab teknis.\n- **Struktur Slide Utama**:\n  - Menit 1-3: Latar belakang masalah bisnis dan kerugian finansial downtime.\n  - Menit 4-7: Desain pipeline, rekayasa fitur fisik, dan pencegahan leakage.\n  - Menit 8-11: Hasil benchmark, tuning model champion, dan confusion matrix test set.\n  - Menit 12-14: Analisis biaya threshold, PDP explainability, dan audit keadilan.\n  - Menit 15: Rekomendasi operasional dan Model Card.",
        "notes": "Memberikan panduan praktis presentasi teknis agar mahasiswa mampu mengomunikasikan hasil Capstone secara meyakinkan kepada penguji atau pemangku kepentingan bisnis."
    },
    {
        "kind": "summary",
        "title": "Kesimpulan & Pencapaian Level 2",
        "body": "- **Penguasaan Komprehensif**: Menguasai seluruh fondasi algoritma pembelajaran mesin tradisional dari regresi linier, klasifikasi, ensemble, hingga evaluasi mendalam.\n- **Disiplin Rekayasa**: Mampu membangun alur kerja ML yang modular, reproducible, dan teruji secara ketat.\n- **Kesiapan Industri**: Memiliki intuisi menghubungkan metrik matematika dengan keputusan bisnis dan etika AI.\n- **Langkah Berikutnya**: Selamat! Anda telah menyelesaikan Level 2 Machine Learning dan siap melangkah ke Level 3 Deep Learning.",
        "notes": "Menutup rangkaian modul Level 2 Machine Learning dengan perayaan pencapaian komprehensif mahasiswa sebelum memasuki Level 3 Deep Learning."
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
