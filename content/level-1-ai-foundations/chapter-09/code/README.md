# Katalog Kode Sumber & Verifikasi Numerik — Chapter 09: AI Ethics, Bias & Responsible AI

Direktori ini memuat implementasi mandiri Python untuk evaluasi etika kecerdasan buatan, komputasi metrik formal keadilan kelompok (*group fairness*), demonstrasi *fairness-utility trade-off*, penyamaran data privasi dan serangan re-identifikasi (*quasi-identifier linkage attack*), serta kerangka kerja penilaian risiko etis sistem AI (*risk assessment matrix*).

---

## 1. Struktur Berkas & Dataset

```text
content/level-1-ai-foundations/chapter-09/code/
├── data/
│   ├── scholarship_historical.csv     # Dataset sintetis seleksi beasiswa (1.000 pelamar, disparitas historis)
│   └── ai_ethics_risk_register.csv    # Registri penilaian risiko etika AI (5 skenario dunia nyata)
├── historical_bias_demo.py            # Simulasi data historis, disparate impact & kaidah 80% EEOC
├── group_fairness_metrics.py          # Komputasi Demographic Parity, Equal Opportunity, TPR, FPR, selection rate
├── fairness_tradeoff_demo.py          # Threshold tuning, mitigasi post-processing, teorema Kleinberg
├── privacy_masking_demo.py            # Masking, tokenisasi hash SHA-256, linkage attack pada DPT, UU PDP No. 27/2022
├── risk_assessment_demo.py            # Matriks risiko Kemungkinan x Dampak, standar NIST AI RMF / ISO 42001
└── README.md                          # Dokumentasi katalog dan hasil verifikasi eksekusi aktual (berkas ini)
```

---

## 2. Ringkasan Skrip dan Hasil Eksekusi Terverifikasi

Seluruh skrip telah diuji menggunakan Python 3.12.10 di dalam lingkungan virtual mandiri (`.venv`) dengan dependensi `pandas`, `numpy`, `scikit-learn`, dan `scipy`.

| Nama Skrip | Fungsi & Metrik Utama | Luaran & Temuan Kunci | Exit Code |
| :--- | :--- | :--- | :---: |
| `historical_bias_demo.py` | Simulasi 1.000 pelamar beasiswa dua kelompok demografis. | Skor tes objektif setara (73.51 vs 71.70), namun tingkat kelulusan historis timpang (45.93% vs 7.99%). Rasio disparate impact 0.1739 (gagal aturan 80%). | 0 |
| `group_fairness_metrics.py` | Evaluasi model Logistic Regression baseline pada data uji ($n=300$). | Selection rate A 44.39% vs B 15.12% ($\Delta DP = 0.2928$). TPR A 78.26% vs B 29.55% ($\Delta EO = 0.4872$). Terbukti diskriminasi sistemik. | 0 |
| `fairness_tradeoff_demo.py` | Mitigasi pasca-pemrosesan ambang batas terpisah ($\tau_A, \tau_B$). | Menurunkan $\tau_B$ dari 0.50 ke 0.10 menekan $\Delta EO$ dari 0.4872 menjadi 0.3735, dengan akurasi global tetap terjaga tinggi (81.00%). | 0 |
| `privacy_masking_demo.py` | Demonstrasi hashing salt & masking NIK serta serangan tautan DPT. | Berhasil mere-identifikasi rekam medis sensitif Eko Prasetyo melalui kombinasi quasi-ID (Usia 26, L, 35145). Landasan UU PDP Pasal 1 & 35. | 0 |
| `risk_assessment_demo.py` | Evaluasi matriks risiko Likelihood (1-5) $\times$ Impact (1-5). | 3 skenario berisiko TINGGI/Kritis (Seleksi Beasiswa [20], Radiologi Medis [15], Facial Recognition [16]), 2 skenario SEDANG. Ekspor ke CSV. | 0 |

---

## 3. Petunjuk Menjalankan Skrip

Pastikan lingkungan virtual Python aktif sebelum menjalankan skrip:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 1. Menghasilkan dataset dan mengaudit ketimpangan historis
python content/level-1-ai-foundations/chapter-09/code/historical_bias_demo.py

# 2. Menghitung metrik keadilan kelompok formal
python content/level-1-ai-foundations/chapter-09/code/group_fairness_metrics.py

# 3. Mengeksplorasi trade-off keadilan vs akurasi utilitas
python content/level-1-ai-foundations/chapter-09/code/fairness_tradeoff_demo.py

# 4. Membuktikan kerentanan re-identifikasi privasi data
python content/level-1-ai-foundations/chapter-09/code/privacy_masking_demo.py

# 5. Mengevaluasi matriks risiko etis AI
python content/level-1-ai-foundations/chapter-09/code/risk_assessment_demo.py
```

