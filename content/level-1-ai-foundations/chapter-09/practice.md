# Panduan Praktik Laboratorium — Chapter 09: AI Ethics, Bias & Responsible AI

Selamat datang di laboratorium **Chapter 09: AI Ethics, Bias & Responsible AI**. Modul praktikum ini membimbing Anda melakukan audit etika kecerdasan buatan, mengukur metrik formal keadilan kelompok (*group fairness*), mengeksplorasi mitigasi ambang batas (*threshold tuning*), menguji kerentanan privasi data di bawah rezim UU PDP (UU No. 27/2022) & GDPR, serta menyusun matriks penilaian risiko etis AI berdasarkan standar NIST AI RMF 1.0.

---

## 1. Tujuan Praktik

Setelah menyelesaikan seluruh langkah praktikum mandiri ini, Anda diharapkan mampu:
1. Mengidentifikasi manifestasi bias historis (*historical bias*) dan bias representasi pada dataset seleksi akademik.
2. Menghitung metrik keadilan kelompok formal: *Selection Rate*, *Demographic Parity Difference* ($\Delta DP$), *True Positive Rate* (TPR / Recall), *False Positive Rate* (FPR), dan *Equal Opportunity Difference* ($\Delta EO$).
3. Mengevaluasi kepatuhan hukum terhadap Kaidah Empat-Perlima (*Four-Fifths Rule / Disparate Impact*).
4. Menerapkan teknik mitigasi pasca-pemrosesan (*post-processing threshold tuning*) untuk menyamakan peluang seleksi kandidat minoritas tanpa merusak akurasi global secara signifikan.
5. Memahami batasan matematika Teorema Ketidakmungkinan Keadilan (*Impossibility Theorem of Fairness*, Kleinberg et al., 2016).
6. Menganalisis kerentanan serangan tautan *quasi-identifier* (*linkage attack*) pada data terpseudonimisasi dan menghubungkannya dengan definisi data pribadi dalam UU PDP No. 27/2022.
7. Menyusun matriks penilaian risiko dampak sosial dan etika AI (*AI Ethics Risk Register*) menggunakan skala Kemungkinan $\times$ Dampak.

---

## 2. Struktur Direktori Praktikum

Seluruh kode sumber mandiri, dataset sintetis, notebook interaktif, dan registri risiko tersimpan pada struktur direktori berikut:

```text
content/level-1-ai-foundations/chapter-09/
├── code/
│   ├── data/
│   │   ├── scholarship_historical.csv       # Dataset 1.000 pelamar beasiswa (disparitas masa lalu)
│   │   └── ai_ethics_risk_register.csv      # Matriks 5 profil risiko etis sistem AI dunia nyata
│   ├── historical_bias_demo.py              # Simulasi ketimpangan historis vs kemampuan murni
│   ├── group_fairness_metrics.py            # Komputasi Demographic Parity, Equal Opportunity, TPR, FPR
│   ├── fairness_tradeoff_demo.py            # Post-processing threshold tuning & trade-off analitis
│   ├── privacy_masking_demo.py              # Demonstrasi masking vs re-identifikasi quasi-identifier
│   ├── risk_assessment_demo.py              # Perhitungan skor risiko Likelihood x Impact & matriks CSV
│   └── README.md                            # Katalog kode dan tabel hasil eksekusi aktual
├── notebooks/
│   └── chapter_09_companion.ipynb           # Interactive Jupyter Companion Notebook (Format v4, 12 sel)
├── practice.md                              # Panduan praktikum laboratorium (berkas ini)
├── index.md                                 # Modul teori utama (Single Source of Truth)
└── metadata.json                            # Registri metadata chapter
```

---

## 3. Prasyarat Lingkungan & Persiapan

Praktikum ini menggunakan lingkungan virtual Python standar (`.venv`) dengan pustaka:
- `pandas` (analisis tabel dan penggabungan dataset quasi-ID)
- `numpy` (komputasi matriks dan penyesuaian threshold)
- `scikit-learn` (pelatihan model baseline LogisticRegression dan metrik confusion matrix)
- `hashlib` (pustaka standar enkripsi SHA-256 untuk tokenisasi data pribadi)

Aktivasi virtual environment:
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

Verifikasi ketersediaan pustaka:
```bash
python -c "import pandas, numpy, sklearn; print('Lingkungan laboratorium etika AI siap!')"
```

---

## 4. Panduan Eksekusi Langkah demi Langkah

### Langkah 1: Simulasi & Audit Ketimpangan Historis
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-09/code/historical_bias_demo.py
```
**Hal penting yang diamati:**
- Skor tes bakat skolastik objektif kedua kelompok disimulasikan setara: Kelompok A memiliki rata-rata $73.51$ dan Kelompok B memiliki rata-rata $71.70$.
- Proporsi kandidat yang berpotensi tinggi (*ground truth*) seimbang: $53.79\%$ (Kelompok A) vs $51.39\%$ (Kelompok B).
- Namun, tingkat kelulusan historis masa lalu sangat timpang: $45.93\%$ (Kelompok A) vs **hanya $7.99\%$ (Kelompok B)**!
- Rasio kelulusan hanya $0.1739$ (jauh di bawah batas $0.80$ Kaidah Empat-Perlima EEOC), membuktikan keberadaan *historical bias* yang parah.

### Langkah 2: Komputasi Metrik Keadilan Kelompok Formal
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-09/code/group_fairness_metrics.py
```
**Hal penting yang diamati:**
- Model klasifikasi baseline dilatih menggunakan label keputusan historis masa lalu.
- Akurasi model tampak memuaskan ($79.67\%$).
- Namun audit keadilan kelompok mengungkap diskriminasi sistemik:
  - *Selection Rate*: Kelompok A ($44.39\%$) vs Kelompok B ($15.12\%$), selisih $\Delta DP = 0.2928$.
  - *True Positive Rate* (Peluang kandidat berkemampuan tinggi untuk diterima): Kelompok A sebesar $78.26\%$, sedangkan Kelompok B **hanya $29.55\%$**!
  - Kesenjangan *Equal Opportunity* ($\Delta EO$) mencapai $0.4872$, melanggar batas toleransi etis ($< 0.10$).

### Langkah 3: Mitigasi Pasca-Pemrosesan (*Threshold Tuning*)
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-09/code/fairness_tradeoff_demo.py
```
**Hal penting yang diamati:**
- Mengapa menurunkan ambang batas ($\tau$) untuk Kelompok B dibenarkan secara etis? Karena skor probabilitas mereka tertekan oleh bias bobot masa lalu.
- Dengan menurunkan ambang batas $\tau_B$ dari $0.50$ ke $0.10$, TPR Kelompok B melonjak dari $29.55\%$ menjadi $40.91\%$, menekan kesenjangan $\Delta EO$ dari $0.4872$ ke $0.3735$.
- *Fairness-Utility Trade-off*: Akurasi global model tetap berada pada level optimal ($81.00\%$), membuktikan bahwa keadilan etis dapat ditingkatkan tanpa mengorbankan performa sistem secara fatal.

### Langkah 4: Pembuktian Kerentanan Privasi & Serangan Tautan
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-09/code/privacy_masking_demo.py
```
**Hal penting yang diamati:**
- Kolom NIK di-hash menggunakan SHA-256 dan nama disamarkan (`Ek********yo`).
- Namun kolom *Quasi-Identifiers* (Usia: 26, Kelamin: L, Kode Pos: 35145) dibiarkan terbuka.
- Melalui operasi *inner join* sederhana dengan data terbuka Daftar Pemilih Tetap (DPT), identitas asli Eko Prasetyo berhasil di-reidentifikasi beserta diagnosis medis sensitifnya (*Cardiac Arrhythmia*).
- **Landasan Hukum**: Sesuai Pasal 1 angka 1 UU PDP No. 27/2022, data gabungan tersebut tetap berstatus data pribadi. *Pseudonimisasi bukanlah anonimisasi mutlak!*

### Langkah 5: Penilaian Risiko Etis AI (*Risk Assessment Matrix*)
Jalankan skrip:
```bash
python content/level-1-ai-foundations/chapter-09/code/risk_assessment_demo.py
```
**Hal penting yang diamati:**
- Matriks mengevaluasi lima domain penerapan AI berdasarkan Kemungkinan ($1-5$) dan Dampak ($1-5$).
- Tiga sistem diklasifikasikan sebagai **TINGGI / Kritis** (Skor $\ge 15$): Seleksi Beasiswa Otomatis ($20$), Sistem Radiologi Medis ($15$), dan Pengawasan Kampus Wajah ($16$).
- Skrip mengekspor berkas registri risiko lengkap ke `code/data/ai_ethics_risk_register.csv`.

---

## 5. Eksplorasi Interaktif dengan Jupyter Notebook

Buka dan jalankan Jupyter Companion Notebook:
```bash
jupyter notebook content/level-1-ai-foundations/chapter-09/notebooks/chapter_09_companion.ipynb
```
Lakukan eksplorasi mandiri terhadap pengaruh perubahan ambang batas ($\tau$) terhadap trade-off keadilan.

---

## 6. Ringkasan Metrik Etika Kunci

| Metrik Etika | Formulasi Matematika | Kriteria Kepatuhan |
| :--- | :--- | :--- |
| **Demographic Parity** | $\mathbb{P}(\hat{Y}=1 \mid A=a) = \mathbb{P}(\hat{Y}=1 \mid A=b)$ | $\Delta DP = \lvert P_A - P_B \rvert < 0.10$ |
| **Disparate Impact (80% Rule)** | $\frac{\mathbb{P}(\hat{Y}=1 \mid A=\text{Minoritas})}{\mathbb{P}(\hat{Y}=1 \mid A=\text{Mayoritas})} \ge 0.80$ | Nilai rasio $\ge 0.80$ (EEOC Guidelines) |
| **Equal Opportunity** | $\mathbb{P}(\hat{Y}=1 \mid Y=1, A=a) = \mathbb{P}(\hat{Y}=1 \mid Y=1, A=b)$ | $\Delta EO = \lvert \text{TPR}_A - \text{TPR}_B \rvert < 0.10$ |
| **Equalized Odds** | Kesetaraan TPR dan FPR antar kelompok | $\Delta \text{TPR} < 0.10$ dan $\Delta \text{FPR} < 0.10$ |
| **Skor Risiko Etis** | $\text{Risk Score} = \text{Likelihood} \times \text{Impact}$ | Rendah (1–6), Sedang (8–12), Tinggi (15–25) |
| **k-Anonymity** | Setiap baris quasi-ID tidak dapat dibedakan dari minimal $k-1$ entri lain | Ukuran kelompok kesetaraan $\ge k$ |

