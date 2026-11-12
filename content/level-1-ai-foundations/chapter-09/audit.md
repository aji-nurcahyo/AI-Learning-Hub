# Audit Publikasi Konten — Chapter 09: AI Ethics, Bias & Responsible AI

Dokumen ini mencatat audit editorial, validasi struktural, integrasi platform, dan verifikasi eksekusi kode Python untuk **Chapter 09: AI Ethics, Bias & Responsible AI**.

---

## 1. Identitas & Metadata Chapter

- **ID Chapter**: 9
- **Slug**: `chapter-09`
- **Tingkat Kurikulum**: Level 1 — AI Foundations (Penutup Level 1)
- **Judul Resmi**: `AI Ethics, Bias & Responsible AI`
- **Status Publikasi**: `published`
- **Prerequisites**: `[8]` (Data Preparation & Exploratory Analysis)
- **Tinjauan Kualitas**: `reviewed: true`, `executionVerified: true`
- **Target Pembelajaran**: 10 measurable objectives berbasis Taksonomi Bloom.
- **Daftar Pustaka Primer**: 5 referensi terverifikasi (Barocas et al./MIT Press, Kleinberg et al./ITCS Impossibility Theorem, Latanya Sweeney/CMU Re-identification, NIST AI RMF 1.0 SP 1270, UNESCO Recommendation on Ethics of AI).

---

## 2. Kelengkapan Deliverables Konten

| Komponen | Jalur Berkas | Keterangan & Verifikasi |
| :--- | :--- | :--- |
| **Modul Utama** | `content/level-1-ai-foundations/chapter-09/index.md` | Single source of truth (Part A s/d M, diagram 4 pilar Responsible AI & siklus bias, 5 latihan kritis, mini project audit beasiswa, rubrik 100 poin, 5 referensi primer). |
| **Slide Presentasi** | `content/level-1-ai-foundations/chapter-09/slides/deck.json` | 27 slide Reveal.js terstruktur, rata kiri, proporsional, tanpa teks mentah, dilengkapi speaker notes mendalam untuk pengajar. |
| **Panduan Praktikum** | `content/level-1-ai-foundations/chapter-09/practice.md` | Panduan laboratorium langkah demi langkah, komputasi metrik formal keadilan, evaluasi privasi UU PDP, dan tabel ringkasan metrik. |
| **Companion Notebook** | `content/level-1-ai-foundations/chapter-09/notebooks/chapter_09_companion.ipynb` | Notebook interaktif Jupyter format v4 (12 sel), tervalidasi dan tereksekusi tanpa error. |
| **Katalog Kode Sumber** | `content/level-1-ai-foundations/chapter-09/code/README.md` | Dokumentasi 5 skrip mandiri Python, dataset CSV sintetis beasiswa & registri risiko, serta tabel hasil eksekusi aktual. |
| **Dataset Beasiswa** | `content/level-1-ai-foundations/chapter-09/code/data/scholarship_historical.csv` | 1.000 baris data pelamar beasiswa dua kelompok demografis (skor tes bakat, IPK, potensi murni, keputusan historis). |
| **Registri Risiko Etis** | `content/level-1-ai-foundations/chapter-09/code/data/ai_ethics_risk_register.csv` | Registri penilaian risiko etis 5 skenario implementasi AI industri (Likelihood x Impact). |

---

## 3. Hasil Verifikasi Eksekusi Skrip Python (3.12.10)

| File Skrip | Pustaka Kunci | Hasil Eksekusi Aktual | Exit Code |
| :--- | :--- | :--- | :---: |
| `historical_bias_demo.py` | Pandas, NumPy | Skor tes objektif setara (73.51 vs 71.70), namun kelulusan historis timpang (45.93% vs 7.99%), rasio disparate impact 0.1739 | 0 |
| `group_fairness_metrics.py` | Scikit-learn, Pandas | Model baseline akurasi 79.67%, selection rate gap 0.2928, TPR A 78.26% vs B 29.55%, Delta EO 0.4872 | 0 |
| `fairness_tradeoff_demo.py` | Scikit-learn, NumPy | Post-processing threshold tuning: menurunkan tau_B dari 0.50 ke 0.10 menaikkan TPR B ke 40.91% dan menekan Delta EO ke 0.3735 dengan akurasi 81.00% | 0 |
| `privacy_masking_demo.py` | Pandas, Hashlib | Berhasil mere-identifikasi rekam medis rahasia target melalui tautan quasi-ID (Usia, Kelamin, Kode Pos) pada data DPT publik; dasar UU PDP Pasal 1 & 35 | 0 |
| `risk_assessment_demo.py` | Pandas | Matriks risiko Likelihood x Impact mengevaluasi 5 skenario: 3 Kritis (Seleksi Beasiswa [20], Radiologi [15], Facial Recognition [16]), 2 Sedang | 0 |

