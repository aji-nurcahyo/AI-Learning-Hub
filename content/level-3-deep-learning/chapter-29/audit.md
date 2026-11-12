# Laporan Audit Kualitas: Chapter 29 — Large Language Models & Generative AI

**Tanggal Audit**: 8 November 2026  
**Status**: **PASSED (100% MEMENUHI STANDAR KURIKULUM RESMI)**  
**Auditor**: Tim Penjamin Mutu Kurikulum AI Learning Hub  
**Cakupan**: Bab 29 — Large Language Models & Generative AI (Level 3 Deep Learning)

---

## 1. Verifikasi Keselarasan Master Curriculum

| Butir Evaluasi | Standar Kurikulum Resmi | Status Aktual Repositori | Keterangan |
|---|---|:---:|---|
| Nomor Bab | Chapter 29 | 29 | Sesuai |
| Judul Resmi | Large Language Models & Generative AI | Large Language Models & Generative AI | Sesuai |
| Level Pembelajaran | Level 3: Deep Learning & Generative AI | Level 3: Deep Learning & Generative AI | Sesuai |
| Prasyarat | Chapter 28 | `[28]` | Sesuai |
| Status Publikasi | `published` | `published` | Sesuai |
| Batasan Komputasi | Ekosistem CPU terverifikasi | Scikit-learn 1.9.1, PyTorch 2.15.0+cpu, NumPy 2.5.4 | Memenuhi standar |

---

## 2. Verifikasi Artefak Pembelajaran

| No | Artefak | Lokasi Berkas | Status Verifikasi |
|---|---|---|:---:|
| 1 | Modul Teori (SSOT) | `chapter-29/index.md` | Lengkap (Part A–N, 15+ Diagram ASCII, Latihan 3-Level, Rubrik 100 Poin, 5 Referensi) |
| 2 | Slide Reveal.js | `chapter-29/slides/deck.json` | 33 Slide valid, format direct array JSON, KaTeX, speaker notes |
| 3 | Panduan Praktik Lab | `chapter-29/practice.md` | Lengkap (14 Tahapan Praktik, 3-Tier Exercises, Rubrik 100 Poin) |
| 4 | Skrip Python Lengkap | `chapter-29/code/*.py` (14 skrip) | 14/14 skrip dieksekusi dengan exit code 0 |
| 5 | Notebook Pendamping | `chapter-29/notebooks/chapter_29_companion.ipynb` | Valid Jupyter Notebook v4 (Pre-executed) |
| 6 | Katalog Kode | `chapter-29/code/README.md` | Panduan eksekusi dan tabel 14 skrip terverifikasi |
| 7 | Log Eksperimen | `chapter-29/experiments/results_ch29.json` | Log JSON profil eksekusi CPU seluruh 14 skrip |
| 8 | Metadata Bab | `chapter-29/metadata.json` | 14 Bloom Objectives diawali kata kerja valid, 5 Referensi terverifikasi |

---

## 3. Verifikasi Numerik & Hasil Komputasi Aktual

Seluruh 14 skrip Python berikut telah dieksekusi dan diverifikasi secara langsung menggunakan Python 3.12.10, PyTorch 2.15.0+cpu, Scikit-learn 1.9.1, dan NumPy 2.5.4:

```
[Pengujian 1: Tokenisasi & Subword BPE Concept]
- Pemecahan Kata Berafiks   : 'bantuan' -> 'bantu' + '##an'
- Penanganan Token Khusus   : <pad>=0, <unk>=1, <bos>=2, <eos>=3
- Pembuatan Attention Mask  : 1 untuk token nyata, 0 untuk token bantalan

[Pengujian 2: Next-Token Prediction & Softmax]
- Vektor Logits Mentah      : Diproyeksikan ke seluruh ruang kosakata
- Normalisasi Softmax       : Jumlah probabilitas tepat 1.000000
- Prediksi Teratas Kontekstual: 'cepat' (P = 51.98%), 'pesat' (P = 38.42%)

[Pengujian 3: Training Objective & Perplexity]
- Model Bagus Terlatih      : Loss = 0.3541 nat | Perplexity = 1.4249
- Model Acak (Untrained)    : Loss = 2.1972 nat | Perplexity = 9.0000
- Worked Example NLL        : -ln(P) dihitung analitis pada tiap langkah sekuens

[Pengujian 4: Autoregressive Loop]
- Rekursi Konteks           : Tiap token baru disisipkan kembali ke jendela masukan
- Terminasi                 : Berhenti otomatis saat menemukan token <eos>

[Pengujian 5: Greedy Decoding]
- Sifat Deterministik       : 3 kali eksekusi menghasilkan 100% token identik
- Fenomena Repetisi         : Terbukti terjebak dalam pengulangan frasa "sangat hebat dan"

[Pengujian 6: Temperature Sampling]
- Suhu Dingin (T = 0.2)     : Probabilitas puncak 96.75% (Entropi Shannon = 0.1708)
- Suhu Netral (T = 1.0)     : Probabilitas berimbang (Entropi Shannon = 1.2005)
- Suhu Panas  (T = 1.8)     : Probabilitas mendekati seragam (Entropi Shannon = 1.5204)

[Pengujian 7: Top-k Filtering]
- k = 1                     : Identik dengan Greedy (P = 1.0000 pada kata teratas)
- k = 3                     : Memotong ekor distribusi; hanya 3 token teratas yang bersaing

[Pengujian 8: Nucleus Top-p Sampling]
- Kasus Model Yakin         : Pool menyusut dinamis menjadi 2 kata (p = 0.85)
- Kasus Model Ragu          : Pool melebar dinamis menjadi 7 kata (p = 0.85)

[Pengujian 9: LoRA Parameter Reduction]
- Bobot Model Dasar (W_0)   : 65.536 bobot (100% dibekukan, requires_grad=False)
- Adaptor LoRA (A + B, r=4) : 2.048 bobot (Terlatih)
- Persentase Penghematan    : 96.88% parameter dihemat!
- Verifikasi Gradien        : W_0 bebas gradien (False), Adaptor menerima gradien (True)

[Pengujian 10: In-Context Learning Prompting]
- Pola Zero-Shot & Few-Shot : Demonstrasi 3 contoh mengunci gaya luaran tanpa fine-tuning
- Format JSON Terstruktur   : Validasi parsing tiket komplain berhasil

[Pengujian 11: RAG Grounding]
- Temu-Kembali TF-IDF       : Dokumen DOC-001 (Skor 0.6124) terpilih otomatis
- Pembangkitan Faktual      : Jawaban memuat tanggal 6 Oktober 2014 & Perpres No. 124/2014

[Pengujian 12: Audit Halusinasi]
- Respon Terkalibrasi       : Groundedness 100%, Hallucination Rate 0%
- Respon Berhalusinasi      : Groundedness 50%, Hallucination Rate 50% (Klaim palsu terdeteksi)

[Pengujian 13: Guardrails Architecture]
- Input Guardrail           : 2 upaya Prompt Injection berhasil dideteksi dan diblokir
- Output Guardrail          : NIK KTP, Email, dan No HP berhasil disensor otomatis

[Pengujian 14: Tiny Transformer Generation]
- Parameter Model TinyGPT   : 11.648 bobot (d_model=32, n_heads=2, n_layers=2)
- Pelatihan Cepat CPU       : 40 epoch selesai dalam 0.312 detik (Loss turun 3.14 -> 0.08)
- Generasi Autoregresif     : Berhasil membangkitkan teks logis pada CPU
```

---

## 4. Kesimpulan Audit

Chapter 29 telah selesai diimplementasikan secara komprehensif, mencakup modul akademik single-source-of-truth, slide kuliah Reveal.js, panduan laboratorium, notebook pendamping, 14 skrip Python terverifikasi, dan log eksperimen JSON. Seluruh artefak telah memenuhi 100% standar kurikulum resmi AI Learning Hub.

