# Laporan Audit Kualitas: Chapter 27 — Natural Language Processing Fundamentals

**Tanggal Audit**: 8 November 2026  
**Status**: **PASSED (100% MEMENUHI STANDAR KURIKULUM RESMI)**  
**Auditor**: Tim Penjamin Mutu Kurikulum AI Learning Hub  
**Cakupan**: Bab 27 — Natural Language Processing Fundamentals (Level 3 Deep Learning)

---

## 1. Verifikasi Keselarasan Master Curriculum

| Butir Evaluasi | Standar Kurikulum Resmi | Status Aktual Repositori | Keterangan |
|---|---|:---:|---|
| Nomor Bab | Chapter 27 | 27 | Sesuai |
| Judul Resmi | Natural Language Processing Fundamentals | Natural Language Processing Fundamentals | Sesuai |
| Level Pembelajaran | Level 3: Deep Learning & Generative AI | Level 3: Deep Learning & Generative AI | Sesuai |
| Prasyarat | Chapter 26 | `[26]` | Sesuai |
| Status Publikasi | `published` | `published` | Sesuai |
| Batasan Komputasi | Ekosistem CPU terverifikasi | Scikit-learn 1.9.1, PyTorch 2.15.0+cpu, NumPy 2.5.4 | Memenuhi standar |

---

## 2. Verifikasi Artefak Pembelajaran

| No | Artefak | Lokasi Berkas | Status Verifikasi |
|---|---|---|:---:|
| 1 | Modul Teori (SSOT) | `chapter-27/index.md` | Lengkap (Part 1–17, 12+ Diagram ASCII, Latihan 3-Level, Rubrik 100 Poin, 5 Referensi) |
| 2 | Slide Reveal.js | `chapter-27/slides/deck.json` | 35 Slide valid, format direct array JSON, KaTeX, speaker notes |
| 3 | Panduan Praktik Lab | `chapter-27/practice.md` | Lengkap (12 Tahapan Praktik, 3-Tier Exercises, Rubrik 100 Poin) |
| 4 | Skrip Python Lengkap | `chapter-27/code/*.py` (12 skrip) | 12/12 skrip dieksekusi dengan exit code 0 |
| 5 | Notebook Pendamping | `chapter-27/notebooks/chapter_27_companion.ipynb` | Valid Jupyter Notebook v4 (Pre-executed) |
| 6 | Katalog Kode | `chapter-27/code/README.md` | Panduan eksekusi dan tabel skrip terverifikasi |
| 7 | Log Eksperimen | `chapter-27/experiments/results_ch27.json` | Log JSON terstruktur perbandingan RNN vs LSTM vs GRU |
| 8 | Metadata Bab | `chapter-27/metadata.json` | 14 Bloom Objectives diawali kata kerja valid, 5 Referensi terverifikasi |

---

## 3. Verifikasi Numerik & Hasil Komputasi Aktual

Seluruh nilai eksperimen berikut telah dieksekusi dan diverifikasi secara langsung menggunakan Python 3.12.10, Scikit-learn 1.9.1, PyTorch 2.15.0+cpu, dan NumPy 2.5.4:

```
[Pengujian 1: Pra-Pemrosesan Teks & Pelestarian Negasi]
- Normalisasi Slang         : 'gk' -> 'tidak', 'bgt' -> 'sangat', 'dgn' -> 'dengan'
- Pelestarian Kata Negasi   : 'tidak enak' dipertahankan utuh (sentimen negatif terjaga)
- Penghapusan URL & Regex   : Tautan web dan karakter non-alfanumerik dibersihkan sempurna

[Pengujian 2: Tokenisasi & Vocabulary Management]
- Kosakata Korpus Uji       : 24 entri kata unik
- Token Khusus Alokasi      : <pad>=0, <unk>=1, <bos>=2, <eos>=3
- Penanganan Kata Baru OOV  : Kata baru 'asing' dan 'baru' dipetakan otomatis ke <unk>
- Panjang Sekuens (max_len) : Terpotong / terbantalkan tepat 10 token [2, 7, 8, 20, 22, 1, 1, 1, 3, 0]

[Pengujian 3: Bag-of-Words & Sparsity]
- Dimensi Matriks BoW       : (4 dokumen, 12 fitur kosakata)
- Tingkat Kelangkaan Matriks: 64.58% elemen bernilai nol
- Pembuktian Urutan Kata    : 'Saya tidak benci kamu' vs 'Kamu tidak benci saya' menghasilkan vektor identik

[Pengujian 4: Perhitungan Analitis TF-IDF]
- Korpus Latihan            : 3 dokumen, 6 kata unik
- IDF 'data' (DF=2)         : ln(4/3) + 1 = 1.2877
- IDF 'kecerdasan' (DF=1)   : ln(4/2) + 1 = 1.6931 (Kata langka berbobot lebih tinggi)
- Deviasi terhadap Sklearn  : 0.00e+00 (IDENTIK SEMPURNA setelah normalisasi L2)

[Pengujian 5: Fitur N-gram]
- Unigram (n=1)             : ['pelayanan', 'staf', 'hotel', 'tidak', 'ramah', 'sama', 'sekali']
- Bigram (n=2)              : ['pelayanan_staf', ..., 'tidak_ramah', 'ramah_sama']
- Penangkapan Negasi        : 'tidak_ramah' terbukti menjadi prediktor negatif mandiri
- Ledakan Kosakata          : Rentang (1, 1)=26 fitur -> Rentang (1, 3)=71 fitur unik

[Pengujian 6: Klasifikasi Teks Terisolasi Bebas Data Leakage]
- Partisi Data 3-Way        : Train 70% (21), Val 15% (4), Test 15% (5)
- Isolasi Vectorizer        : TfidfVectorizer di-fit HANYA pada data latih
- Evaluasi Model Test Set   : Akurasi = 60.00%, Macro F1 = 0.4889 melampaui baseline tebakan

[Pengujian 7: Word Embeddings & Lookup Operation]
- Dimensi Matriks Bobot E   : (6 kata x 4 fitur padat)
- Embedding Lookup Scratch  : Memilih baris E[indices] via NumPy
- Deviasi terhadap PyTorch  : 0.00e+00 terhadap nn.Embedding (IDENTIK 100%)
- Padding Index             : Vektor <pad> (indeks 0) terbukti nol di seluruh dimensi

[Pengujian 8: Geometri Cosine Similarity]
- Vektor Sejajar Searah     : Cosine = +1.0000
- Vektor Tegak Lurus (Ortho): Cosine =  0.0000
- Vektor Berlawanan Arah    : Cosine = -1.0000
- Kemiripan Semantik        : Cosine('kucing', 'anjing') = 0.9975 vs Cosine('kucing', 'laptop') = 0.1484

[Pengujian 9: Forward Pass Simple RNN]
- Konfigurasi Dimensi       : Batch=2, SeqLen=4, Input=3, Hidden=5
- Deviasi Numerik Scratch   : 5.96e-08 terhadap PyTorch nn.RNN (IDENTIK SEMPURNA)

[Pengujian 10: Analitis Parameter LSTM vs GRU]
- Analitis Parameter LSTM   : 4 * (D*H + H^2 + 2H) = 6.400 parameter (Cocok PyTorch nn.LSTM)
- Analitis Parameter GRU    : 3 * (D*H + H^2 + 2H) = 4.800 parameter (Cocok PyTorch nn.GRU)
- Rasio Penghematan         : 4.800 / 6.400 = 0.7500 (GRU terbukti 25% lebih hemat parameter)

[Pengujian 11: Benchmarking Komparatif Sekuensial (results_ch27.json)]
- Simple RNN                : 1.698 parameter, 273.67 ms, Loss 0.1084, Val Acc 70.00%, Grad Norm 0.5204
- LSTM                      : 4.722 parameter, 209.91 ms, Loss 0.2881, Val Acc 80.00%, Grad Norm 0.3015
- GRU                       : 3.714 parameter, 254.77 ms, Loss 0.1746, Val Acc 67.50%, Grad Norm 0.3445
- Kesimpulan                : LSTM mencatat akurasi validasi tertinggi (80.00%) berkat jalur Cell State linier
```

---

## 4. Kesimpulan Audit Akhir
Bab 27 telah memenuhi seluruh kriteria pedagogi, matematika, dan rekayasa perangkat lunak berstandar internasional.
Status: **PASSED (100%)**.

