# Audit Publikasi: Chapter 01 — Introduction to Artificial Intelligence

Dokumen ini memuat bukti verifikasi akademik, teknis, dan integritas materi untuk publikasi resmi Chapter 01 pada Artificial Intelligence Learning Hub.

## 1. Identitas Chapter
- **Nomor Chapter**: 01
- **Judul Resmi**: Introduction to Artificial Intelligence
- **Tingkat (Level)**: Level 1 — AI Foundations
- **Pengembang**: Aji Pamungkas Tri Nurcahyo (Institut Teknologi Sumatera)
- **Status Publikasi**: `published`
- **Tanggal Audit**: 4 November 2026

## 2. Keselarasan Standar Kurikulum & Verifikasi Rujukan
Materi Chapter 01 telah dipetakan dan diverifikasi secara langsung terhadap rujukan akademik dan industri primer:
1. **ACM/IEEE-CS/AAAI CS2023 Curricula**:
   - Area: *Artificial Intelligence (AI) Knowledge Area*.
   - Pemetaan: Fondasi agen cerdas, pembedaan pendekatan simbolik vs statistik, anatomi inferensi, dan etika algoritma.
   - Status Rujukan: `verified` (URL: https://csed.acm.org/).
2. **UNESCO AI Competency Framework for Students (2024)**:
   - Pemetaan: Aspek literasi kritis AI, pemahaman keterbatasan model, bahaya antropomorfisme, dan prinsip *human agency*.
   - Status Rujukan: `verified` (URL: https://www.unesco.org/en/articles/ai-competency-framework-students).
3. **NIST AI Risk Management Framework (AI RMF 1.0)**:
   - Pemetaan: Empat fungsi inti (Govern, Map, Measure, Manage) serta pilar validitas, keandalan, keadilan (*fairness*), transparansi, dan keamanan sistem cerdas.
   - Status Rujukan: `verified` (URL: https://www.nist.gov/itl/ai-risk-management-framework).
4. **Google Machine Learning Crash Course**:
   - Pemetaan: Pergeseran paradigma dari *programming logic* ke *machine learning*, terminologi training vs inference, serta analisis kesalahan kuadran confusion matrix.
   - Status Rujukan: `verified` (URL: https://developers.google.com/machine-learning/crash-course/).

## 3. Bukti Verifikasi Eksekusi Kode (Python Examples)
Seluruh kode dieksekusi secara nyata pada lingkungan komputasi:
- **Lingkungan**: Python 3.12.10 (Windows 64-bit AMD64)
- **Pustaka**: Standard Library murni (bebas ketergantungan paket pihak ketiga yang rapuh)
- **Hasil Eksekusi 5 Contoh**:
  1. `01_rule_based_recommender.py`: Berhasil menghasilkan rekomendasi deterministik untuk 3 profil mahasiswa; mencatat keterbatasan kekakuan logika terhadap input asing.
  2. `02_keyword_text_classifier.py`: Berhasil mengklasifikasikan pesan dan membuktikan terjadinya False Positive pada pesan akademik yang memuat kata 'gratis', serta False Negative pada pesan penipuan dengan kata 'dana'.
  3. `03_exact_matching_vs_variation.py`: Berhasil membuktikan kerapuhan exact string match (gagal pada 4 dari 5 variasi input alami manusia).
  4. `04_explicit_accuracy_calc.py`: Berhasil menghitung akurasi 80.0% secara manual dari daftar 10 prediksi ilustratif.
  5. `05_simple_confusion_analysis.py`: Berhasil menghitung kuadran TP=2, FP=2, TN=14, FN=2; membuktikan bahwa akurasi 80% dapat menyembunyikan kenyataan bahwa 50% fraud nyata lolos (FN=2 dari 4).
- Seluruh output aktual telah dicatat dan didokumentasikan di `content/level-1-ai-foundations/chapter-01/code/README.md`.

## 4. Verifikasi Interactive Notebook
- **File**: `content/level-1-ai-foundations/chapter-01/notebooks/chapter_01_companion.ipynb`
- **Format**: nbformat 4 (nbformat_minor: 5)
- **Kernelspec**: Python 3
- **Eksekusi Bersih**: Seluruh sel kode telah dieksekusi secara berurutan (*sequential execution*) dari kernel bersih, dengan nomor urut `execution_count` dan output stream terekam lengkap tanpa *hidden state*.

## 5. Verifikasi Presentasi Reveal.js
- **File**: `content/level-1-ai-foundations/chapter-01/slides/deck.json`
- **Jumlah Slide**: 20 slide terstruktur
- **Integritas Schema**:
  - Seluruh slide memiliki `kind` valid (`cover`, `content`, atau `code`).
  - Seluruh slide memiliki speaker notes mendalam untuk memandu dosen/pengajar.
  - Panjang teks `body` tidak melebihi 1000 karakter per slide (mencegah overflow visual).
  - Seluruh blok `code` ringkas (< 18 baris) untuk kenyamanan baca di proyektor.

## 6. Verifikasi Latihan & Rubrik Penilaian
- **Latihan Terstruktur (`exercises/exercises.md`)**:
  - Memuat 6 soal berjenjang: Bagian A (Dasar: Identifikasi konsep), Bagian B (Menengah: Anatomi dan hitungan metrik), Bagian C (Lanjut: Analisis kelayakan dan Responsible AI).
  - Dilengkapi kunci jawaban dan pembahasan analitis pada bagian terpisah.
- **Tugas Mini Project (`assignments/mini_project.md`)**:
  - Judul: "Analisis Kelayakan Solusi AI untuk Masalah Nyata".
  - Memuat 9 deliverables lengkap tanpa mewajibkan training model pada Chapter 01.
  - Dilengkapi rubrik terperinci total 100 poin berbobot dengan indikator: Nilai Penuh (*Full Marks*), Parsial (*Partial Marks*), dan Tidak Terpenuhi (*Unsatisfactory*).

## 7. Rekam Pemeriksaan Publication Guard
- `validateChapter()`: PASS (Tidak ada error path, slug, level, objectives, atau resources).
- `validateAll()`: PASS (40 chapter terdaftar, Level 1 Chapter 01 published, Chapter 02-40 tetap berstatus coming-soon).
- Unit Tests: 11 tests PASS.
- Typecheck: 0 error.
- Linting: 0 warning, 0 error.

## 8. Kesimpulan Audit
Chapter 01 memenuhi seluruh persyaratan editorial, pedagogis, komputasional, dan regulasi publication guard platform AI Learning Hub. Materi secara resmi siap diakses oleh mahasiswa dan publik.

