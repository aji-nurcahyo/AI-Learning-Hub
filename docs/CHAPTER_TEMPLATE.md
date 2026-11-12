# Template Modul Chapter (`index.md`)

Salin kerangka ini ke `index.md` chapter. Modul merupakan **single source of truth** untuk materi teori, latihan, pembahasan, tugas mini project, rubrik penilaian, dan referensi akademik.

---

## 1. Ringkasan Chapter
[Deskripsi singkat ruang lingkup chapter dan relevansinya dalam kurikulum AI.]

## 2. Tujuan Pembelajaran
- [Tujuan terukur 1, selaras dengan metadata.json]
- [Tujuan terukur 2]
- [Tujuan terukur 3]

## 3. Prasyarat
- [Prasyarat pengetahuan atau lingkungan sebelum memulai modul ini.]

## 4. Penjelasan Konsep Inti
[Penjelasan mendalam dan terstruktur dengan subbab ### jika diperlukan. Pertahankan istilah teknis berbahasa Inggris.]

## 5. Dasar Matematika / Formulasi Formal
[Notasi matematika dengan KaTeX $$ ... $$ jika relevan, definisi formal parameter dan fungsi objektif.]

## 6. Algoritma dan Metodologi
[Langkah komputasi, pseudocode, atau algoritma yang diimplementasikan.]

## 7. Diagram Arsitektur atau Workflow
[Visualisasi alur data, hierarki arsitektur, atau alur pengambilan keputusan.]

## 8. Implementasi Bertahap
[Penjelasan rancangan kode dan fungsi utama yang akan dibangun.]

## 9. Contoh Python Runnable
[Cuplikan kode Python mandiri yang dapat langsung dijalankan oleh mahasiswa.]

## 10. Output Eksekusi Terverifikasi
[Hasil terminal atau evaluasi riil dari eksekusi kode di atas.]

## 11. Dataset dan Lisensi
[Karakteristik data yang digunakan, jumlah sampel/fitur, sumber rujukan, dan lisensi penggunaan.]

## 12. Kesalahan Umum dan Keterbatasan
[Pitfalls yang sering dihadapi pemula, asumsi yang tidak realistis, dan batasan pendekatan.]

## 13. Ringkasan Modul
[Takeaway utama yang harus diingat mahasiswa setelah membaca modul.]

## 14. Latihan Terstruktur
### Latihan 1: Pemahaman Konseptual
[Pertanyaan reflektif atau analisis kritis.]

### Latihan 2: Implementasi Dasar
[Modifikasi atau pembuatan script terpandu.]

### Latihan 3: Analisis & Evaluasi
[Perbandingan performa, trade-off, atau skenario terapan.]

## 15. Pembahasan dan Panduan Jawaban Latihan
[Kunci jawaban, pedoman reasoning, dan alternatif solusi untuk latihan di atas.]

## 16. Tugas Mini Project / Proyek Terapan
### Deskripsi Tugas
[Skenario problem statement riil.]

### Ruang Lingkup dan Deliverables
1. Deliverable 1
2. Deliverable 2
...

### Petunjuk Pengumpulan
[Format submission, struktur folder, dan tenggat waktu.]

## 17. Rubrik Penilaian Proyek (Total 100 Poin)

| Kriteria Penilaian | Bobot | Indikator Kinerja Unggul | Bukti yang Dinilai |
|---|---:|---|---|
| Perumusan Masalah & Metodologi | 20 Poin | Formulasi masalah jelas, batasan dipahami | Dokumen perumusan masalah |
| Implementasi Kode & Pipeline | 25 Poin | Kode modular, runnable, bebas error | Script Python / Notebook |
| Eksperimen & Evaluasi | 25 Poin | Metrik dihitung tepat, analisis komprehensif | Tabel metrik & grafik |
| Analisis Keterbatasan & Responsible AI | 15 Poin | Analisis bias, transparansi, batasan sistem | Laporan refleksi kritis |
| Kerapian & Dokumentasi | 15 Poin | README jelas, reproduksibilitas terjamin | Dokumentasi & struktur file |

## 18. Referensi dan Sumber Belajar Primer
1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach (4th ed.)*. Pearson.
2. [Rujukan buku teks, paper seminal, atau dokumentasi resmi lainnya]

---

# Template Panduan Praktik (`practice.md`)

Gunakan kerangka ini pada `content/<level>/chapter-NN/practice.md` untuk pilar Praktik (jika chapter memerlukan aktivitas hands-on).

```markdown
# Panduan Praktik: [Judul Chapter]

Dokumen ini memandu pelaksanaan sesi laboratorium mandiri dan implementasi kode.

## 1. Tujuan Praktik
- [Tujuan praktis 1]
- [Tujuan praktis 2]

## 2. Persyaratan Lingkungan
- Python >= 3.12
- Dependencies terpasang (lihat virtual environment setup di Getting Started GS-04)
- CPU standar (GPU tidak diwajibkan untuk materi dasar)

## 3. Langkah Menjalankan Kode
1. Aktifkan virtual environment.
2. Masuk ke direktori `code/`.
3. Jalankan script Python:
   \`\`\`bash
   python 01_nama_script.py
   \`\`\`

## 4. Daftar Skrip Praktik
- `code/01_*.py`: [Penjelasan singkat fungsi script]
- `code/02_*.py`: [Penjelasan singkat fungsi script]

## 5. Companion Notebook
- Tersedia file `notebooks/chapter_NN_companion.ipynb` yang dapat dijalankan di VS Code atau Google Colab.
- Semua cell telah diverifikasi dieksekusi secara berurutan (*Restart & Run All*).

## 6. Dataset dan Catatan Lisensi
- Data sintetis atau benchmark yang digunakan disertai atribusi lisensi permissif (MIT / CC-BY-4.0 / Public Domain).
```
