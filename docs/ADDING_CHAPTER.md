# Menambahkan dan Mengembangkan Chapter (Arsitektur 3 Pilar)

Metadata di `content/<level>/chapter-NN/metadata.json` adalah satu-satunya sumber judul, ID, slug, deskripsi, navigasi, tujuan pembelajaran, dan ketersediaan resource. Jangan membuat daftar judul kedua. Empat level didefinisikan pada `content/levels.json`; 5 specialization tracks pada `content/specializations/tracks.json`.

## 1. Tiga Pilar Pembelajaran Chapter

Halaman chapter disajikan dengan arsitektur 3 pilihan aksi utama (hanya muncul bila resource berstatus `available`):

1. **Modul (`reading`)**
   - File: `content/<level>/chapter-NN/index.md`
   - Single source of truth untuk seluruh penjelasan konsep, perumusan matematika, diagram arsitektur/workflow, latihan terstruktur dan pembahasannya, tugas mini project/proyek terapan, rubrik penilaian 100 poin, serta daftar referensi primer terverifikasi.
   - Sediakan tombol Cetak Modul / Simpan PDF di reading toolbar browser.

2. **Slide (`slides`)**
   - File: `content/<level>/chapter-NN/slides/deck.json`
   - Presentasi Reveal.js sebagai ringkasan visual untuk pengajaran di kelas (15–25 slide).
   - Format presentasi utama; cetak PDF tersedia via rute `?print-pdf` di browser. Tidak memerlukan pembuatan file PowerPoint (.pptx) atau PDF manual terpisah.

3. **Praktik (`practice`)**
   - File: `content/<level>/chapter-NN/practice.md`
   - Panel atau panduan terpadu aktivitas hands-on: tujuan praktik, spesifikasi environment, petunjuk eksekusi langkah-demi-langkah, katalog script Python runnable (`code/`), companion notebook (`notebooks/*.ipynb`), dataset sintetis/benchmark (`datasets/`), serta verifikasi output riil.
   - Opsional: Untuk chapter konseptual murni, resource `practice` dapat berstatus `unavailable` sehingga hanya kartu Modul dan Slide yang ditampilkan.

## 2. Struktur Metadata Resource (`metadata.json`)

Gunakan skema 3 key `resources`:

```json
{
  "id": 1,
  "level": "level-1-ai-foundations",
  "track": "core",
  "slug": "introduction-to-ai",
  "title": "Introduction to Artificial Intelligence",
  "subtitle": "Definisi, sejarah, paradigma, agen cerdas, dan lanskap kecerdasan buatan modern",
  "status": "published",
  "summary": "...",
  "prerequisites": ["..."],
  "learningObjectives": ["..."],
  "resources": {
    "reading": { "status": "available", "path": "index.md" },
    "slides": { "status": "available", "path": "slides/deck.json" },
    "practice": { "status": "available", "path": "practice.md" }
  },
  "references": [
    { "title": "...", "authors": "...", "year": 2020, "status": "verified" }
  ],
  "validation": {
    "reviewed": true,
    "executionVerified": true
  }
}
```

*Catatan:*
- Untuk chapter berstatus `coming-soon`, biarkan ketiga resource berstatus `"unavailable"` dengan `"path": null`. UI akan otomatis menampilkan notice tunggal yang informatif tanpa deretan kartu kosong.

## 3. Penulisan Slide Reveal.js (`deck.json`)

Slide menggunakan skema terstruktur dengan tata letak reusable:
- `kind: "cover"`: Slide judul (cover) — teks rata tengah.
- `kind: "content"`: Slide materi umum — teks rata kiri, semantic bullet points (`bullets: [...]`), paragraf pembuka (`lead`), atau diagram langkah (`diagram: [...]`).
- `kind: "split"`: Slide dua kolom/perbandingan — menyediakan `columns: [{ heading, points: [...] }]`.
- `kind: "case"`: Slide studi kasus / diskusi — menyediakan `caseBox: { title, scenario, options: [...] }` dan pemantik diskusi.
- `kind: "summary"`: Slide ringkasan / takeaway materi.
- `kind: "code"`: Slide cuplikan kode terformat.

Prinsip Slide:
- Jangan memasukkan paragraf terlalu panjang ke satu slide; pecah materi padat menjadi beberapa slide bertahap.
- Perataan teks kiri secara default untuk slide pembelajaran agar nyaman diproyeksikan.
- Sertakan `notes` (speaker notes) edukatif pada setiap slide untuk pemateri/dosen.
- Hindari asterisk markdown mentah; gunakan formatting semantik yang didukung renderer.

## 4. Langkah Pengembangan & Publikasi

1. **Scaffold**: Buat folder chapter di bawah level terkait. Biarkan status `"coming-soon"`.
2. **Modul (`index.md`)**: Susun materi lengkap dari konsep, diagram, latihan berjenjang, tugas, rubrik 100 poin, hingga referensi.
3. **Slide (`deck.json`)**: Rancang 15–25 slide visual proporsional dengan speaker notes.
4. **Praktik (`practice.md`, `code/`, `notebooks/`)**: Sediakan script mandiri tanpa error, sertakan companion notebook yang dieksekusi bersih.
5. **Verifikasi Lingkungan**: Uji eksekusi kode pada Python 3.12+ dan catat output aktual.
6. **Publikasi**: Setelah materi direview dan eksekusi terverifikasi, ubah `status` menjadi `"published"`, serta set `"reviewed": true` dan `"executionVerified": true`.
7. **Jalankan Uji Otomatis**:
   ```bash
   npm run check:content
   npm test
   npm run typecheck
   npm run lint
   npm run build
   npm run test:http
   ```
