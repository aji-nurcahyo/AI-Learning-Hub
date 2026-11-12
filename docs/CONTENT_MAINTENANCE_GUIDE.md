# Content Maintenance & Platform Architecture Guide
**Artificial Intelligence Learning Hub — Institut Teknologi Sumatera**

Panduan ini dirancang bagi pengajar dan pengelola repositori agar pembaruan materi akademik (modul, ringkasan, slide, skrip Python, notebook, maupun jalur spesialisasi) dapat dilakukan **tanpa memodifikasi komponen antarmuka React (`src/`)**.

---

## 1. Content Architecture & Separation of Concerns

Platform memisahkan enam lapisan utama secara tegas:

| Lapisan | Lokasi Berkas | Peran Tunggal |
| :--- | :--- | :--- |
| **1. Academic Content (SSOT)** | `content/level-*/chapter-*/index.md`, `practice.md`, `code/`, `notebooks/` | Sumber tunggal teori, matematika, latihan, skrip runnable, dataset, dan rubrik. |
| **2. Presentation Metadata** | `content/level-*/chapter-*/metadata.json` | Identitas chapter, ringkasan pendek (`description`), 4–6 `highlights`, prasyarat, tag, status, dan referensi. |
| **3. Slide Content** | `content/level-*/chapter-*/slides/deck.json` | Array JSON terstruktur (dikompilasi oleh engine presentasi internal tanpa label framework publik). |
| **4. Track Configuration** | `content/specializations/tracks.json` + `content/tracks/<id>/` | Indeks dan detail jalur spesialisasi (roadmap, applied labs, studi kasus, capstone & rubrik 100 poin). |
| **5. UI Rendering & Routing** | `src/main.tsx`, `src/content.ts`, `src/routes.ts` | Komponen data-driven murni tanpa teks kurikulum yang di-hardcode. |
| **6. Automated QA & Build Guards** | `scripts/validate.mjs`, `check-formatting.mjs`, `tests.mjs`, `test-http.mjs` | Pengaman regresi skema, tautan, panjang ringkasan, format slide, build Vite, dan HTTP asset check. |

---

## 2. Directory & Naming Conventions

```text
content/
├── levels.json                           # 4 level kurikulum
├── getting-started/                      # GS-01 s/d GS-08
├── templates/slides.json                   # Templat preview pengajar
├── specializations/tracks.json           # Indeks 5 jalur spesialisasi
├── tracks/
│   ├── computer-vision/                  # Track A (metadata.json, overview.md, practice.md, capstone.md, labs/)
│   ├── nlp-llm/                          # Track B
│   ├── aiot-edge/                        # Track C
│   ├── telecommunications/               # Track D
│   └── robotics/                         # Track E
└── level-<N>-<slug>/chapter-<NN>/
    ├── metadata.json                     # Identitas & presentation summary
    ├── index.md                          # Modul lengkap (SSOT)
    ├── practice.md                       # Panduan laboratorium terpadu
    ├── audit.md                          # Catatan audit publikasi
    ├── slides/deck.json                  # Dek slide presentasi
    ├── code/                             # Skrip Python mandiri + code/README.md
    ├── notebooks/chapter_<NN>_companion.ipynb
    └── experiments/                      # Grafik & log hasil JSON lokal
```

---

## 3. Chapter Metadata Schema (`metadata.json`)

```json
{
  "id": 27,
  "slug": "chapter-27",
  "levelId": 3,
  "title": "Natural Language Processing Fundamentals",
  "status": "published",
  "description": "Ringkasan editorial 2–3 kalimat (maksimal ~65 kata / <= 480 karakter).",
  "highlights": [
    "Poin ringkas 1 (4–6 item total untuk tampilan landing page)",
    "Poin ringkas 2"
  ],
  "objectives": ["Menjelaskan ... (daftar tujuan lengkap di modul)"],
  "prerequisites": [11, 14, 21],
  "tags": ["Deep Learning & Generative AI", "NLP"],
  "resources": {
    "reading": { "status": "available", "path": "index.md" },
    "slides": { "status": "available", "path": "slides/deck.json" },
    "practice": { "status": "available", "path": "practice.md" }
  },
  "references": [
    { "title": "Penulis (Tahun), Judul", "url": "https://...", "status": "verified" }
  ],
  "validation": { "reviewed": true, "executionVerified": true }
}
```

**Aturan Penting Overview (`description` & `highlights`):**
- Pertahankan `description` ringkas (2–3 kalimat editorial, maksimal **65 kata**). Jangan menumpuk rincian implementasi atau hasil benchmark di landing page—simpan detail teknis di `index.md` atau `practice.md`.
- Gunakan `4–6` butir pendek di `highlights` untuk panel *"Yang akan dipelajari"*.

---

## 4. Workflow Pemeliharaan Konten Langkah-demi-Langkah

1. **Mengubah Judul atau Summary Chapter:** Sunting field `title` atau `description` pada `content/level-*/chapter-XX/metadata.json`.
2. **Memperbarui Learning Highlights:** Sunting array `highlights` (4–6 item ringkas) di `metadata.json`.
3. **Mengedit Modul Pembelajaran:** Sunting langsung `index.md` (tetap pertahankan bagian Latihan, Tugas/Mini Project, Rubrik 100 poin, dan Referensi).
4. **Menambahkan Referensi Akademik:** Tambahkan entri ber-URL `https://` dan `"status": "verified"` pada `references` di `metadata.json` serta di bagian akhir `index.md`.
5. **Memperbarui Slide Perkuliahan:** Sunting `slides/deck.json` (atau jalankan skrip pembangunnya jika tersedia). Setiap objek slide memiliki key `kind` (`cover`, `content`, `code`, `split`, `case`, `summary`), `title`, `body` ($\le 1000$ karakter), `notes` (wajib terisi), serta opsional `code` ($\le 18$ baris) atau `tex`. **Jangan** mencantumkan label framework presentasi pada judul/isi slide publik.
6. **Memperbarui Contoh Python & Notebook:** Ubah skrip di `code/`, jalankan ulang di CPU menggunakan Python virtual environment (`.venv`), lalu regenerasi atau jalankan ulang `notebooks/chapter_XX_companion.ipynb` (`nbformat v4`).
7. **Memperbarui Dataset Sintetis:** Simpan skrip pembangkit atau metadata berlisensi di `data/` dan perbarui dokumentasi `practice.md`.
8. **Memperbarui Specialization Tracks:**
   - Indeks jalur di `content/specializations/tracks.json`.
   - Metadata jalur di `content/tracks/<id>/metadata.json` (`roadmap: {foundation, core, advanced}`, `labs`, `capstone`, `references`).
   - Materi markdown di `overview.md`, `practice.md`, dan `capstone.md`.
9. **Menjalankan Validasi & Pengujian:**
   ```bash
   npm run check:content
   npm run typecheck
   npm run lint
   npm test
   npm run build
   npm run test:http
   ```
10. **Pratinjau Lokal & Commit:** Jalankan `npm run preview`, periksa halaman di browser pada mode Terang/Gelap dan ukuran Mobile, lalu lakukan `git commit` setelah peninjauan.

---

## 5. GitHub Pages Compatibility Notes

- **Base Path:** Konfigurasi Vite memakai `VITE_BASE_PATH` default `/ai-learning-hub/` (diatur otomatis pada build produksi). Gunakan selalu base relatif dari `import.meta.env.BASE_URL` saat menautkan file statis seperti `.ipynb` atau `docs/*.md`.
- **Hash Routing:** Seluruh rute (`#chapter/27/read`, `#track/robotics/capstone`, dll.) menggunakan fragment hash `#` sehingga aman di-refresh pada GitHub Pages tanpa membutuhkan server-side rewrite.
- **Static Artifacts:** `public/.nojekyll` dan `dist/404.html` menjaga agar file dengan awalan garis bawah atau aset statis tidak terblokir.

---

## 6. Common Mistakes, Troubleshooting & Known Limitations

- **Error `summary exceeds 65 words`:** Ringkas `description` di `metadata.json` agar tetap 2–3 kalimat efektif.
- **Error `measurable objectives required`:** Kalimat pertama tiap entri `objectives` wajib diawali kata kerja Bloom Bahasa Indonesia yang terdaftar pada `scripts/validate.mjs`.
- **Error `Slide ... Code exceeds 18 lines`:** Pecah blok kode panjang menjadi beberapa slide berurutan agar tidak meluber di layar proyektor.
- **Tautan Lab Track Patah:** Pastikan setiap path di `content/tracks/<id>/metadata.json` -> `labs[].scriptPaths` benar-benar ada di disk.
- **Keterbatasan Diketahui:** Notebook awal Chapter 03–05 bertipe `nbformat v4` valid namun belum menyimpan output pra-eksekusi di sel; eksperimen perangkat keras/medis/keamanan bersifat simulasi edukatif luring.

---

## 7. Quick Maintenance Checklist

- [ ] `description` chapter berkisar 25–65 kata dan tidak menduplikasi seluruh isi modul.
- [ ] `highlights` berisi 4–6 butir ringkas.
- [ ] Tidak ada penyebutan `Reveal.js` di label antarmuka publik.
- [ ] Semua tautan skrip track & `.ipynb` valid dan dapat diunduh.
- [ ] `npm run check:content`, `npm run typecheck`, `npm run lint`, `npm test`, `npm run build`, dan `npm run test:http` seluruhnya **PASS**.

