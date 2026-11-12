# Artificial Intelligence Learning Hub

**From AI Fundamentals to Intelligent Systems**

Platform terbuka berbahasa Indonesia untuk Artificial Intelligence, Machine Learning, Deep Learning, Generative AI, dan pengembangan intelligent systems.

Creator & Developer: **Aji Pamungkas Tri Nurcahyo** · **Institut Teknologi Sumatera**.

Kurikulum lengkap: empat level pembelajaran (40 chapter akademik terbit), lima specialization tracks, Getting Started guides, slide presentasi interaktif, serta latihan komputasi dan notebook.

## Menjalankan

Node.js 24 LTS dan npm (teruji Node 24.15.0).

```sh
npm ci
npm run dev
```

Buka http://127.0.0.1:8933/ai-learning-hub/ .

```sh
npm run lint
npm test
npm run build
npm run test:http
npm run preview
```

## Arsitektur

React + TypeScript + Vite, Reveal.js, KaTeX dan Shiki. Komponen tema, kode, matematika dan shell presentasi diadaptasi dari referensi pengguna; lihat THIRD_PARTY_NOTICES.md dan LICENSE.upstream. Struktur metadata per chapter adalah single source of truth. Vite glob memuat konten tanpa perubahan React setiap chapter; build menyalin resource statis dan dokumentasi. React lazy-load memuat Reveal saat presentasi dibuka.

- content/level-*/chapter-*/metadata.json: 40 chapter.
- content/levels.json: empat level.
- content/specializations/tracks.json: lima jalur pilihan.
- content/templates/: template teknis pengajar.
- src/: UI, router hash, content loader, presentation shell.
- docs/: standar akademik, referensi, lisensi, audit, panduan bahasa dan penulisan.
- scripts/: validasi metadata/resource, unit tests, build paths.

## GitHub Pages — persiapan saja

Repository yang disarankan: ai-learning-hub. Default base /ai-learning-hub/. Untuk nama lain atur environment VITE_BASE_PATH saat build, contoh PowerShell:

```powershell
$env:VITE_BASE_PATH='/nama-repo/'
npm run build
```

URL: https://USERNAME.github.io/ai-learning-hub/#chapter/13 . Hash routing mendukung refresh nested view; slides: #chapter/13/slides; template: #chapter/13/template; print: ?print-pdf=#chapter/13/template. Root hosting: VITE_BASE_PATH=/ .

Workflow `.github/workflows/pages.yml` divalidasi dan dideploy melalui GitHub Actions (`workflow_dispatch`). Aktifkan deployment melalui GitHub Pages → GitHub Actions pada repository GitHub.

Lihat `docs/ADDING_CHAPTER.md`, `docs/CONTENT_AUDIT.md`, `DESIGN.md`, dan `docs/DEVELOPMENT_PLAN.md` untuk pemeliharaan, panduan desain, dan referensi akademik.



