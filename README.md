# Artificial Intelligence Learning Hub

**From AI Fundamentals to Intelligent Systems**

Platform terbuka berbahasa Indonesia untuk Artificial Intelligence, Machine Learning, Deep Learning, Generative AI, dan pengembangan intelligent systems.

Creator & Developer: **Aji Pamungkas Tri Nurcahyo** · **Institut Teknologi Sumatera**.

Tahap awal: empat level, 40 chapter Coming Soon, lima track overview, template presentasi dan pipeline konten. Tidak ada akun, tracking progres, unlocking, ranking atau sertifikat. Template tidak dianggap materi chapter selesai.

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

Workflow .github/workflows/pages.yml berjalan hanya melalui workflow_dispatch, memvalidasi lalu menyiapkan artifact dan deploy jika kelak dijalankan pemilik. Aktifkan GitHub Pages → GitHub Actions setelah repository sendiri dibuat. Tidak ada remote yang disalin/diubah dan tidak ada push/deployment selama implementasi ini.

Lihat docs/ADDING_CHAPTER.md, docs/CONTENT_AUDIT.md dan docs/DEVELOPMENT_PLAN.md untuk pemeliharaan, hasil tes aktual, batasan, dan roadmap.
