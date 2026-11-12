# Rencana pengembangan

## Fondasi yang digunakan kembali
Referensi lokal dibaca saja. Theme styles.css, usePreferences/ThemeSelect, Icon, Code, Shiki highlight, KaTeX Math, dan DeckShell Reveal.js disalin lalu diadaptasi. TypeScript/ESLint dan dependency lock memakai fondasi referensi. UI mempertahankan typography, token warna, kartu, toolbar dan pola navigation/search. Registry chapter hardcoded diganti import.meta.glob metadata; renderer slide dipisahkan dari output Python lama. Tidak menyalin .git, remote, materi lama, dataset, slide kursus lama, atau workflow deployment tujuan lama.

Perbaikan: namespace preferences ai-hub, default 16:9, print mempertahankan route, empat level, resource availability eksplisit, konten JSON/Markdown, safe text renderer, hash router dan build checks. Validasi resource dan build artifact mengikuti pola skrip referensi namun diimplementasikan untuk kontrak AI.

## Roadmap editorial
1. Chapter 01–06 dan 09: review konsep, logic/probability/search, diagram, latihan dan risiko; verifikasi perhitungan manual.
2. Chapter 07–08 dan 10: environment CPU, dataset kecil berlisensi jelas, satu capstone foundations.
3. Chapter 11–20: sklearn, split sebelum preprocessing, baseline, cross-validation dan error analysis.
4. Chapter 21–30: PyTorch CPU dahulu, kemudian eksperimen GPU opsional; lisensi YOLO/model diperiksa per artefak.
5. Chapter 31–40: retrieval evaluation, agent sandbox/tool permission, monitoring, optimization dan governance; capstone terintegrasi.
6. Track spesialisasi setelah fondasi relevan tervalidasi. Track tidak mengunci akses chapter.

Setiap batch melewati review akademik, eksekusi, lisensi, accessibility/responsive dan presentasi. PDF/PPTX ditambahkan hanya bila file nyata telah diverifikasi. Tidak ada training backend atau API berbayar wajib.

## Batas teknis
Markdown sengaja subset kecil; tidak mendukung HTML/MDX, tabel atau inline link rendering. Diagram slide memakai workflow langkah; model architecture dapat dijelaskan dalam diagram sederhana atau dikembangkan dengan asset reviewer pada iterasi berikutnya. Notebook dieksekusi lokal, bukan browser/server Pages. Tidak ada materi lengkap maupun hasil model saat tahap ini.

## Validasi tambahan

Jalankan npm run test:http setelah build untuk memeriksa response HTTP entry assets dan resource produksi. npm run check:licenses memperbarui register versi/lisensi dari paket terpasang; commit perubahan notice bersamaan dengan lockfile. Visual QA belum selesai karena akses browser terhalang; lihat CONTENT_AUDIT.md.
