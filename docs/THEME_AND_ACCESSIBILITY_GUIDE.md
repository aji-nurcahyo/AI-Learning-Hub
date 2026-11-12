# Theme System, Accessibility & Visual QA Guide
**Artificial Intelligence Learning Hub — Institut Teknologi Sumatera**

Panduan ini menjelaskan arsitektur tema **Light / Dark / Auto**, daftar *semantic design tokens*, aturan kontras **WCAG 2.2 AA**, penanganan SVG/ikon/KaTeX/chart/slide, serta cara menjalankan pengujian otomatis `npm run test:theme`.

---

## 1. Existing Theme Architecture

- Pengaturan preferensi tema dikelola oleh [`usePreferences`](../src/deck-shell/preferences.tsx) melalui atribut `document.documentElement.dataset.theme` (`'auto'`, `'light'`, atau `'dark'`) dan disimpan pada `localStorage` dengan kunci `ai-hub.theme` serta parameter URL opsional `?theme=light|dark|auto`.
- Seluruh definisi token berada di satu sumber otoritatif: [`src/theme/styles.css`](../src/theme/styles.css):
  - `:root, :root[data-theme="light"]` untuk Light Mode.
  - `:root[data-theme="dark"]` untuk Dark Mode eksplisit.
  - `@media (prefers-color-scheme: dark)` membungkus `:root[data-theme="auto"]` sehingga preferensi sistem tidak pernah menimpa pilihan eksplisit pengguna (`light` atau `dark`).

---

## 2. Semantic Design Tokens

| Kategori | Token Utama | Fungsi |
| :--- | :--- | :--- |
| **Background / Surface** | `--bg`, `--surface`, `--surface2`, `--surface-high`, `--surface-hover`, `--surface-translucent` | Latar halaman, kartu, permukaan tinggi, hover interaktif, header transparan. |
| **Text Hierarchy** | `--ink`, `--ink-secondary`, `--muted`, `--on-primary` | Teks utama, sekunder, keterangan pendukung, serta teks di atas warna primer. |
| **Borders** | `--line`, `--line-subtle`, `--line-strong` | Garis pemisah standar, tipis, dan tegas antar komponen. |
| **Interactive & Brand** | `--primary`, `--primary-hover`, `--primary-soft`, `--brand`, `--brand-dark`, `--accent`, `--cyan`, `--link`, `--focus-ring` | Warna identitas, tombol utama, hover state, tautan, dan ring fokus keyboard (`:focus-visible`). |
| **Status Indicators** | `--status-success[-bg]`, `--status-warning[-bg]`, `--status-error[-bg]`, `--status-info[-bg]` | Lencana status berstandar kontras AA ($\ge 4.5:1$) di Light maupun Dark Mode. |
| **Content & Code** | `--font-mono`, `--code-bg`, `--code-fg`, `--shiki-*`, `--table-header-bg`, `--table-border`, `--math-fg`, `--blockquote-border` | Pewarnaan blok kode Shiki adaptif dua tema, tabel, kutipan, serta formula KaTeX. |
| **Charts / Visuals** | `--chart-bg`, `--chart-axis`, `--chart-grid` | Wadah terang stabil untuk gambar grafik statis Matplotlib pada kedua tema. |

---

## 3. Light / Dark Styling Conventions & SVG / Icon Rules

- **Dilarang hardcode `#ffffff` atau `#000000` untuk teks/latar UI** tanpa token semantik (`var(--ink)`, `var(--on-primary)`, `var(--surface)`).
- **Ikon Monokrom (`Material Symbols Rounded` / [`Icon.tsx`](../src/ui/Icon.tsx)):** Mewarisi `currentColor` dari elemen induknya; setiap tombol ikon wajib memiliki `aria-label` atau `title` deskriptif.
- **Inline SVG (mis. `.network-art` pada Beranda):** Menggunakan `currentColor` / `var(--primary)` untuk gradien dan garis agar tetap serasi di Light dan Dark Mode.
- **Anti-Override Theme:** Hindari mendefinisikan ulang `:root` pada file CSS sekunder (`ai.css`) yang dapat menabrak `[data-theme="dark"]` di `styles.css`.

---

## 4. KaTeX, Markdown, Charts & Presentation Mapping

- **Markdown & Syntax Highlighting:** Token `--shiki-*` pada `styles.css` memiliki palet terpisah untuk Light (`--shiki-token-comment: #475569` demi kontras $\ge 4.5:1$) dan Dark (`#94a3b8`).
- **KaTeX:** `.katex, .katex-display { color: var(--ink) !important; }` memastikan rumus selalu mengikuti tema halaman.
- **Static Matplotlib Charts:** Selektor `.reading-article img, .concept-diagram img` menerapkan latar `var(--chart-bg)` (`#ffffff`), border, dan padding agar sumbu gelap grafik ekspor PNG tetap terbaca di Dark Mode tanpa mengubah piksel atau data benchmark.
- **Presentation Engine Mapping (Internal):** Latar, teks, progress bar, dan nomor slide dikunci melalui `.deck-page .reveal` agar memiliki spesifisitas lebih tinggi dari stylesheet bawaan pustaka slide, tanpa menampilkan nama framework ke pengguna (tetap berlabel **Slide Perkuliahan**).

---

## 5. Accessibility Requirements (WCAG 2.2 AA) & Running `npm run test:theme`

Target ambang kontras minimum:
- Teks normal: **$\ge 4.5:1$**
- Teks besar ($\ge 24\text{px}$ atau bold $\ge 18.67\text{px}$): **$\ge 3.0:1$**
- Komponen UI & indikator fokus (`:focus-visible`): terlihat kontras pada kedua tema.

### Menjalankan Audit Tema Otomatis Lokal
```bash
npm run build
npm run test:theme
```
- Opsi penyimpanan screenshot Light/Dark: `node scripts/test-theme.mjs --screenshots` (output tersimpan di `reports/theme-audit/` yang diabaikan oleh `.gitignore`).
- Skrip akan keluar dengan kode non-nol bila ditemukan pelanggaran kontras terkonfirmasi (*confirmed failure*), *overflow* horizontal/slide, error KaTeX, atau eksepsi JavaScript.

### Debugging Kegagalan Kontras & Penambahan Komponen Baru
1. Buka `reports/theme-audit/theme-audit-report.json` bagian `detailsWithIssues` untuk melihat `route`, `theme`, `viewport`, `selector`, nilai RGB `fg`/`bg`, serta `ratio`.
2. Pastikan komponen baru memakai token `--surface*`, `--ink`, `--muted`, `--primary`, `--on-primary`, dan `--line` tanpa hardcode warna atau variabel tak terdefinisi.
3. Jalankan kembali `npm run build && npm run test:theme` untuk memverifikasi perbaikan.

---

## 6. Known Limitations
- Audit browser otomatis via `npm run test:theme` membutuhkan Microsoft Edge atau Google Chrome lokal pada mesin pengujian. Pipeline CI GitHub Actions (`.github/workflows/ci.yml`) menjalankan validasi konten, TypeScript, ESLint, unit tests, production build, dan HTTP preview check pada lingkungan Linux Ubuntu, sedangkan inspeksi visual Edge Headless CDP dilakukan pada verifikasi lokal sebelum rilis.
- Pemindai kontras otomatis mengevaluasi teks & kontrol DOM nyata; grafik ekspor `.png` statis dan kasus kompleks berlatar gradien ditandai untuk tinjauan visual manual berkala.

