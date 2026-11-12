# DESIGN.md: Modern Academic Learning Library

Project: **Artificial Intelligence Learning Hub** (`AI Learning Hub`)  
Tagline: From AI Fundamentals to Intelligent Systems  
Institution: Institut Teknologi Sumatera  
Author: Aji Pamungkas Tri Nurcahyo

`Dial: ENERGY 2 / RHYTHM 3 / MOTION 1`

> Design Read: Open-access university AI textbook and curriculum portal for students and lecturers at Institut Teknologi Sumatera, built in a **Modern Academic Learning Library** editorial visual language, dial **ENERGY 2 / RHYTHM 3 / MOTION 1**.

---

## 1. Visual Direction & Character

**Modern Academic Learning Library** treats AI curriculum delivery like a structured university reference work paired with clean interactive lab tools. Instead of dark neon hero bands, glowing orbs, frosted-glass floating cards, or promotional SaaS taglines, every interface surface serves study comprehension, rapid chapter discovery, readable long-form reading, and projection clarity.

Non-Clone Declaration (`R-30`): Layout composition draws from academic presses, university syllabus catalogs, and digital reference monographs rather than Linear, Vercel, Stripe, or Notion templates.

---

## 2. Three Liveliness Dials & Design Levers

| Dial | Level | Application in AI Learning Hub |
| :--- | :--- | :--- |
| **ENERGY** | `2 (Balanced)` | Crisp display serif-inspired typographic weight (`700–800`), deep royal cobalt accent reserved for focal CTAs, active tabs, and level progression markers on calm high-contrast slate/paper surfaces. |
| **RHYTHM** | `3 (Varied)` | Distinct composition per section on the landing page and overview pages: (1) asymmetric editorial Hero + curriculum architecture diagram, (2) stepped 4-stage level index, (3) interactive search + tabular syllabus index rows, (4) structured 2-column specialization track catalog with lab/capstone signals, (5) quiet academic attribution strip. |
| **MOTION** | `1 (Hover / Focus Only)` | Functional border, underline, and background color shifts (`0.15s ease`) on pointer hover and `3px` `:focus-visible` keyboard outlines. Zero ambient pulse animations, zero infinite keyframes, and zero bounce/floating card motion; respects `prefers-reduced-motion: reduce`. |

Core Levers:
1. **One focal point per screen:** Hero CTA button on Homepage, Chapter `<h1>` + `Modul Pembelajaran` on Chapter Overview, centered prose stream on Reading mode, active slide canvas on Slides.
2. **Whitespace as structure:** Generous editorial padding separates syllabus sections without heavy boxes or full-width colored divider banners.
3. **One deliberate accent:** Cobalt primary (`#1556c0` Light / `#8ab4f8` Dark) marks primary navigation, active tab selection, and interactive focus alongside neutral slate surfaces; teal focus accent (`#006878` / `#70d8ec`) anchors keyboard accessibility outlines cleanly.
4. **Identity motif:** Paired `ai_` monospace catalog mark, tabular chapter numbers (`01`–`40`), and clean architectural hairline frames (`1px solid var(--line)`).

---

## 3. Color System (`R-25`, `R-29`, `R-34`)

Active palette stays strictly within **1 neutral base + 1 primary + 1 accessibility accent**:

| Role | Light Mode Token | Dark Mode Token | Purpose & Minimum Contrast |
| :--- | :--- | :--- | :--- |
| Page background (`--bg`) | `#f8fafc` | `#0b1320` | Paper/midnight ground with zero glow orbs |
| Card surface (`--surface`) | `#ffffff` | `#131f33` | Primary reading and panel canvas |
| Secondary surface (`--surface2`) | `#f1f5f9` | `#1b2a44` | Code bars, table headers, prerequisite sidebar, callouts |
| Primary text (`--ink`) | `#0f172a` | `#f8fafc` | Headings and reading prose (`>= 17:1`) |
| Muted text (`--muted`) | `#475569` | `#94a3b8` | Descriptions, metadata, subtitles (`>= 7.2:1`) |
| Brand primary (`--primary`) | `#1556c0` | `#8ab4f8` | Primary CTA, active tab, focal links (`on-primary` `#ffffff` / `#041e49`, `>= 6.7:1`) |
| Focus accent (`--focus-ring`) | `#006878` | `#70d8ec` | Keyboard `:focus-visible` indicator (`>= 3.0:1` UI contrast) |
| Matplotlib container (`--chart-bg`) | `#ffffff` | `#ffffff` | Preserves dark PNG chart axis readability on Dark Mode |

---

## 4. Typography & Editorial Scale (`R-06`)

- **Primary sans (`'Google Sans Flex Variable', system-ui, sans-serif`):** Selected for crisp Indonesian diacritics, dense technical explanations, and balanced stroke contrast across desktop monitors and classroom projectors.
- **Technical monospace (`'Google Sans Code Variable', monospace`):** Restricted strictly to code blocks, `.inline-code`, `ai_` catalog badge, and tabular chapter index numerals (`01`–`40`), never used for oversized decorative headings.
- **Type Hierarchy:**
  - Display hero H1: `clamp(2.4rem, 4.2vw, 3.8rem)`, tight letter-spacing `-0.035em`.
  - Page/chapter H1: `clamp(2.1rem, 3.6vw, 3.2rem)`.
  - Section H2: `1.65rem`–`1.85rem`.
  - Reading prose (`p`, `li`, `blockquote`): `1.04rem`, `line-height: 1.85`, constrained to `max-width: 72ch` for comfortable eye tracking (`~65–75` chars/line), while code cards, math blocks, and data tables expand up to container width (`100%`).

---

## 5. Border Radius, Elevation, Iconography & Copy Rules (`R-02`, `R-04`, `R-08`, `R-11`, `R-12`, `R-15`)

- **3-Tier Radius Scale (`R-11`):**
  - `--radius-sm: 6px`: `<select>`, `<input>`, compact tags, quick links.
  - `--radius-md: 10px`: primary/secondary buttons, tab pills, code cards, callout notices.
  - `--radius-lg: 14px`: level panels, track panels, chapter action cards.
  - `--radius-pill: 999px`: reserved strictly for non-button semantic status/level badges (`.status-pill`, `.level-badge`, `.slide-number`).
- **Elevation (`R-12`):** Surfaces stay flat (`1px solid var(--line)`) at rest; slight surface tint or border sharpening signals interactivity without heavy blurred card shadows or hover bounce transforms.
- **Header (`R-10`, `R-03`):** Solid `--surface` sticky header on desktop (`> 960px`) without translucent blur stacking; transitions to non-sticky flow (`position: static`) on tablet/mobile (`<= 960px`) so small screens preserve full vertical reading height.
- **Icons & Arrows (`R-04`, `R-08`):** Single icon set (`Material Symbols Rounded`) only for functional tools (theme, print, slide controls, copy code, keyboard help). Directional arrows are limited to: `↗` for external URLs (`target="_blank"`), `←`/`→` for sequential previous/next chapter/module/slide navigation, and `↓` for file downloads (`.ipynb` / docs `.md`).
- **Copywriting (`R-02`, `R-15`, `R-16`):** Natural Indonesian academic prose preserving English technical terms; zero em dashes (`—`) in UI strings; zero hype adjectives or generic CTA templates; every displayed statistic matches verifiable repository data (`08` setup modules, `04` levels, `40` chapters, `05` specialization tracks).

---

## 6. Documented One-Line Reasons (`R-31`)

1. **Color Palette:** High-contrast slate surfaces paired with single cobalt brand primary `#1556c0`/`#8ab4f8` distinguish actionable academic paths without visual noise.
2. **Hero Layout:** Asymmetric prose column paired with a clean 4-level architectural SVG schematic replaces ambient radial glow with structural curriculum orientation.
3. **Typography:** Pairing sans prose at `72ch` measure with tabular monospace codes maximizes readability across long-form modules, equations, and code blocks.
4. **Spacing & Radius:** Three discrete radii (`6px`, `10px`, `14px`) separate form controls, buttons, and containers so controls read immediately by silhouette.
5. **Cards vs Lists:** Chapters render as compact filterable index rows (`01`–`40`) while Levels and Tracks use distinct structured panels reflecting pedagogical hierarchy.
6. **Iconography & Motion:** Monochrome Material Symbols support toolbar affordances only, and hover/focus states omit looping pulse keyframes (`MOTION 1`).

