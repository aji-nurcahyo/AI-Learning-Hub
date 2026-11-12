# Panduan Praktik Laboratorium: AI Agents & Tool Integration
**Artificial Intelligence Learning Hub — Level 4 (Chapter 34)**  
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

---

## 1. DESKRIPSI DAN PRASYARAT PRAKTIKUM

Praktikum ini memberikan pengalaman praktis dalam merancang, membangun, dan menguji **AI Agent terarah (Supervised AI Agent)** yang mampu menyelesaikan tugas multi-langkah melalui pemanggilan alat (*tool calling*), mematuhi skema parameter ketat, menjalankan siklus penalaran ReAct, dan tunduk pada tata kelola keselamatan manusia (*Human-in-the-Loop*).

Seluruh sistem berjalan 100% mandiri (*offline*) pada CPU lokal, tanpa memerlukan API berbayar, tanpa unduhan model raksasa, dan menerapkan prinsip *least privilege* yang ketat (tanpa eksekusi shell bebas atau akses filesystem di luar sandbox).

### Prasyarat Pengetahuan
- Pemahaman konsep Intelligent Agents dan fungsi utilitas (Chapter 03).
- Pemahaman arsitektur model bahasa dan batasan inferensi teks (Chapter 29).
- Pemrograman Python berorientasi objek: kelas, enkapsulasi, penanganan eksepsi, skema JSON, dan ekspresi reguler.

### Lingkungan Eksekusi
- **Bahasa:** Python 3.10+ (Direkomendasikan Python 3.12).
- **Pustaka Utama:** Pustaka standar Python (`json`, `re`, `time`, `hashlib`, `datetime`, `enum`, `typing`).
- **Akselerasi:** Berjalan penuh pada CPU standar (RAM $\ge$ 4 GB).

---

## 2. PANDUAN LANGKAH DEMI LANGKAH EKSEKUSI

Praktikum ini terdiri atas 14 script Python mandiri di direktori `content/level-4-advanced-ai/chapter-34/code/`:

### Langkah 1: Arsitektur Dasar Agen & Siklus Keputusan
```bash
python content/level-4-advanced-ai/chapter-34/code/agent_architecture.py
```
- **Tujuan:** Memahami interaksi antara Controller agen, mesin kebijakan (*decision policy*), dan memori status (*agent state*).
- **Hasil Verifikasi:** Agen memproses sasaran (*goal*), memilih tindakan atomik, menerima pengamatan, dan mengakhiri sesi secara aman.

### Langkah 2: Registri Alat & Klasifikasi Profil Risiko
```bash
python content/level-4-advanced-ai/chapter-34/code/tool_registry.py
```
- **Tujuan:** Mendaftarkan alat komputasi dengan skema parameter deklaratif dan klasifikasi risiko eksplisit (`READ_ONLY` vs `STATE_CHANGING`).
- **Hasil Verifikasi:** Registri mengekspos spesifikasi fungsi JSON Schema standar industri.

### Langkah 3: Validasi Skema Parameter Masukan (JSON Schema)
```bash
python content/level-4-advanced-ai/chapter-34/code/tool_schema_validation.py
```
- **Tujuan:** Menguji validator skema yang memverifikasi kehadiran parameter wajib, memeriksa kesesuaian tipe data, dan menolak argumen asing ilegal.
- **Hasil Verifikasi:** Mencegah injeksi parameter liar sebelum fungsi dieksekusi oleh mesin.

### Langkah 4: Siklus Hidup Eksekusi Pemanggilan Alat (*Tool Calling*)
```bash
python content/level-4-advanced-ai/chapter-34/code/tool_calling_demo.py
```
- **Tujuan:** Menjalankan 5 tahapan pemanggilan alat terisolasi (Lookup -> Validation -> Permission -> Gate -> Execution) di dalam batas pengaman eksepsi.
- **Hasil Verifikasi:** Kesalahan waktu nyata (seperti pembagian nol) ditangkap tanpa mematikan proses utama.

### Langkah 5: Finite State Machine & Batas Langkah Eksekusi
```bash
python content/level-4-advanced-ai/chapter-34/code/agent_state.py
```
- **Tujuan:** Menerapkan mesin status terbatas (*Finite State Machine*) yang melarang transisi status ilegal dan menegakkan kuota langkah maksimum (*step budget*).
- **Hasil Verifikasi:** Transisi status terpelihara konsisten dari `INITIALIZED` hingga `TERMINATED`.

### Langkah 6: Siklus Penalaran & Tindakan ReAct
```bash
python content/level-4-advanced-ai/chapter-34/code/planning_loop.py
```
- **Tujuan:** Mempraktikkan alur ReAct (*Thought -> Action -> Observation -> Thought*) dengan dekomposisi tujuan dan pencegahan perulangan tak hingga (*infinite loop*).
- **Hasil Verifikasi:** Jejak pemikiran dan urutan tindakan tercatat secara berurutan dan transparan.

### Langkah 7: Gerbang Persetujuan Manusia (*Human Approval Gate*)
```bash
python content/level-4-advanced-ai/chapter-34/code/human_approval_gate.py
```
- **Tujuan:** Menangguhkan tindakan berisiko tinggi (`STATE_CHANGING`) hingga pengguna manusia memberikan konfirmasi izin secara sadar (*Human-in-the-Loop*).
- **Hasil Verifikasi:** Aksi baca-saja disetujui otomatis, sedangkan aksi modifikasi diblokir jika otorisasi ditolak.

### Langkah 8: Penegakan Prinsip Hak Akses Terendah (*Least Privilege*)
```bash
python content/level-4-advanced-ai/chapter-34/code/permission_manager.py
```
- **Tujuan:** Menerapkan kontrol akses berbasis peran (RBAC) dengan pemisahan cakupan izin (*scopes*) untuk peran Mahasiswa, Dosen, dan Administrator.
- **Hasil Verifikasi:** Upaya pemanggilan alat di luar hak peran diblokir pada tingkat manajer izin.

### Langkah 9: Penanganan Kesalahan Alat & Panduan Koreksi Diri
```bash
python content/level-4-advanced-ai/chapter-34/code/tool_error_handling.py
```
- **Tujuan:** Membungkus pesan galat runtime menjadi observasi informatif yang memberikan petunjuk kepada agen untuk memperbaiki tindakannya sendiri.
- **Hasil Verifikasi:** Agen menerima masukan konstruktif saat terjadi kegagalan aritmatika atau parameter hilang.

### Langkah 10: Jejak Audit Kriptografis Tahan Manipulasi (*Tamper-Evident*)
```bash
python content/level-4-advanced-ai/chapter-34/code/audit_logging.py
```
- **Tujuan:** Mencatat setiap jejak pemikiran, argumen alat, dan keputusan persetujuan dalam struktur rantai hash SHA-256 yang saling mengunci.
- **Hasil Verifikasi:** Modifikasi ilegal pada baris riwayat audit akan langsung membatalkan verifikasi integritas rantai.

### Langkah 11: Konsep Model Context Protocol (MCP) Berbasis JSON-RPC 2.0
```bash
python content/level-4-advanced-ai/chapter-34/code/mcp_concepts_demo.py
```
- **Tujuan:** Mempelajari arsitektur klien-server protokol terbuka MCP melalui pertukaran pesan metode `tools/list`, `tools/call`, dan `resources/read`.
- **Hasil Verifikasi:** Klien berhasil menemukan alat secara dinamis dan membaca sumber daya konteks silabus.

### Langkah 12: Reference Project: Supervised AI Learning Assistant
```bash
python content/level-4-advanced-ai/chapter-34/code/supervised_agent.py
```
- **Tujuan:** Menjalankan proyek acuan lengkap yang mengintegrasikan seluruh komponen dalam sebuah asisten AI akademik yang diawasi penuh.
- **Hasil Verifikasi:** Menguji skenario pencarian materi aman, penolakan izin modifikasi mahasiswa, dan persetujuan tindakan admin.

### Langkah 13: Suite Evaluasi Benchmark Kinerja Agen
```bash
python content/level-4-advanced-ai/chapter-34/code/agent_evaluation.py
```
- **Tujuan:** Mengukur secara objektif *Task Success Rate*, *Tool Selection Accuracy*, dan *Permission Compliance Rate* pada 5 skenario benchmark.
- **Hasil Verifikasi:** Memperoleh nilai evaluasi kuantitatif 100% kepatuhan tata kelola.

### Langkah 14: Alur Kerja Defensif terhadap Prompt Injection pada Alat
```bash
python content/level-4-advanced-ai/chapter-34/code/safe_agent_workflow.py
```
- **Tujuan:** Mensterilkan muatan teks musuh yang diselipkan pada keluaran alat (*indirect prompt injection*) dan memastikan agen memperlakukan keluaran sebagai data pasif murni.
- **Hasil Verifikasi:** Perintah sistem ilegal dalam teks alat dinetralisir sebelum memengaruhi kebijakan agen.

---

## 3. LATIHAN BERTINGKAT (3 TIERS)

### Tingkat Dasar: Pemahaman & Replikasi
1. **Latihan 1.1:** Tambahkan sebuah tool baru bernama `string_reverser` pada `tool_registry.py` yang menerima satu parameter string `"text"` dan mengembalikan teks terbalik. Daftarkan dengan profil risiko `READ_ONLY`.
   *Solusi Mandiri:* Implementasikan fungsi pembalik string `return text[::-1]`, definisikan skema JSON dengan properti `text` bertipe string dalam `required`, dan daftarkan ke registri menggunakan `registry.register_tool()`.
2. **Latihan 1.2:** Pada `agent_state.py`, ubah nilai `max_steps = 2`. Jalankan tugas yang membutuhkan 3 langkah eksekusi. Amati status terminasi yang dihasilkan oleh FSM!
   *Solusi Mandiri:* FSM akan menghentikan eksekusi dengan status `TERMINATED_MAX_STEPS` untuk mencegah pemborosan komputasi.

### Tingkat Menengah: Eksperimen & Optimasi
1. **Latihan 2.1:** Modifikasi `permission_manager.py` untuk menambahkan peran baru `AUDITOR`. Peran auditor hanya diizinkan membaca log dan memanggil tool `search_knowledge_base`, namun dilarang keras memanggil tool matematika `calculator` maupun tool modifikasi.
2. **Latihan 2.2:** Pada `human_approval_gate.py`, implementasikan batas waktu persetujuan (*approval timeout*). Jika pengguna manusia tidak memberikan respon dalam kurun waktu 10 detik, sistem harus secara otomatis memilih tindakan `DENIED` (*fail-safe default*).

### Tingkat Mahir: Desain & Implementasi Lanjutan
1. **Latihan 3.1:** Rancang detektor perulangan (*loop detector*) pada `planning_loop.py` yang mendeteksi jika agen memanggil tool yang sama dengan argumen yang identik sebanyak dua kali berturut-turut tanpa menghasilkan progres baru, lalu paksa agen mengubah strateginya.
2. **Latihan 3.2:** Hubungkan `supervised_agent.py` dengan pipeline RAG dari Chapter 33, sehingga tool `search_knowledge_base` melakukan pencarian semantik nyata di atas `ai_course_corpus.json` dan menyertakan sitasi `[Doc: <id>]` dalam observasi agen.

---

## 4. PANDUAN MINI PROJECT

### Judul Proyek
**Building a Supervised Tool-Using AI Agent**

### Deskripsi Tugas
Mahasiswa diminta merancang dan membangun purwarupa sistem agen cerdas terarah (*Supervised AI Agent*) berbasis Python modular. Agen harus memiliki kemampuan perencanaan ReAct, mengelola memori status FSM, terhubung ke minimal dua tool baca (`READ_ONLY`) dan satu tool modifikasi (`STATE_CHANGING`), menegakkan gerbang persetujuan manusia (*human-in-the-loop*), mensterilkan muatan keluaran alat dari injeksi instruksi, serta mencatat seluruh alur penalaran ke dalam jejak audit kriptografis berantai SHA-256.

### Luaran yang Wajib Dikumpulkan
1. **Laporan Arsitektur & Analisis (PDF/Markdown):**
   - Diagram arsitektur modular agen dan diagram transisi status FSM.
   - Definisi skema JSON Schema untuk seluruh alat yang didaftarkan.
   - Analisis tata kelola persetujuan manusia dan matriks izin RBAC.
   - Hasil evaluasi kinerja pada minimal 5 skenario uji terstandar.
2. **Kode Sumber (Python):**
   - Berkas controller, registri alat, pengelola izin, gerbang persetujuan, dan audit logger yang teruji 100% mandiri pada CPU.
3. **Berkas Jejak Audit (JSON):**
   - Log eksekusi berantai yang membuktikan validitas hash kriptografis pada skenario yang disetujui maupun yang ditolak.

---

## 5. RUBRIK PENILAIAN (100 POIN)

| Komponen Penilaian | Bobot | Kriteria Pemenuhan Standar |
| :--- | :---: | :--- |
| **1. Architecture & Problem Understanding** | **15 Poin** | Pemahaman mendalam mengenai perbedaan chatbot, workflow, dan agent; dekomposisi tujuan yang tepat; serta perancangan diagram FSM yang lengkap. |
| **2. Tool Integration & Schema Engineering** | **25 Poin** | Definisi skema JSON Schema yang valid dan lengkap; penegakan validasi tipe data parameter masukan; isolasi batas penanganan galat eksekusi tanpa mematikan sistem. |
| **3. Agent Workflow, State & Memory** | **20 Poin** | Implementasi siklus ReAct yang transparan (*Thought -> Action -> Observation*); penegakan kuota langkah maksimum; dan manajemen memori kerja yang disiplin. |
| **4. Security, Human Approval & Governance** | **25 Poin** | Penegakan prinsip *least privilege* (RBAC); gerbang persetujuan manusia untuk aksi `STATE_CHANGING`; penanganan penolakan izin yang anggun; pertahanan terhadap injeksi teks alat; serta rantai jejak audit SHA-256 yang valid. |
| **5. Testing, Benchmark & Documentation** | **15 Poin** | Suite evaluasi kuantitatif yang menguji *success rate* dan kepatuhan izin; kebersihan kode sumber berstandar PEP-8; serta panduan reproduksibilitas 100% pada CPU. |

---

## 6. DAFTAR PUSTAKA & REFERENSI AKADEMIK

1. **Yao, S., et al. (2022).** *ReAct: Synergizing Reasoning and Acting in Language Models.* International Conference on Learning Representations (ICLR 2023).
2. **Schick, T., et al. (2023).** *Toolformer: Language Models Can Teach Themselves to Use Tools.* Advances in Neural Information Processing Systems (NeurIPS 2023).
3. **Anthropic. (2024).** *Model Context Protocol (MCP) Specification.* Open-source standard for AI tool integrations. https://modelcontextprotocol.io.
4. **Russell, S., & Norvig, P. (2020).** *Artificial Intelligence: A Modern Approach (4th Edition).* Pearson.
5. **National Institute of Standards and Technology (NIST). (2023).** *Artificial Intelligence Risk Management Framework (AI RMF 1.0).* NIST AI 100-1.

