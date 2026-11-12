# Track B Capstone Specification: Grounded Academic AI Assistant

**Jalur:** Track B — NLP, LLM & Generative AI  
**Lingkungan Wajib:** CPU Lokal (Gunakan *deterministic extractive/retrieval baseline* bila LLM lokal tidak tersedia—**dilarang** menyebut prototipe retrieval-only sebagai sistem LLM generatif penuh).

---

## 1. Deskripsi Proyek
Bangun asisten tanya-jawab akademik berbasis korpus materi terbuka **AI Learning Hub** (atau korpus berlisensi terbuka lainnya) dengan fokus akurasi penelusuran, kejujuran sitasi sumber, dan keamanan terhadap injeksi prompt.

## 2. Fitur & Deliverables Wajib
1. **Document Ingestion & Chunking:** Pemrosesan korpus bersih lengkap dengan metadata sumber dan ukuran chunk teruji.
2. **Hybrid Document Retrieval:** Kombinasi leksikal (BM25/TF-IDF) dan semantik (*Dense Vectors*) lengkap dengan metrik `Recall@k` / `MRR`.
3. **Source Attribution:** Setiap jawaban menyertakan rujukan `[Doc: ID]` terverifikasi.
4. **Unsupported Answer Detection (Abstention):** Menolak menjawab secara jujur bila kueri berada di luar cakupan korpus.
5. **Security Checks:** Uji otomatis pemblokiran *direct/indirect prompt injection*.
6. **Evaluation & Limitations Report:** Pembahasan jelas perbedaan sistem berbasis retrieval ekstraktif vs model generatif LLM penuh.

## 3. Rubrik Penilaian Track B (Total 100 Poin)

| Komponen Penilaian | Bobot | Kriteria Keunggulan |
| :--- | :---: | :--- |
| **Ingestion, Chunking & Retrieval Architecture** | **25 poin** | Indeks retrieval berjalan cepat di CPU dengan *chunk overlap* & provenance jelas. |
| **Grounding, Citation & Abstention** | **25 poin** | Jawaban terverifikasi mengutip sumber sah dan jujur menolak kueri di luar korpus. |
| **Security & Prompt Boundary Controls** | **20 poin** | Pola injeksi diisolasi sebagai data inert tanpa kebocoran konteks atau eksekusi liar. |
| **Quantitative Evaluation (`Recall@k`, `MRR`, Latency)** | **20 poin** | Benchmark metrik retrieval & latensi CPU terdokumentasi rapi terhadap baseline sederhana. |
| **Documentation & Honest Scope Disclaimer** | **10 poin** | Membedakan jujur kapabilitas *extractive baseline* vs LLM generatif beserta keterbatasannya. |

