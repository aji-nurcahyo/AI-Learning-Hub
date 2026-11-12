# Track B — NLP, LLM & Generative AI: Overview, Roadmap & Studi Kasus

**Fokus Jalur:** Pemrosesan bahasa alami, representasi TF-IDF & embeddings, arsitektur Transformer, *Large Language Models*, penelusuran semantik, RAG tergrounding, serta keamanan agen.

---

## 1. Pengantar Jalur Spesialisasi
Jalur **NLP, LLM & Generative AI** membimbing mahasiswa merangkai teknik pemrosesan bahasa dari dasar statistik leksikal hingga arsitektur aplikasi LLM tergrounding dan aman di lingkungan CPU lokal. Bila model generatif lokal berukuran besar tidak tersedia, jalur ini menggunakan *extractive retrieval baseline* secara jujur tanpa menyamakannya dengan model generatif penuh.

## 2. Roadmap Pembelajaran Terkurasi
1. **Tahap Fondasi:** **Chapter 07** (Python Ecosystem), **Chapter 11** (Intro ML), **Chapter 14** (Classification & Naive Bayes).
2. **Tahap Inti NLP & Transformer:**
   - **Chapter 27** — Natural Language Processing Fundamentals.
   - **Chapter 28** — Attention & Transformer Architecture.
   - **Chapter 29** — Large Language Models & Generative AI.
3. **Tahap Lanjutan Aplikasi & Keamanan:**
   - **Chapter 31** — Transfer Learning & PEFT/LoRA.
   - **Chapter 32** — Embeddings & Vector Search.
   - **Chapter 33** — Retrieval-Augmented Generation (RAG).
   - **Chapter 34** — AI Agents & Tool Integration.
   - **Chapter 39** — AI Safety, Security & Governance.

## 3. Studi Kasus Domain NLP & LLM

### Studi Kasus 1: Asisten Pencarian Silabus & Modul Akademik Tergrounding
- **Masalah:** Pencarian kata kunci biasa sering melewatkan istilah sinonim sementara jawaban bebas berisiko berhalusinasi.
- **Solusi:** Hybrid Search (BM25 + Dense Vector via RRF pada `Chapter 33`) disertai atribusi sitasi `[ChXX]` dan penolakan otomatis (`ABSTAIN`) saat kemiripan berada di bawah ambang (`nlp_grounded_qa_lab.py`).

### Studi Kasus 2: Pertahanan Prompt Injection pada Dokumen Eksternal
- **Masalah:** Dokumen pihak ketiga dapat menyisipkan teks instruksi tersembunyi (`"Ignore previous instructions"`).
- **Solusi:** Isolasi konten tak tepercaya menggunakan delimiter XML (`Chapter 39`), penyaringan pola injeksi sebelum retrieval/eksekusi alat, dan RBAC (*Chapter 34*).

