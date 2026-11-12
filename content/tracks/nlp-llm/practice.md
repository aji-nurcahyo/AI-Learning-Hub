# Track B — NLP, LLM & Generative AI: Applied Laboratories

Panduan delapan laboratorium terapan yang menghubungkan pemrosesan bahasa klasik, Transformer, Vector Search, RAG, hingga keamanan Agent di CPU lokal.

---

## 1. Lab Domain Jalur NLP & Grounded QA (Run Langsung)
```bash
python content/tracks/nlp-llm/labs/nlp_grounded_qa_lab.py
```
- **Output Terverifikasi:** Menjawab pertanyaan relevan ber-sitasi `[Ch28]` / `[Ch33]`, menolak menjawab pertanyaan tanpa bukti (`ABSTAIN`), dan memblokir pola injeksi (`BLOCKED_INJECTION`). Catatan jujur: baseline ini menggunakan *deterministic TF-IDF extractive retrieval*, bukan LLM generatif penuh.

---

## 2. Daftar Delapan Laboratorium Terintegrasi

1. **Lab 1 — Text Preprocessing (Chapter 27):** `bag_of_words.py` & `ngram_features.py`.
2. **Lab 2 — TF-IDF & Sequence Classification (Chapter 27):** `text_classification.py` & `rnn_lstm_gru_comparison.py`.
3. **Lab 3 — Embedding Similarity (Chapter 32):** `cosine_similarity.py`.
4. **Lab 4 — Attention Mechanism (Chapter 28):** `scaled_dot_product_attention.py` & `multi_head_attention_torch.py`.
5. **Lab 5 — Semantic Search (Chapter 32):** `semantic_search.py`.
6. **Lab 6 — RAG Pipeline (Chapter 33):** `hybrid_retrieval.py`.
7. **Lab 7 — Grounded Answer Evaluation (Chapter 33 & Track B):** `citation_validation.py` & `nlp_grounded_qa_lab.py`.
8. **Lab 8 — Safe Tool Calling (Chapter 34 & 39):** `human_approval_gate.py` & `prompt_injection_defense_demo.py`.

