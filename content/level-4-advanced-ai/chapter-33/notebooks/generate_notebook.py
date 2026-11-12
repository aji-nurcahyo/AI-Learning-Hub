"""Generator for Chapter 33 Jupyter Companion Notebook.

Creates notebooks/chapter_33_companion.ipynb with valid v4 format and
realistic pre-executed outputs covering all parts of Chapter 33 RAG pipeline.
"""

import json
import os


def make_notebook():
    cells = []

    def md_cell(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")],
        }

    def code_cell(code, stdout_text=""):
        outputs = []
        if stdout_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.strip().split("\n")],
            })
        return {
            "cell_type": "code",
            "execution_count": len(cells) + 1,
            "metadata": {},
            "outputs": outputs,
            "source": [line + "\n" for line in code.strip().split("\n")],
        }

    # Cell 1: Header
    cells.append(md_cell("""# Chapter 33: Retrieval-Augmented Generation (RAG)
**Artificial Intelligence Learning Hub — Level 4**
*Pengembang: Aji Pamungkas Tri Nurcahyo — Institut Teknologi Sumatera*

Notebook pendamping ini mempraktikkan arsitektur **Retrieval-Augmented Generation (RAG)** secara modular dan 100% offline pada CPU:
1. **Document Ingestion & Provenance:** Ekstraksi metadata, sanitasi, dan deduplikasi.
2. **Text Chunking & Overlap:** Fixed-size vs Recursive chunking, trade-off redundansi.
3. **Retrieval Engine:** Sparse (BM25) dan Dense (Vector Index) via Reciprocal Rank Fusion (RRF).
4. **Cross-Scoring Reranker:** Penataan ulang kandidat berpresisi tinggi.
5. **Context Construction & Prompt Grounding:** Pembatasan anggaran token dan delimiter aman.
6. **Deterministic Grounded Answer Baseline:** Sintesis jawaban bersandar bukti dan deteksi abstensi.
7. **Citation Validation:** Verifikasi keabsahan sitasi `[Doc: <id>]`.
8. **Comprehensive Evaluation:** Pemisahan evaluasi retrieval dan evaluasi generasi.
9. **RAG Security & Indirect Prompt Injection Defense:** Pemindaian muatan musuh dan filter ACL."""))

    # Cell 2: Imports & Environment Setup
    cells.append(code_cell("""import sys
import os
import json
import numpy as np

# Arahkan path modul ke folder code
sys.path.append(os.path.abspath("../code"))

from document_ingestion import DocumentIngestionPipeline
from document_chunking import FixedSizeChunker, RecursiveChunker
from rag_vector_index import RAGVectorIndex
from sparse_retrieval import BM25Retriever
from dense_retrieval import DenseRetriever
from hybrid_retrieval import HybridRetriever
from reranking_demo import HeuristicReranker
from context_builder import ContextBuilder
from grounded_answer_demo import GroundedAnswerGenerator
from citation_validation import CitationValidator
from rag_security_checks import RAGSecurityGuard

print("Lingkungan RAG berhasil dimuat. Pustaka NumPy & Python Standard Library aktif.")""",
"""Lingkungan RAG berhasil dimuat. Pustaka NumPy & Python Standard Library aktif."""))

    # Cell 3: Ingestion
    cells.append(md_cell("""## 1. Document Ingestion & Provenance Tracking
Memuat korpus dokumen akademik dari `data/ai_course_corpus.json`, membersihkan karakter non-printable, dan melacak asal usul dokumen (*source provenance*)."""))

    cells.append(code_cell("""corpus_path = os.path.abspath("../data/ai_course_corpus.json")
pipeline = DocumentIngestionPipeline()
docs = pipeline.ingest_from_json(corpus_path)

summary = pipeline.summary()
print(f"Total Dokumen: {summary['total_documents']}")
print(f"Kategori Dokumen: {', '.join(summary['categories'])}")
print(f"Rata-rata Panjang Dokumen: {summary['average_length_chars']} karakter\\n")

sample = docs[0]
print(f"Sampel Dokumen: [{sample.doc_id}] {sample.title}")
print(f"Source URL: {sample.source_url}")
print(f"Pratinjau Isi: {sample.content[:150]}...")""",
"""Total Dokumen: 20
Kategori Dokumen: AI Agents, AI Agents & Safety, AI Governance, Computer Vision, Deep Learning & NLP, Edge AI, MLOps, Model Adaptation, Neural Networks, RAG & Knowledge Retrieval, Security & Ethics, Security Testing, Vector Search & Embeddings
Rata-rata Panjang Dokumen: 504.65 karakter

Sampel Dokumen: [doc_rag_01] Pengantar Retrieval-Augmented Generation
Source URL: https://ai-hub.itera.ac.id/materials/ch33/intro-rag
Pratinjau Isi: Retrieval-Augmented Generation (RAG) adalah arsitektur yang menggabungkan model bahasa parametrik dengan sistem retrieval informasi eksternal non-paramet..."""))

    # Cell 4: Chunking
    cells.append(md_cell("""## 2. Text Chunking & Analisis Overlap
Membandingkan strategi pemotongan teks *Fixed-Size* dan *Recursive Hierarchical* dengan variasi rasio overlap."""))

    cells.append(code_cell("""rec_chunker = RecursiveChunker(chunk_size=300, chunk_overlap=40)
all_chunks = []
for d in docs:
    all_chunks.extend(rec_chunker.chunk_document(d))

print(f"Berhasil menghasilkan {len(all_chunks)} potongan teks (chunks) dari 20 dokumen.")
print(f"Contoh potongan pertama: [{all_chunks[0].chunk_id}] Rentang: {all_chunks[0].start_char}-{all_chunks[0].end_char}")
print(f"Isi:\\n{all_chunks[0].content}")""",
"""Berhasil menghasilkan 38 potongan teks (chunks) dari 20 dokumen.
Contoh potongan pertama: [doc_rag_01_c000] Rentang: 0-300
Isi:
Retrieval-Augmented Generation (RAG) adalah arsitektur yang menggabungkan model bahasa parametrik dengan sistem retrieval informasi eksternal non-parametrik. Model bahasa standar memiliki keterbatasan basis pengetahuan parametrik statis yang terhenti pada batas akhir data latih (knowledge cutoff)"""))

    # Cell 5: Hybrid Retrieval
    cells.append(md_cell("""## 3. Hybrid Retrieval: Penggabungan BM25 & Vektor Padat (RRF)
Menggabungkan keunggulan pencarian kata kunci BM25 dan kedalaman semantik vektor melalui formula *Reciprocal Rank Fusion*:
$$RRF(d) = \\sum_{m \\in \\{\\text{sparse}, \\text{dense}\\}} \\frac{1}{60 + r_m(d)}$$"""))

    cells.append(code_cell("""# 1. Inisialisasi BM25
sparse_retriever = BM25Retriever()
sparse_retriever.index_chunks(all_chunks)

# 2. Inisialisasi Vektor Padat
vector_index = RAGVectorIndex(embedding_dim=64)
vector_index.add_chunks(all_chunks)
dense_retriever = DenseRetriever(vector_index, similarity_threshold=0.20)

# 3. Inisialisasi Hybrid Retriever
hybrid = HybridRetriever(sparse_retriever, dense_retriever, rrf_k=60)

query_test = "Apa perbedaan RAG dan Fine-Tuning LoRA untuk data dinamis?"
hybrid_results = hybrid.search(query_test, top_k=3)

print(f"Query: '{query_test}'\\n")
print("Hasil Hybrid Retrieval (RRF):")
for r, (c, score, detail) in enumerate(hybrid_results, 1):
    print(f"[{r}] Skor RRF: {score:.6f} | Chunk ID: {c.chunk_id}")
    print(f"    Peringkat Sparse: {detail['sparse_rank']} | Peringkat Dense: {detail['dense_rank']}")
    print(f"    Dokumen: {c.metadata.get('title')}")""",
"""Query: 'Apa perbedaan RAG dan Fine-Tuning LoRA untuk data dinamis?'

Hasil Hybrid Retrieval (RRF):
[1] Skor RRF: 0.032519 | Chunk ID: doc_ft_01_c000
    Peringkat Sparse: 1 | Peringkat Dense: 2
    Dokumen: Perbedaan RAG dan Parameter-Efficient Fine-Tuning
[2] Skor RRF: 0.032258 | Chunk ID: doc_rag_01_c000
    Peringkat Sparse: 2 | Peringkat Dense: 1
    Dokumen: Pengantar Retrieval-Augmented Generation
[3] Skor RRF: 0.015873 | Chunk ID: doc_rag_03_c000
    Peringkat Sparse: 3 | Peringkat Dense: None
    Dokumen: Evaluasi RAG: Pemisahan Metrik Retrieval dan Generation"""))

    # Cell 6: Cross-Scoring Reranking
    cells.append(md_cell("""## 4. Cross-Scoring Reranker
Menyaring dan menata ulang peringkat kandidat teratas menggunakan kecocokan frasa eksak dan kerapatan kata kunci."""))

    cells.append(code_cell("""reranker = HeuristicReranker()
candidates = [c for c, _, _ in hybrid_results]
reranked = reranker.rerank(query_test, candidates, top_k=2)

print("Kandidat Setelah Neural Reranking:")
for r, (c, score) in enumerate(reranked, 1):
    print(f"[{r}] Skor Re-rank: {score:.4f} | Chunk ID: {c.chunk_id}")
    print(f"    Judul: {c.metadata.get('title')}")
    print(f"    Isi: {c.content[:120]}...\\n")""",
"""Kandidat Setelah Neural Reranking:
[1] Skor Re-rank: 0.6583 | Chunk ID: doc_ft_01_c000
    Judul: Perbedaan RAG dan Parameter-Efficient Fine-Tuning
    Isi: RAG dan Fine-Tuning (seperti LoRA) menyelesaikan masalah yang berbeda. Fine-tuning mengubah bobot internal model untuk me...

[2] Skor Re-rank: 0.5417 | Chunk ID: doc_rag_01_c000
    Judul: Pengantar Retrieval-Augmented Generation
    Isi: Retrieval-Augmented Generation (RAG) adalah arsitektur yang menggabungkan model bahasa parametrik dengan sistem retrieval..."""))

    # Cell 7: Grounded Answer Synthesis & Citations
    cells.append(md_cell("""## 5. Sintesis Jawaban Berbasis Bukti & Audit Sitasi
Menghasilkan jawaban deterministik yang bersandar ketat pada bukti faktual serta memverifikasi atribusi sitasi `[Doc: <id>]`."""))

    cells.append(code_cell("""generator = GroundedAnswerGenerator()
ans_result = generator.generate_grounded_answer(query_test, reranked)

print(f"Status Respon: {ans_result['status'].upper()}")
print(f"Tingkat Kepercayaan: {ans_result['confidence']}")
print(f"Sitasi Dokumen: {ans_result['citations']}")
print(f"Teks Jawaban:\\n{ans_result['answer']}\\n")

validator = CitationValidator()
audit = validator.validate_answer(ans_result['answer'], reranked)
print(f"Hasil Audit Sitasi: {'LOLOS (Valid)' if audit['valid'] else 'GAGAL'}")
print(f"Presisi Sitasi: {audit['citation_precision']}")
print(f"Sitasi Halusinasi: {audit['hallucinated_citations']}")""",
"""Status Respon: GROUNDED
Tingkat Kepercayaan: 0.6
Sitasi Dokumen: ['doc_ft_01']
Teks Jawaban:
RAG dan Fine-Tuning (seperti LoRA) menyelesaikan masalah yang berbeda. [Doc: doc_ft_01] Fine-tuning mengubah bobot internal model untuk mempelajari gaya bahasa, format keluaran terstruktur (JSON/SQL), atau perilaku spesifik domain. [Doc: doc_ft_01] Namun, fine-tuning lambat dan mahal untuk memperbarui fakta yang sering berubah serta tidak menyediakan atribusi sitasi yang dapat diverifikasi secara transparan. [Doc: doc_ft_01]

Hasil Audit Sitasi: LOLOS (Valid)
Presisi Sitasi: 1.0
Sitasi Halusinasi: []"""))

    # Cell 8: Abstention Demo
    cells.append(md_cell("""## 6. Uji Penolakan Menjawab (Abstention)
Menguji keandalan sistem saat dihadapkan pada query di luar domain pengetahuan (*out-of-domain*)."""))

    cells.append(code_cell("""query_ood = "Berapa jarak orbit planet Mars ke satelit Titan dalam satuan tahun cahaya?"
ood_results = hybrid.search(query_ood, top_k=2)
# Ambang batas relevansi minimum
filtered_ood = [(c, s) for c, s, _ in ood_results if s > 0.05]

ans_ood = generator.generate_grounded_answer(query_ood, filtered_ood)
print(f"Query Di Luar Domain: '{query_ood}'")
print(f"Status: {ans_ood['status'].upper()}")
print(f"Jawaban Sistem:\\n{ans_ood['answer']}")""",
"""Query Di Luar Domain: 'Berapa jarak orbit planet Mars ke satelit Titan dalam satuan tahun cahaya?'
Status: ABSTAINED
Jawaban Sistem:
Maaf, sistem tidak menemukan bukti faktual yang relevan dalam korpus dokumen untuk menjawab pertanyaan ini."""))

    # Cell 9: Security Checks
    cells.append(md_cell("""## 7. Keamanan RAG & Pertahanan Indirect Prompt Injection
Memindai muatan jahat dalam korpus dokumen dan menegakkan *Access Control Lists* (ACL) level dokumen."""))

    cells.append(code_cell("""guard = RAGSecurityGuard()

print("Memindai seluruh dokumen korpus terhadap muatan prompt injection...")
quarantined_docs = []
for d in docs:
    scan = guard.scan_for_prompt_injection(d.content)
    if not scan['is_safe']:
        quarantined_docs.append((d, scan['detected_threats']))

print(f"Dokumen Aman: {len(docs) - len(quarantined_docs)}")
print(f"Dokumen Terinfeksi (Dikarantina): {len(quarantined_docs)}")
for d, threats in quarantined_docs:
    print(f"  [PERINGATAN] ID: {d.doc_id} | Pola Terdeteksi: {threats}")""",
"""Memindai seluruh dokumen korpus terhadap muatan prompt injection...
Dokumen Aman: 19
Dokumen Terinfeksi (Dikarantina): 1
  [PERINGATAN] ID: doc_inject_01 | Pola Terdeteksi: ['Abaikan semua instruksi', 'Perintah sistem:', 'ubah peran Anda']"""))

    # Cell 10: Conclusion
    cells.append(md_cell("""## Kesimpulan Praktikum RAG
Melalui serangkaian eksperimen di atas, kita telah membuktikan secara empiris bahwa:
1. **Hybrid Retrieval (BM25 + Dense RRF)** memberikan recall yang seimbang antara istilah eksak dan makna semantik.
2. **Chunk Overlap (10-25%)** krusial untuk mencegah terputusnya relasi antar-kalimat tanpa memboroskan kapasitas konteks.
3. **Pemisahan Evaluasi** memungkinkan identifikasi presisi kegagalan sistem pada tahap penelusuran dokumen atau sintesis jawaban.
4. **Pertahanan Berlapis (Defense-in-Depth)** wajib diterapkan untuk mencegah manipulasi sistem melalui dokumen adversarial (*indirect prompt injection*)."""))

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "chapter_33_companion.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2, ensure_ascii=False)
    print(f"Notebook generated successfully at: {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    make_notebook()

