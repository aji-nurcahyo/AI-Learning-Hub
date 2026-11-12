# Chapter 38: Multimodal AI Systems — Code Catalog

Koleksi 16 script Python mandiri, deterministik, dan berjalan pada CPU lokal (PyTorch, NumPy, Matplotlib) tanpa ketergantungan GPU atau bobot CLIP raksasa eksternal.

## Daftar Script

1. `multimodal_foundations.py`: Ilustrasi mengapa *Multimodal AI* membutuhkan interaksi lintas modalitas dibanding menjalankan model unimodal terpisah.
2. `paired_dataset_generator.py`: Pembuat dataset sintetis citra–teks (32x32 RGB objek geometris, 4 warna $\times$ 4 bentuk = 16 konsep) dengan pemecahan strict *train/test split* dan metadata `data/paired_dataset_info.json`.
3. `image_encoder.py`: `TinyImageEncoder` berbasis CNN ringan yang menghasilkan peta patch spasial 4x4 (16 token) maupun vektor global *pooled* 64-d.
4. `text_encoder.py`: `TinyTextEncoder` berbasis *Token Embedding* + proyeksi yang menghasilkan sekuens token maupun vektor *masked mean-pooled*.
5. `early_fusion.py`: Arsitektur *Early Fusion* menggabungkan vektor citra & teks di tahap fitur awal (`concat`) + uji degradasi saat satu modalitas hilang.
6. `late_fusion.py`: Arsitektur *Late Fusion* melatih cabang unimodal independen dan menggabungkan probabilitas keputusan (*ensemble*) yang tangguh saat modalitas hilang.
7. `cross_attention_fusion.py`: Verifikasi aljabar paritas NumPy vs PyTorch untuk *Scaled Dot-Product Cross-Attention* ($Q=\text{Text Tokens}$, $K, V=\text{Image Patches}$) beserta pelatihan pengklasifikasi lengkap.
8. `multimodal_embeddings.py`: Lapisan proyeksi bersama + normalisasi L2 pada hipersfer satuan serta demonstrasi *Modality Gap* sebelum pelatihan kontrastif.
9. `contrastive_loss.py`: Perhitungan manual langkah demi langkah *Symmetric InfoNCE / CLIP Loss* pada matriks kesamaan $3\times 3$ yang diverifikasi identik dengan PyTorch `F.cross_entropy`.
10. `clip_miniature.py`: Pelatihan `MiniatureDualEncoder` bergaya arsitektur CLIP menggunakan *unique-concept batch sampling* (16 konsep per batch, nol *false negatives*) dan penyimpanan bobot `experiments/clip_miniature.pt`.
11. `similarity_matrix.py`: Penghitungan dan penyimpanan visualisasi matriks kesamaan kosinus $16\times 16$ (`experiments/similarity_matrix.png`).
12. `text_to_image_retrieval.py`: Penelusuran silang *Text-to-Image* dari kueri teks ke 96 gambar uji tak terlihat (*unseen test pool*).
13. `image_to_text_retrieval.py`: Penelusuran silang *Image-to-Text* dari gambar uji ke 16 teks deskripsi konsep kanonik.
14. `cross_modal_evaluation.py`: Evaluasi kuantitatif dua arah (`I->T` dan `T->I`) melaporkan `Recall@1`, `Recall@3`, `Recall@5`, `Precision@6`, dan `Median Rank` murni pada set uji tak terlihat.
15. `vqa_concept_demo.py`: Pipeline konseptual *Visual Question Answering* menggunakan inspektor atribut geometris–warna deterministik (berlabel eksplisit bukan neural generative VQA).
16. `multimodal_comparison.py`: Benchmark komparatif *Early Fusion* vs *Late Fusion* vs *Cross-Attention* pada kondisi lengkap maupun kehilangan modalitas (`experiments/multimodal_fusion_comparison.png`).

## Cara Menjalankan

```bash
# Latih Miniature Dual Encoder dan evaluasi retrieval silang
python content/level-4-advanced-ai/chapter-38/code/clip_miniature.py
python content/level-4-advanced-ai/chapter-38/code/cross_modal_evaluation.py

# Bandingkan strategi fusi di kondisi normal vs hilang modalitas
python content/level-4-advanced-ai/chapter-38/code/multimodal_comparison.py
```

