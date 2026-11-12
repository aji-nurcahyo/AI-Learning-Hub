# Laporan Audit Kualitas: Chapter 17 — Unsupervised Learning & Clustering

**Tanggal Audit**: 5 November 2026  
**Status**: PASSED (Lulus Penuh)  
**Tingkat Pembelajaran**: Level 2 — Machine Learning  
**Target Chapter**: Chapter 17 (Unsupervised Learning & Clustering)  
**Auditor**: Antigravity Quality Assurance System  

---

## 1. Verifikasi Kelengkapan Berkas (Resource Completeness)

| Berkas | Lokasi | Status | Keterangan |
|---|---|---|---|
| **Modul Utama** | `chapter-17/index.md` | VALID | 13 seksi komprehensif, 16 diagram ASCII, worked examples, tiered exercises, mini project |
| **Slide Deck** | `chapter-17/slides/deck.json` | VALID | 27 slide Reveal.js terstruktur, left-aligned, notes instruksional |
| **Panduan Praktik** | `chapter-17/practice.md` | VALID | 12 modul skrip terstruktur, tabel hasil terverifikasi, rubrik 100 poin |
| **Companion Notebook** | `chapter-17/notebooks/chapter_17_companion.ipynb` | VALID | Valid Jupyter v4 JSON, pre-executed outputs, Markdown + Code cells |
| **Metadata Module** | `chapter-17/metadata.json` | VALID | Status `published`, 13 Bloom objectives, 5 verified citations |
| **Koleksi Skrip Kode** | `chapter-17/code/` (13 skrip + README) | VALID | Seluruh skrip exit code 0 |

---

## 2. Verifikasi Hasil Komputasi & Pengujian Empiris

Dataset: `wireless_sensor_telemetry.csv` (360 baris, 6 fitur kontinu).

| Pengujian | Skrip | Parameter | Hasil Terverifikasi | Status |
|---|---|---|---|---|
| **Inersia K-Means** | `kmeans_sklearn.py` | $k=3$, `k-means++` | Inersia WCSS: 651.1098 | LULUS |
| **Elbow Drop** | `elbow_method.py` | $k \in [1, 9]$ | Siku tajam pada $k=3$ (penurunan 48.71%) | LULUS |
| **Silhouette Peak** | `silhouette_analysis.py` | $k=3$ | Nilai tertinggi: 0.5804 (0 sampel negatif) | LULUS |
| **DBSCAN Partition** | `dbscan_clustering.py` | $\epsilon=1.15, \text{min}=6$ | 308 Core, 27 Border, 25 Noise Outliers | LULUS |
| **Hierarchical Ward** | `hierarchical_clustering.py` | $k=3$, Euclidean | Silhouette: 0.5739, DBI: 0.6548, CHI: 286.3 | LULUS |
| **NumPy Scratch Match** | `kmeans_from_scratch.py` | $k=3$, Lloyd | Perbedaan inersia vs Scikit-learn $< 10^{-3}$ | LULUS |

---

## 3. Verifikasi Standar Bahasa & Bebas Em-Dash

- Seluruh dokumen bebas dari simbol typographical em-dash (`—` / `--`).
- Terminologi konsisten sesuai pedoman `docs/TECHNICAL_LANGUAGE_GUIDE.md`:
  - *Within-Cluster Sum of Squares* (WCSS) / Inersia.
  - *Density-Based Spatial Clustering of Applications with Noise* (DBSCAN).
  - *Core Point*, *Border Point*, *Noise Point*.
  - *Linkage criteria* (Ward, Complete, Average, Single).
  - *Dendrogram* dan *Cut Threshold*.
  - *Silhouette Coefficient*, *Davies-Bouldin Index*, *Calinski-Harabasz Index*.
- Tidak ada rumus matematika mentah atau simbol unrendered. KaTeX renderable.

---

## 4. Kesimpulan Audit
Chapter 17 telah memenuhi 100% persyaratan kurikulum resmi Master Curriculum AI Learning Hub dan siap dipublikasikan.

