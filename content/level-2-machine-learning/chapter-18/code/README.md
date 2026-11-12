# Katalog Kode Praktikum Chapter 18: Dimensionality Reduction

Katalog ini berisi skrip Python modular untuk implementasi, verifikasi matematis, dan benchmark reduksi dimensi (*dimensionality reduction*).

---

## Struktur Berkas

```
content/level-2-machine-learning/chapter-18/code/
├── data/
│   └── wireless_sensor_telemetry.csv   # Dataset 360 sampel telemetri sensor (6 fitur)
├── high_dimensional_data.py           # Eksplorasi fenomena kutukan dimensi (Curse of Dimensionality)
├── pca_from_scratch.py                 # Implementasi PCA dari dasar (NumPy murni) vs Scikit-Learn
├── pca_sklearn.py                      # Ekstraksi bobot pemuatan (loadings) & interpretasi fisis
├── explained_variance.py               # Analisis scree plot data, kriteria Kaiser & ambang batas
├── pca_reconstruction.py               # Inverse transform & galat rekonstruksi (Norma Frobenius)
├── svd_decomposition.py                # Dekomposisi SVD penuh & Teorema Eckart-Young-Mirsky
├── truncated_svd.py                    # TruncatedSVD vs Standard PCA (uncentered vs centered)
├── pca_pipeline.py                     # Pipeline ML bebas kebocoran data (zero data leakage)
├── tsne_visualization.py               # Manifold learning t-SNE & tuning perplexity
├── umap_visualization.py               # Eksplorasi UMAP (penanganan gracefully fallback)
├── dimensionality_comparison.py        # Benchmark komprehensif seluruh metode reduksi dimensi
└── README.md                           # Dokumentasi katalog kode praktikum ini
```

---

## Petunjuk Eksekusi

Pastikan virtual environment telah aktif:

```bash
# Menjalankan seluruh pengujian skrip secara berurutan
python content/level-2-machine-learning/chapter-18/code/high_dimensional_data.py
python content/level-2-machine-learning/chapter-18/code/pca_from_scratch.py
python content/level-2-machine-learning/chapter-18/code/pca_sklearn.py
python content/level-2-machine-learning/chapter-18/code/explained_variance.py
python content/level-2-machine-learning/chapter-18/code/pca_reconstruction.py
python content/level-2-machine-learning/chapter-18/code/svd_decomposition.py
python content/level-2-machine-learning/chapter-18/code/truncated_svd.py
python content/level-2-machine-learning/chapter-18/code/pca_pipeline.py
python content/level-2-machine-learning/chapter-18/code/tsne_visualization.py
python content/level-2-machine-learning/chapter-18/code/umap_visualization.py
python content/level-2-machine-learning/chapter-18/code/dimensionality_comparison.py
```

---

## Hasil Pengujian & Temuan Kunci

1. **Konsentrasi Jarak**: Pada dimensi $D=1000$, rasio jarak minimum terhadap jarak maksimum melonjak drastis hingga **$0.8682$**, membuktikan hilangnya ketajaman metrik jarak di ruang berdimensi tinggi.
2. **PCA Presisi Mutlak**: Implementasi PCA dari dasar berbasis NumPy menghasilkan selisih komponen dan proyeksi sebesar **$< 10^{-15}$** terhadap Scikit-Learn.
3. **Rasio Varians Terjelaskan**:
   - PC1 merangkum **$80.51\%$** varians total (indeks degradasi global).
   - PC2 merangkum **$6.95\%$** varians total.
   - Proyeksi 2D mempertahankan **$87.45\%$** informasi total dataset.
4. **Batas Teoretis SVD**: Galat aproksimasi rank-$k$ empiris persis menyentuh batas teoretis Teorema Eckart-Young-Mirsky ($\sqrt{\sum_{j=k+1}^D s_j^2}$).
5. **Efisiensi Pipeline**: Reduksi dari 6 fitur ke 2 komponen utama mempertahankan akurasi klasifikasi prima (**$96.67\%$**) dengan efisiensi komputasi lebih tinggi.

