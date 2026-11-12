# Chapter 36: MLOps, Model Deployment & Monitoring — Code Catalog

Koleksi 17 script Python mandiri, deterministik, dan berjalan 100% lokal pada CPU untuk menguasai rekayasa MLOps tanpa dependensi Docker, Kubernetes, maupun layanan cloud berbayar.

## Daftar Script

1. `mlops_lifecycle.py`: Kontrak tahapan arsitektural siklus hidup MLOps dan tata kelola gerbang validasi (*governance gates*).
2. `reproducible_training.py`: Pipeline pelatihan Scikit-learn bebas kebocoran data (*zero data leakage*) dengan pemecahan berstrata sebelum penskalaan, random seed tetap, serta pembuatan checksum SHA-256.
3. `experiment_tracking.py`: Pelacak eksperimen lokal berbasis SQLite (`mlops_tracker.db`) mencatat parameter, metrik, versi dataset, metadata lingkungan, dan path artefak.
4. `experiment_comparison.py`: Eksekusi perbandingan hiperparameter (*LogisticRegression* vs *RandomForest*) dan seleksi kandidat terbaik berdasarkan ROC-AUC/F1.
5. `model_versioning.py`: Pengelolaan *Semantic Versioning* (`MAJOR.MINOR.PATCH`) serta pencatatan silsilah model (*model lineage*).
6. `local_model_registry.py`: Registri model edukasional berbasis SQLite (`model_registry.db`) dengan status terkendali (`CANDIDATE`, `VALIDATED`, `STAGED_LOCAL_SERVING`, `ARCHIVED`) dan gerbang persetujuan manusia.
7. `model_artifact_validation.py`: Verifikasi kriptografis SHA-256 sebelum deserialisasi `pickle.load()` guna menangkal eksekusi kode berbahaya dari artefak yang dimodifikasi.
8. `request_validation.py`: Validator skema masukan JSON (*dimensionality check*, tipe numerik, penolakan boolean/NaN/Inf) sebelum inferensi model.
9. `model_serving_api.py`: Server REST API lokal ringan berbasis Python stdlib `http.server` yang mengikat eksklusif pada `127.0.0.1` dengan endpoint `GET /health`, `GET /model/info`, `POST /predict`, dan `GET /metrics`.
10. `api_client_demo.py`: Suite pengujian integrasi klien HTTP menguji validasi skema (HTTP 400), prediksi sukses (HTTP 200), dan metrik latensi, lalu menutup server secara bersih.
11. `data_drift_detection.py`: Deteksi *Data Drift* statistik menggunakan *Population Stability Index* (PSI) dan uji dua sampel *Kolmogorov–Smirnov* (KS Test) murni dengan NumPy.
12. `concept_drift_simulation.py`: Simulasi aliran data berlabel yang membedakan *Covariate Data Drift* (akurasi tetap tinggi) dari *Abrupt/Gradual Concept Drift* (akurasi merosot akibat perubahan $P(y \mid X)$).
13. `prediction_monitoring.py`: Pemantauan pergeseran distribusi kelas prediksi (*prediction drift*) dan pita ketidakpastian probabilitas sebelum label kebenaran tiba.
14. `inference_latency.py`: Pengukuran latensi inferensi CPU aktual (Mean, P50 Median, P95, P99 ms, dan Throughput req/s) di memori dan melalui `127.0.0.1`.
15. `monitoring_dashboard.py`: Agregasi telemetri pemantauan multi-jendela dan pembuatan grafik dashboard observability (`experiments/monitoring_dashboard.png`).
16. `model_rollback_simulation.py`: Simulasi peringatan pelanggaran SLA, pencegahan *rollback* tanpa izin, dan eksekusi *rollback* terkendali disetujui manusia ke versi `v1.0.0`.
17. `mlops_pipeline_demo.py`: Demonstrasi pipeline MLOps menyeluruh yang mengintegrasikan seluruh tahapan dari pelatihan hingga pemantauan drift.

## Cara Menjalankan

```bash
# Jalankan demonstrasi pipeline MLOps menyeluruh
python content/level-4-advanced-ai/chapter-36/code/mlops_pipeline_demo.py

# Jalankan pengujian integrasi REST API lokal (127.0.0.1)
python content/level-4-advanced-ai/chapter-36/code/api_client_demo.py
```

