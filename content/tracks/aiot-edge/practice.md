# Track C — AIoT & Edge AI: Applied Laboratories

Delapan laboratorium berurutan dari analitik telemetri sensor hingga kompresi model INT8 dan pemantauan drift di CPU lokal.

---

## 1. Lab Terapan Kompresi Model Sensor Edge (Run Langsung)
```bash
python content/tracks/aiot-edge/labs/aiot_sensor_edge_lab.py
```
- **Output Terverifikasi:** Menunjukkan penyusutan ukuran state dict dari `13.27 KB` (FP32) menjadi `7.50 KB` (Dynamic INT8) pada akurasi uji `96.00%` dan latensi `0.3005 ms`, disertai penafian simulasi CPU (bukan pengukuran hardware fisik ESP32).

---

## 2. Daftar Delapan Laboratorium Terintegrasi

1. **Lab 1 — Sensor Telemetry Analysis (Chapter 08):** `exploratory_analysis.py`.
2. **Lab 2 — Feature Engineering (Chapter 12):** `feature_engineering.py`.
3. **Lab 3 — Fault Classification (Chapter 14):** `classification_model_comparison.py`.
4. **Lab 4 — Anomaly/Pattern Clustering (Chapter 17):** `clustering_comparison.py`.
5. **Lab 5 — Model Compression Pruning (Chapter 37):** `structured_pruning_demo.py`.
6. **Lab 6 — Quantization (Chapter 37 & Track C):** `dynamic_quantization.py` & `aiot_sensor_edge_lab.py`.
7. **Lab 7 — CPU Inference Benchmarking (Chapter 37):** `inference_latency_benchmark.py`.
8. **Lab 8 — Drift Monitoring (Chapter 36):** `data_drift_detection.py`.

