# Track D — AI for Telecommunications: Applied Laboratories

Delapan laboratorium terapan dari pemrosesan sinyal/telemetri jaringan radio hingga deteksi anomali, simulasi RL, dan pemantauan drift.

---

## 1. Lab Domain Klasifikasi Kualitas Sinyal Nirkabel (Run Langsung)
```bash
python content/tracks/telecommunications/labs/wireless_quality_lab.py
```
- **Output Terverifikasi:** Menegakkan pemisahan `SNR` (`Mean 15.79 dB`) vs `SINR` (`Mean 10.55 dB`) di antara 7 fitur autentik (`RSRP`, `RSRQ`, `SNR`, `SINR`, `Latency`, `Throughput`, `Packet Loss`), mencatat akurasi `98.50%` (`RandomForest`) vs baseline `81.50%`, dengan penafian dataset sintetis.

---

## 2. Daftar Delapan Laboratorium Terintegrasi

1. **Lab 1 — Wireless Network Dataset Analysis (Track D & Ch 08):** `wireless_quality_lab.py` & `exploratory_analysis.py`.
2. **Lab 2 — Signal Quality Classification (Track D & Ch 14):** `wireless_quality_lab.py`.
3. **Lab 3 — Network Performance Prediction (Chapter 13):** `regression_metrics.py`.
4. **Lab 4 — Network Anomaly Detection (Chapter 17):** `dbscan_clustering.py`.
5. **Lab 5 — Network Fault Classification (Chapter 15):** `random_forest_classifier.py`.
6. **Lab 6 — Time-Series Drift Monitoring (Chapter 36):** `concept_drift_simulation.py`.
7. **Lab 7 — RL Resource Allocation Simulation (Chapter 35):** `mdp_transitions.py`.
8. **Lab 8 — Model Evaluation & Explainability (Chapter 19 & 36):** `permutation_importance.py` & `data_drift_detection.py`.

