# Chapter 40: Advanced AI Capstone Project — Reference Implementation

Implementasi proyek referensi end-to-end berbasis CPU lokal: **Intelligent Sensor Monitoring with Machine Learning, Evaluation & Responsible AI**.

## Struktur Modul Python (`code/`)

1. `config.py`: Konfigurasi terpusat dan deterministik (`seed=42`, rasio split `60/20/20`, ambang *Human Escalation*, dan batas valid fitur sensor).
2. `dataset_generator.py`: Generator telemetri motor sintetis 6-fitur + kamus data lengkap `data/sensor_data_dictionary.json`.
3. `preprocessing.py`: Pembagian dataset berstrata **zero data leakage** (`StandardScaler` di-*fit* murni pada training fold).
4. `baseline_model.py`: Pelatihan model baseline sederhana (`DummyClassifier` & `LogisticRegression`).
5. `candidate_models.py`: Perbandingan kandidat (`LogisticRegression`, `RandomForest`, `MLPClassifier`), profiling ukuran `.pkl` & latensi CPU (Mean/P50/P95), serta penyimpanan checksum **SHA-256**.
6. `evaluation_and_error_analysis.py`: Evaluasi akhir pada held-out test set, matriks kebingungan, *permutation feature importance*, dan audit contoh kesalahan prediksi (`experiments/capstone_evaluation_chart.png`).
7. `local_inference.py`: Inferensi lokal tervalidasi dengan penolakan `NaN`/`Inf`/boolean/nilai ekstrem serta *Human Escalation Fallback*.
8. `monitoring_and_drift.py`: Simulasi pasca-deployment dengan penghitungan **PSI** & uji **Kolmogorov-Smirnov** (`experiments/capstone_drift_monitoring.png`).
9. `responsible_ai_assessment.py`: Risk Register NIST AI RMF dan generator Model Card (`experiments/capstone_model_card.json`).
10. `run_capstone_pipeline.py`: Eksekutor terpadu end-to-end menyimpan seluruh metrik ke `experiments/capstone_report_metrics.json`.
11. `test_capstone_suite.py`: Suite pengujian unit (`unittest`) memverifikasi determinisme, zero leakage, penjagaan skema, dan sensitivitas PSI.

## Cara Menjalankan

```bash
python content/level-4-advanced-ai/chapter-40/code/run_capstone_pipeline.py
python content/level-4-advanced-ai/chapter-40/code/test_capstone_suite.py
```

