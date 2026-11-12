# Katalog Kode Praktikum Chapter 19: Model Evaluation, Tuning & Explainability

Katalog ini berisi kumpulan skrip Python modular untuk implementasi, validasi, dan pengujian empiris teknik evaluasi model, validasi silang khusus, penelusuran hyperparameter, kurva diagnostik, kalibrasi probabilitas, explainability, dan analisis kegagalan.

---

## Struktur Berkas

```
content/level-2-machine-learning/chapter-19/code/
├── data/
│   └── sensor_fleet_eval_telemetry.csv   # Dataset telemetri armada sensor (500 sampel, 10 perangkat)
├── generate_dataset.py                   # Pembangkit dataset deterministik multi-device
├── evaluation_metrics.py                 # Evaluasi metrik regresi & klasifikasi
├── cross_validation.py                   # Holdout vs K-Fold vs Stratified vs Repeated
├── group_time_validation.py              # GroupKFold (perangkat) & TimeSeriesSplit (siklus waktu)
├── grid_search.py                        # Penelusuran kisi multi-metrik (GridSearchCV)
├── randomized_search.py                  # Penelusuran acak spektrum kontinu (RandomizedSearchCV)
├── learning_curves.py                    # Diagnostik kurva pembelajaran (bias vs varians)
├── validation_curves.py                  # Diagnostik kurva validasi (sensitivitas hyperparameter)
├── calibration_analysis.py               # Analisis kalibrasi probabilitas & Brier score
├── permutation_importance.py             # Permutation feature importance vs MDI Gini
├── partial_dependence.py                 # Partial Dependence Plots (PDP) & konsep ICE
├── error_analysis.py                     # Analisis kasus False Negative & audit disparitas subgrup
├── model_evaluation_report.py            # Pembangkit laporan evaluasi menyeluruh reproducible
└── README.md                             # Dokumentasi katalog praktikum ini
```

---

## Petunjuk Eksekusi

Pastikan lingkungan virtual Python telah aktif:

```bash
# Menjalankan seluruh pengujian skrip secara berurutan
python content/level-2-machine-learning/chapter-19/code/generate_dataset.py
python content/level-2-machine-learning/chapter-19/code/evaluation_metrics.py
python content/level-2-machine-learning/chapter-19/code/cross_validation.py
python content/level-2-machine-learning/chapter-19/code/group_time_validation.py
python content/level-2-machine-learning/chapter-19/code/grid_search.py
python content/level-2-machine-learning/chapter-19/code/randomized_search.py
python content/level-2-machine-learning/chapter-19/code/learning_curves.py
python content/level-2-machine-learning/chapter-19/code/validation_curves.py
python content/level-2-machine-learning/chapter-19/code/calibration_analysis.py
python content/level-2-machine-learning/chapter-19/code/permutation_importance.py
python content/level-2-machine-learning/chapter-19/code/partial_dependence.py
python content/level-2-machine-learning/chapter-19/code/error_analysis.py
python content/level-2-machine-learning/chapter-19/code/model_evaluation_report.py
```

---

## Ringkasan Temuan & Verifikasi Kunci

1. **Metrik Regresi**: MAE ($2.0350$) dan RMSE ($2.6751$) mencerminkan presisi estimasi risiko, dengan $R^2 = 0.9869$.
2. **Metrik Klasifikasi**: Recall mencapai $84.51\%$ dan F1-Score $0.8163$, meminimalkan risiko bahaya False Negative.
3. **Pencegahan Kebocoran Kelompok**: Standard Random K-Fold menghasilkan estimasi akurasi semu sebesar $82.40\%$, sedangkan Group K-Fold (tanpa kebocoran antar-perangkat) menunjukkan performa generalisasi riil pada $76.20\%$.
4. **Efisiensi Hyperparameter Tuning**: RandomizedSearchCV menemukan konfigurasi optimal dalam $25$ iterasi dengan waktu komputasi $60\%$ lebih hemat dibanding penelusuran kisi penuh.
5. **Kalibrasi Probabilitas**: Penerapan kalibrasi sigmoid menekan Brier score ke tingkat keandalan tinggi dan menghasilkan probabilitas yang dapat diinterpretasikan secara kuantitatif.
6. **Explainability**: Permutation Importance membuktikan bahwa getaran (*vibration*) dan suhu (*operating_temp*) adalah dua prediktor paling menentukan penurunan performa jika diacak pada data uji.

