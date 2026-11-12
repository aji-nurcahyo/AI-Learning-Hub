"""
Script 10: Scikit-learn Pipeline Construction
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan pembuatan end-to-end Pipeline menggunakan Scikit-learn:
1. Menggabungkan tahapan imputasi, penskalaan, dan estimasi ke dalam satu objek tunggal.
2. Memastikan transformasi otomatis fit-transform saat fit() dan hanya transform saat predict().
3. Menjamin keterulangan (reproducibility) dan mencegah resiko kebocoran data.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI SCIKIT-LEARN PIPELINE")
    print("=" * 75)
    
    df = load_telemetry_data()
    feature_cols = ["operating_hours", "vibration_rms", "temperature_c", "pressure_kpa"]
    X = df[feature_cols]
    y = df["failure"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Membangun Pipeline terstruktur
    ml_pipe = Pipeline([
        ("median_imputer", SimpleImputer(strategy="median")),
        ("standard_scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42))
    ])
    
    print("Struktur Pipeline:")
    for step_name, step_obj in ml_pipe.named_steps.items():
        print(f"  -> [{step_name}]: {step_obj}")
        
    print("\nMelakukan training pipeline (pipe.fit)...")
    ml_pipe.fit(X_train, y_train)
    print("Training berhasil!\n")
    
    print("Parameter internal yang dipelajari dalam pipeline:")
    print(f"  * Imputer medians : {ml_pipe.named_steps['median_imputer'].statistics_}")
    print(f"  * Scaler mean     : {ml_pipe.named_steps['standard_scaler'].mean_}")
    print(f"  * Model weights   : {ml_pipe.named_steps['classifier'].coef_[0]}")
    
    # Inferensi langsung dari data mentah
    print("\nEvaluasi pada Test Set (Data Mentah langsung masuk pipe.predict):")
    y_pred = ml_pipe.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))
    
    # Simulasi data sampel baru dari sensor
    sample_raw = pd.DataFrame([{
        "operating_hours": 4200.0,
        "vibration_rms": None,  # Missing value
        "temperature_c": 82.5,  # Panas
        "pressure_kpa": 340.0
    }])
    pred_sample = ml_pipe.predict(sample_raw)[0]
    prob_sample = ml_pipe.predict_proba(sample_raw)[0, 1]
    
    print("Uji Prediksi Sampel Sensor Baru (Mengandung NaN):")
    print(f"  * Prediksi Status  : {'FAIL' if pred_sample == 1 else 'NORMAL'}")
    print(f"  * Probabilitas Fail: {prob_sample:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

