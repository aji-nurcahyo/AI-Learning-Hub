"""
Script 11: ColumnTransformer for Heterogeneous Data
Chapter 12: ML Workflow, Dataset & Feature Engineering

Mendemonstrasikan pra-pemrosesan data heterogen (bertipe campuran):
1. Numeric Pipeline: SimpleImputer (median) + StandardScaler.
2. Categorical Nominal Pipeline: OneHotEncoder(handle_unknown='ignore').
3. Categorical Ordinal Pipeline: OrdinalEncoder.
4. Penggabungan dengan ColumnTransformer dan integrasi ke Pipeline model utama.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score


def load_telemetry_data() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / "data" / "equipment_telemetry.csv"
    return pd.read_csv(data_path)


def main():
    print("=" * 75)
    print("DEMONSTRASI COLUMNTRANSFORMER UNTUK DATA HETEROGEN")
    print("=" * 75)
    
    df = load_telemetry_data()
    
    numeric_features = ["operating_hours", "vibration_rms", "temperature_c", "pressure_kpa"]
    nominal_features = ["cooling_status"]
    ordinal_features = ["warning_flag"]
    target_col = "failure"
    
    X = df[numeric_features + nominal_features + ordinal_features]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Data Latih  : {X_train.shape[0]} baris, {X_train.shape[1]} kolom mentah")
    print(f"Data Uji    : {X_test.shape[0]} baris, {X_test.shape[1]} kolom mentah\n")
    
    # 1. Sub-pipeline untuk Fitur Numerik
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # 2. Sub-pipeline untuk Fitur Nominal
    nom_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # 3. Sub-pipeline untuk Fitur Ordinal
    ord_pipeline = Pipeline([
        ("ordinal", OrdinalEncoder(categories=[["LOW", "MEDIUM", "HIGH"]], handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    # 4. ColumnTransformer Utama
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("nom", nom_pipeline, nominal_features),
            ("ord", ord_pipeline, ordinal_features)
        ],
        remainder="drop"
    )
    
    # 5. Full End-to-End Modeling Pipeline
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=42))
    ])
    
    print("Melatih End-to-End ColumnTransformer Pipeline...")
    full_pipeline.fit(X_train, y_train)
    print("Training berhasil!\n")
    
    # Evaluasi pada data uji mentah
    y_pred = full_pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_test, y_pred)
    
    print("Evaluasi Performa Test Set:")
    print(f"  * F1-Score (Failure)       : {f1:.4f}")
    print(f"  * Balanced Accuracy        : {bacc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))
    
    # Uji coba inferensi dengan data heterogen baru
    new_equipment_data = pd.DataFrame([{
        "operating_hours": 3800.0,
        "vibration_rms": 4.1,
        "temperature_c": 79.2,
        "pressure_kpa": 335.0,
        "cooling_status": "OFF",
        "warning_flag": "HIGH"
    }])
    
    pred = full_pipeline.predict(new_equipment_data)[0]
    prob = full_pipeline.predict_proba(new_equipment_data)[0, 1]
    
    print("Uji Prediksi Mesin Kritis:")
    print(f"  * Status Prediksi : {'BAHAYA / FAILURE' if pred == 1 else 'NORMAL'}")
    print(f"  * Skor Risiko     : {prob:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

