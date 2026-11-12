"""
Chapter 20 - Machine Learning Capstone
Script 4: Cross-Validation & Hyperparameter Tuning
Mengoptimalkan Random Forest dan Gradient Boosting dengan GridSearchCV
dan validasi 5-Fold Stratified Cross-Validation pada data latih.
"""

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from preprocessing_pipeline import build_preprocessing_pipeline

DATA_PATH = Path(__file__).parent / "data" / "predictive_maintenance_telemetry.csv"

def run_tuning():
    print("=" * 80)
    print("CAPSTONE TAHAP 4: HYPERPARAMETER TUNING & MODEL SELECTION")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "vibration_rms_g", "bearing_temperature_c", "rotational_speed_rpm",
        "hydraulic_pressure_bar", "motor_current_amp", "oil_acidity_tan"
    ]
    X = df[feature_cols].values
    y = df['machine_failure'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Tuning Random Forest
    rf_pipeline = Pipeline([
        ('prep', build_preprocessing_pipeline()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    rf_param_grid = {
        'rf__n_estimators': [50, 100, 150],
        'rf__max_depth': [5, 8, 12, None],
        'rf__min_samples_split': [2, 5, 10]
    }

    print("\n[1/2] Melakukan GridSearchCV untuk Random Forest Classifier...")
    rf_grid = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_train, y_train)

    print(f"  Best RF Params : {rf_grid.best_params_}")
    print(f"  Best RF F1-Score (5-Fold CV): {rf_grid.best_score_:.4f}")

    # 2. Tuning Gradient Boosting
    gb_pipeline = Pipeline([
        ('prep', build_preprocessing_pipeline()),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])

    gb_param_grid = {
        'gb__n_estimators': [50, 100, 150],
        'gb__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'gb__max_depth': [3, 5, 7]
    }

    print("\n[2/2] Melakukan GridSearchCV untuk Gradient Boosting Classifier...")
    gb_grid = GridSearchCV(
        estimator=gb_pipeline,
        param_grid=gb_param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    gb_grid.fit(X_train, y_train)

    print(f"  Best GB Params : {gb_grid.best_params_}")
    print(f"  Best GB F1-Score (5-Fold CV): {gb_grid.best_score_:.4f}")

    champion = "Random Forest" if rf_grid.best_score_ >= gb_grid.best_score_ else "Gradient Boosting"
    best_estimator = rf_grid.best_estimator_ if champion == "Random Forest" else gb_grid.best_estimator_

    print("\n" + "=" * 80)
    print(f"CHAMPION MODEL TERPILIH: {champion}")
    print(f"Cross-Validation F1-Score: {max(rf_grid.best_score_, gb_grid.best_score_):.4f}")
    print("=" * 80)

    return champion, best_estimator

if __name__ == "__main__":
    run_tuning()
