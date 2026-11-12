"""Reproducible ML Training Pipeline with Zero Data Leakage.

Implements:
1. Synthetic deterministic dataset creation with explicit provenance hash.
2. Stratified train/test split prior to any scaling/preprocessing (preventing leakage).
3. Scikit-learn Pipeline wrapping StandardScaler + Classifier.
4. Trusted local artifact serialization with SHA-256 cryptographic checksum.
5. Environment metadata recording (Python, NumPy, Scikit-learn versions).
"""

import hashlib
import json
import os
import pickle
import platform
import sys
from typing import Any, Dict, Tuple

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_telemetry_dataset(n_samples: int = 1200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, str]:
    """Generates synthetic 6-feature tabular data with fixed seed and provenance hash."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        weights=[0.65, 0.35],
        class_sep=1.2,
        random_state=seed,
    )
    dataset_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()[:16]
    return X, y, f"telemetry_ds_{dataset_hash}"


def train_and_save_pipeline(
    model_type: str = "logistic_regression",
    C: float = 1.0,
    n_estimators: int = 50,
    seed: int = 42,
    output_dir: str = None,
    artifact_filename: str = "model_v1.pkl",
) -> Dict[str, Any]:
    """Trains a leak-free Scikit-learn Pipeline and saves artifact with checksum."""
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments"))
    os.makedirs(output_dir, exist_ok=True)

    X, y, dataset_version = generate_telemetry_dataset(n_samples=1200, seed=42)

    # STRICT ZERO-LEAKAGE: Split before any fit_transform!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    if model_type == "logistic_regression":
        clf = LogisticRegression(C=C, max_iter=500, random_state=seed)
        hyperparams = {"model_type": model_type, "C": C, "seed": seed}
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=6, random_state=seed)
        hyperparams = {"model_type": model_type, "n_estimators": n_estimators, "max_depth": 6, "seed": seed}
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }

    artifact_path = os.path.join(output_dir, artifact_filename)
    with open(artifact_path, "wb") as f:
        pickle.dump(pipeline, f)

    checksum = compute_file_sha256(artifact_path)

    # Also store reference split for drift monitoring scripts
    data_dir = os.path.abspath(os.path.join(output_dir, "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    ref_path = os.path.join(data_dir, "reference_split.npz")
    np.savez(ref_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    env_meta = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }

    return {
        "dataset_version": dataset_version,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "hyperparams": hyperparams,
        "metrics": metrics,
        "artifact_path": artifact_path,
        "checksum_sha256": checksum,
        "env_metadata": env_meta,
    }


def main():
    print("=" * 75)
    print("CHAPTER 36: REPRODUCIBLE ML TRAINING PIPELINE (ZERO LEAKAGE)")
    print("=" * 75)

    res = train_and_save_pipeline(model_type="logistic_regression", C=1.0, seed=42)
    print(f"Dataset Version : {res['dataset_version']} (Train: {res['n_train']}, Test: {res['n_test']})")
    print(f"Hyperparameters : {res['hyperparams']}")
    print(f"Test Metrics    : Accuracy={res['metrics']['accuracy']}, F1={res['metrics']['f1_score']}, AUC={res['metrics']['roc_auc']}")
    print(f"Artifact Saved  : {res['artifact_path']}")
    print(f"SHA-256 Checksum: {res['checksum_sha256']}")
    print(f"Environment     : {res['env_metadata']}")
    print("\n[SUCCESS] Reproducible leak-free training pipeline completed.")


if __name__ == "__main__":
    main()

