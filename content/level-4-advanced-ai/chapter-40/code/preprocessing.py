"""Zero-Leakage Stratified Dataset Splitting & StandardScaler Preprocessing."""

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import CapstoneConfig
from dataset_generator import generate_sensor_telemetry


@dataclass
class PreparedTelemetryData:
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def prepare_splits(cfg: CapstoneConfig = CapstoneConfig()) -> PreparedTelemetryData:
    X, y = generate_sensor_telemetry(cfg.n_samples, cfg.seed)

    # First carve out held-out test set (never touched during model selection)
    X_dev, X_test_raw, y_dev, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    # Split remaining development pool into train and validation
    val_rel = cfg.val_size / (1.0 - cfg.test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_rel, random_state=cfg.seed, stratify=y_dev
    )

    # Fit scaler STRICTLY on training fold only (Zero Data Leakage!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    assert abs(X_train.mean()) < 1e-9
    return PreparedTelemetryData(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
    )


def main():
    data = prepare_splits()
    print("=" * 75)
    print("CHAPTER 40 CAPSTONE: ZERO-LEAKAGE DATA PREPARATION")
    print("=" * 75)
    print(f"Train Shape : {data.X_train.shape} (Fault Rate={100*data.y_train.mean():.2f}%)")
    print(f"Val Shape   : {data.X_val.shape} (Fault Rate={100*data.y_val.mean():.2f}%)")
    print(f"Test Shape  : {data.X_test.shape} (Fault Rate={100*data.y_test.mean():.2f}%)")
    print("Verified: StandardScaler fit strictly on X_train before Val/Test transforms.")


if __name__ == "__main__":
    main()

