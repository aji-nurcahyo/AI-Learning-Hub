"""Edge AI Foundations and Shared Baseline Model.

Provides:
1. Synthetic IoT/Industrial Sensor Classification dataset (16 features, 4 classes, deterministic seed=42).
2. EdgeTeacherMLP baseline neural network (~45K parameters) sized meaningfully for
   CPU compression, pruning, quantization, and distillation studies.
3. Cloud vs Edge AI trade-off summary demonstration.
"""

import os
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_edge_sensor_dataset(
    n_samples: int = 2400,
    n_features: int = 16,
    n_classes: int = 4,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates leak-free standardized IoT/industrial sensor telemetry tensors."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=1.3,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )


class EdgeTeacherMLP(nn.Module):
    """Baseline FP32 neural network (16 -> 256 -> 128 -> 64 -> 4)."""

    def __init__(self, in_features: int = 16, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)


def train_baseline_model(epochs: int = 25, lr: float = 1e-3, seed: int = 42) -> Tuple[EdgeTeacherMLP, Tuple[torch.Tensor, ...]]:
    """Trains and returns the deterministic FP32 baseline model and data splits."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_test, y_test = get_edge_sensor_dataset(seed=seed)
    model = EdgeTeacherMLP(in_features=16, num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    return model, (X_train, y_train, X_test, y_test)


def evaluate_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X), dim=1)
        acc = float((preds == y).float().mean().item())
    return round(acc, 4)


def main():
    print("=" * 75)
    print("CHAPTER 37: EDGE AI FOUNDATIONS & BASELINE MODEL")
    print("=" * 75)

    model, (X_train, y_train, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    train_acc = evaluate_accuracy(model, X_train, y_train)
    test_acc = evaluate_accuracy(model, X_test, y_test)
    total_params = sum(p.numel() for p in model.parameters())

    print("Baseline Task: 4-Class Industrial/IoT Sensor Anomaly Classification (16 features)")
    print(f"- Architecture : Linear(16->256->128->64->4) = {total_params:,} parameters")
    print(f"- Train Acc    : {train_acc * 100:.2f}% (N={len(X_train)})")
    print(f"- Test Acc     : {test_acc * 100:.2f}% (N={len(X_test)})")

    print("\nCloud Inference vs On-Device Edge Inference Trade-off:")
    print("- Cloud Inference   : High compute, but suffers 50-300ms WAN latency & connectivity dependency.")
    print("- Edge Inference    : Sub-millisecond local response, strict data privacy, offline resilience,")
    print("                      but tightly constrained by SRAM (KB-MB), Flash storage, and battery.")
    print("\n[SUCCESS] Edge AI baseline dataset & FP32 model verified.")


if __name__ == "__main__":
    main()

