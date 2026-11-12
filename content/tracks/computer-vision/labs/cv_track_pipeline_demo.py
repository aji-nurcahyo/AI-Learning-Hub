"""Track A (Computer Vision) Applied Lab: Lightweight Spatial Filtering & CPU Inference."""

import time
import numpy as np
import torch
import torch.nn as nn


class TinyVisionClassifier(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_synthetic_shapes(n_per_class: int = 40, seed: int = 42):
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for cls in range(3):
        for _ in range(n_per_class):
            img = rng.normal(0.05, 0.03, size=(3, 24, 24)).astype(np.float32)
            if cls == 0:
                img[0, 6:18, 6:18] += 0.8   # Red square signal
            elif cls == 1:
                img[1, 4:20, 10:14] += 0.8  # Green vertical bar
            else:
                img[2, 10:14, 4:20] += 0.8  # Blue horizontal bar
            imgs.append(np.clip(img, 0.0, 1.0))
            labels.append(cls)
    return torch.tensor(np.array(imgs)), torch.tensor(np.array(labels))


def main():
    torch.manual_seed(42)
    X, y = make_synthetic_shapes(n_per_class=40, seed=42)
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    X_tr, X_te = X[:90], X[90:]
    y_tr, y_te = y[:90], y[90:]

    model = TinyVisionClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(35):
        opt.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1)
        acc = float((preds == y_te).float().mean().item())
        t0 = time.perf_counter()
        for _ in range(100):
            _ = model(X_te[:1])
        lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    params = sum(p.numel() for p in model.parameters())
    print("=" * 72)
    print("TRACK A (COMPUTER VISION): LIGHTWEIGHT VISUAL INFERENCE LAB")
    print("=" * 72)
    print(f"TinyVisionClassifier Params : {params:,}")
    print(f"Synthetic Test Accuracy     : {acc * 100:.2f}% (N={len(y_te)} unseen)")
    print(f"Mean CPU Latency (1 image)  : {lat_ms:.4f} ms")
    assert acc >= 0.90


if __name__ == "__main__":
    main()

