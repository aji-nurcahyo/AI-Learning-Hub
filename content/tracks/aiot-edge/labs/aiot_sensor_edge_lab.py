"""Track C (AIoT & Edge AI) Applied Lab: Quantized Edge Footprint Simulation.

DISCLAIMER: Evaluates quantized size & CPU latency locally. Does NOT claim
physical validation on ESP32 or bare-metal ARM Cortex-M silicon.
"""

import io
import time
import numpy as np
import torch
import torch.nn as nn


def main():
    torch.manual_seed(42)
    torch.backends.quantized.engine = "onednn"

    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1.0, size=(400, 8)).astype(np.float32)
    y = ((0.8 * X[:, 0] + 0.6 * X[:, 1] - 0.5 * X[:, 2]) > 0.2).astype(np.int64)

    X_tr, X_te = torch.tensor(X[:300]), torch.tensor(X[300:])
    y_tr, y_te = torch.tensor(y[:300]), torch.tensor(y[300:])

    model = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(80):
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(X_tr), y_tr).backward()
        opt.step()

    model.eval()
    q_model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    def size_kb(m):
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        return len(buf.getvalue()) / 1024.0

    with torch.no_grad():
        fp32_acc = float((model(X_te).argmax(1) == y_te).float().mean().item())
        int8_acc = float((q_model(X_te).argmax(1) == y_te).float().mean().item())
        t0 = time.perf_counter()
        for _ in range(200):
            _ = q_model(X_te[:1])
        lat_ms = ((time.perf_counter() - t0) / 200.0) * 1000.0

    print("=" * 75)
    print("TRACK C (AIoT & EDGE AI): DYNAMIC INT8 TELEMETRY COMPRESSION LAB")
    print("=" * 75)
    print(f"FP32 Model : Size={size_kb(model):6.2f} KB | Test Acc={fp32_acc*100:.2f}%")
    print(f"INT8 Model : Size={size_kb(q_model):6.2f} KB | Test Acc={int8_acc*100:.2f}% | CPU Lat={lat_ms:.4f} ms")
    print("Note: Simulated CPU evaluation only (no physical ESP32 hardware claim).")
    assert int8_acc >= 0.85


if __name__ == "__main__":
    main()

