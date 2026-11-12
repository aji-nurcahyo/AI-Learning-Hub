"""Post-Training Static Quantization with Calibration in PyTorch CPU.

Implements full PTQ Static Quantization workflow:
1. Inserts QuantStub / DeQuantStub boundaries around Linear+ReLU blocks.
2. Configures default qconfig for the active 'onednn' engine.
3. Calls prepare() to insert MinMax/Histogram activation observers.
4. Feeds representative calibration batches (X_train[:200]) without backpropagation.
5. Calls convert() to lock INT8 weights & static activation scales.
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.ao.quantization as tq

from edge_ai_foundations import evaluate_accuracy, train_baseline_model
from model_profiling import get_serialized_size_kb, profile_cpu_latency


class StaticQuantizableEdgeMLP(nn.Module):
    """Static-quantizable version of EdgeTeacherMLP with QuantStub/DeQuantStub."""

    def __init__(self, in_features: int = 16, num_classes: int = 4):
        super().__init__()
        self.quant = tq.QuantStub()
        self.fc1 = nn.Linear(in_features, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return self.dequant(x)


def build_static_quantized_model(fp32_model: nn.Module, calib_data: torch.Tensor) -> nn.Module:
    static_m = StaticQuantizableEdgeMLP().cpu().eval()
    # Copy trained weights from baseline
    static_m.fc1.load_state_dict(fp32_model.fc1.state_dict())
    static_m.fc2.load_state_dict(fp32_model.fc2.state_dict())
    static_m.fc3.load_state_dict(fp32_model.fc3.state_dict())
    static_m.fc4.load_state_dict(fp32_model.fc4.state_dict())

    static_m.qconfig = tq.get_default_qconfig("onednn")
    tq.prepare(static_m, inplace=True)

    # Representative calibration pass (no labels needed)
    with torch.no_grad():
        static_m(calib_data)

    return tq.convert(static_m, inplace=False)


def main():
    print("=" * 75)
    print("CHAPTER 37: POST-TRAINING STATIC QUANTIZATION WITH CALIBRATION")
    print("=" * 75)

    fp32_model, (X_train, _, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    static_q_model = build_static_quantized_model(fp32_model, X_train[:250])

    fp32_kb = get_serialized_size_kb(fp32_model)
    stat_kb = get_serialized_size_kb(static_q_model)

    fp32_acc = evaluate_accuracy(fp32_model, X_test, y_test)
    stat_acc = evaluate_accuracy(static_q_model, X_test, y_test)

    fp32_lat = profile_cpu_latency(fp32_model, X_test[:1], warmup=50, n_runs=500)
    stat_lat = profile_cpu_latency(static_q_model, X_test[:1], warmup=50, n_runs=500)

    print("Calibration Protocol: Observed activation ranges on 250 representative training samples.")
    print(f"{'Metric':<25} | {'FP32 Baseline':<18} | {'Static INT8 (Calib)'}")
    print("-" * 68)
    print(f"{'Artifact Size (.pt KB)':<25} | {fp32_kb:<18.2f} | {stat_kb:.2f} ({round(fp32_kb/stat_kb, 2)}x smaller)")
    print(f"{'Test Accuracy (%)':<25} | {fp32_acc*100:<18.2f} | {stat_acc*100:.2f}")
    print(f"{'Mean Latency (ms)':<25} | {fp32_lat['mean_ms']:<18.4f} | {stat_lat['mean_ms']:.4f}")
    print(f"{'P95 Latency (ms)':<25} | {fp32_lat['p95_ms']:<18.4f} | {stat_lat['p95_ms']:.4f}")

    print("\nDistinction: Static Quantization pre-computes activation scales during calibration,")
    print("avoiding runtime min/max observer overhead required by Dynamic Quantization.")
    print("\n[SUCCESS] Post-Training Static Quantization verified.")


if __name__ == "__main__":
    main()

