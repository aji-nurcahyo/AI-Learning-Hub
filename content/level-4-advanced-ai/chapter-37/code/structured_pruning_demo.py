"""Structured Neuron/Channel Pruning for Physical Compression.

Demonstrates how removing entire lowest-L1-norm neurons across hidden layers
physically shrinks layer dimensions:
   Baseline: 16 -> 256 -> 128 -> 64 -> 4
   Pruned (keep 50% neurons): 16 -> 128 -> 64 -> 32 -> 4
Delivers genuine reductions in parameter count, .pt disk size, and FLOPs.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from edge_ai_foundations import evaluate_accuracy, train_baseline_model
from model_profiling import get_serialized_size_kb, profile_cpu_latency


class CompactPrunedMLP(nn.Module):
    """Physically compressed MLP preserving retained high-magnitude neurons."""

    def __init__(self, in_f: int, h1: int, h2: int, h3: int, out_f: int):
        super().__init__()
        self.fc1 = nn.Linear(in_f, h1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(h2, h3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(h3, out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.fc4(x)


def apply_structured_l1_pruning(teacher_model: nn.Module, keep_ratio: float = 0.5) -> CompactPrunedMLP:
    """Prunes lowest-L1-norm output neurons from fc1, fc2, fc3 structurally."""
    with torch.no_grad():
        # Rank neurons of each hidden layer by L1 norm of output weights
        idx1 = torch.topk(torch.sum(torch.abs(teacher_model.fc1.weight), dim=1), k=int(256 * keep_ratio)).indices.sort().values
        idx2 = torch.topk(torch.sum(torch.abs(teacher_model.fc2.weight), dim=1), k=int(128 * keep_ratio)).indices.sort().values
        idx3 = torch.topk(torch.sum(torch.abs(teacher_model.fc3.weight), dim=1), k=int(64 * keep_ratio)).indices.sort().values

        compact = CompactPrunedMLP(16, len(idx1), len(idx2), len(idx3), 4)

        # Transfer retained weight slices and biases cleanly
        compact.fc1.weight.copy_(teacher_model.fc1.weight[idx1, :])
        compact.fc1.bias.copy_(teacher_model.fc1.bias[idx1])

        compact.fc2.weight.copy_(teacher_model.fc2.weight[idx2, :][:, idx1])
        compact.fc2.bias.copy_(teacher_model.fc2.bias[idx2])

        compact.fc3.weight.copy_(teacher_model.fc3.weight[idx3, :][:, idx2])
        compact.fc3.bias.copy_(teacher_model.fc3.bias[idx3])

        compact.fc4.weight.copy_(teacher_model.fc4.weight[:, idx3])
        compact.fc4.bias.copy_(teacher_model.fc4.bias)

    return compact


def main():
    print("=" * 75)
    print("CHAPTER 37: STRUCTURED PRUNING (PHYSICAL ARCHITECTURE COMPRESSION)")
    print("=" * 75)

    fp32_model, (X_train, y_train, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    base_params = sum(p.numel() for p in fp32_model.parameters())
    base_kb = get_serialized_size_kb(fp32_model)
    base_acc = evaluate_accuracy(fp32_model, X_test, y_test)
    base_lat = profile_cpu_latency(fp32_model, X_test[:1], warmup=50, n_runs=300)

    # Structurally prune 50% neurons in hidden layers
    compact = apply_structured_l1_pruning(fp32_model, keep_ratio=0.5)
    acc_zero_shot = evaluate_accuracy(compact, X_test, y_test)

    # Recover accuracy with 10-epoch fine-tuning
    compact.train()
    opt = optim.Adam(compact.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for _ in range(10):
        opt.zero_grad()
        loss = crit(compact(X_train), y_train)
        loss.backward()
        opt.step()
    compact.eval()

    comp_params = sum(p.numel() for p in compact.parameters())
    comp_kb = get_serialized_size_kb(compact)
    comp_acc = evaluate_accuracy(compact, X_test, y_test)
    comp_lat = profile_cpu_latency(compact, X_test[:1], warmup=50, n_runs=300)

    print(f"Baseline (16->256->128->64->4)   : Params = {base_params:,} | Size = {base_kb:.2f} KB | Acc = {base_acc*100:.2f}% | Lat = {base_lat['mean_ms']:.4f} ms")
    print(f"Structured Pruned (Zero-Shot)    : Params = {comp_params:,} | Acc = {acc_zero_shot*100:.2f}%")
    print(f"Structured Pruned + 10-Epoch FT  : Params = {comp_params:,} | Size = {comp_kb:.2f} KB | Acc = {comp_acc*100:.2f}% | Lat = {comp_lat['mean_ms']:.4f} ms")

    print(f"\nGenuine Physical Reduction:")
    print(f"- Parameter Reduction : {round((1 - comp_params/base_params)*100, 1)}% ({base_params:,} -> {comp_params:,})")
    print(f"- Disk Artifact Ratio : {round(base_kb / comp_kb, 2)}x smaller ({base_kb:.2f} KB -> {comp_kb:.2f} KB)")
    print("\n[SUCCESS] Structured pruning verified with physical parameter reduction.")


if __name__ == "__main__":
    main()

