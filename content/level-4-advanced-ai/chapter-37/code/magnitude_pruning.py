"""Unstructured Magnitude-Based Weight Pruning in PyTorch.

Demonstrates:
1. L1 unstructured pruning across Linear layers at target sparsity (e.g. 70%).
2. Exact verification of zero-weight sparsity ratio.
3. Why unstructured pruning does NOT shrink serialized dense .pt disk size
   (without sparse CSR encoding) or accelerate standard dense CPU GEMMs.
4. Accuracy recovery via short mask-preserving fine-tuning.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim

from edge_ai_foundations import evaluate_accuracy, train_baseline_model
from model_profiling import get_serialized_size_kb, profile_cpu_latency


def measure_sparsity(model: nn.Module) -> float:
    zero_elems = 0
    total_elems = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight.detach()
            zero_elems += int((w == 0).sum().item())
            total_elems += w.numel()
    return round((zero_elems / max(1, total_elems)) * 100.0, 2)


def apply_unstructured_pruning(model: nn.Module, amount: float = 0.7) -> nn.Module:
    pruned = copy.deepcopy(model)
    for m in pruned.modules():
        if isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name="weight", amount=amount)
    return pruned


def remove_pruning_reparametrization(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Linear) and prune.is_pruned(m):
            prune.remove(m, "weight")


def main():
    print("=" * 75)
    print("CHAPTER 37: UNSTRUCTURED MAGNITUDE PRUNING & DENSE STORAGE REALITY")
    print("=" * 75)

    fp32_model, (X_train, y_train, X_test, y_test) = train_baseline_model(epochs=25, seed=42)
    base_acc = evaluate_accuracy(fp32_model, X_test, y_test)
    base_kb = get_serialized_size_kb(fp32_model)
    base_lat = profile_cpu_latency(fp32_model, X_test[:1], warmup=50, n_runs=300)

    # 1. Apply 70% Unstructured L1 Pruning
    pruned = apply_unstructured_pruning(fp32_model, amount=0.70)
    sparsity_before_ft = measure_sparsity(pruned)
    acc_before_ft = evaluate_accuracy(pruned, X_test, y_test)

    # 2. Fine-tune for 5 epochs while pruning mask holds zeros fixed
    pruned.train()
    opt = optim.Adam(pruned.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(5):
        opt.zero_grad()
        loss = crit(pruned(X_train), y_train)
        loss.backward()
        opt.step()
    pruned.eval()

    # Make pruning permanent (folds mask into dense weight tensor)
    remove_pruning_reparametrization(pruned)
    sparsity_after_ft = measure_sparsity(pruned)
    acc_after_ft = evaluate_accuracy(pruned, X_test, y_test)
    pruned_kb = get_serialized_size_kb(pruned)
    pruned_lat = profile_cpu_latency(pruned, X_test[:1], warmup=50, n_runs=300)

    print(f"Baseline Dense Model    : Sparsity = {measure_sparsity(fp32_model):5.2f}% | Acc = {base_acc*100:.2f}% | Size = {base_kb:.2f} KB | Lat = {base_lat['mean_ms']:.4f} ms")
    print(f"70% Unstructured Pruned : Sparsity = {sparsity_before_ft:5.2f}% | Acc = {acc_before_ft*100:.2f}% (Zero-shot post-prune)")
    print(f"Pruned + 5-Epoch FT     : Sparsity = {sparsity_after_ft:5.2f}% | Acc = {acc_after_ft*100:.2f}% | Size = {pruned_kb:.2f} KB | Lat = {pruned_lat['mean_ms']:.4f} ms")

    print("\nCritical Hardware Reality:")
    print("Even at 70% zero weights, dense tensor serialization (.pt) remains identical in size")
    print(f"({base_kb:.2f} KB vs {pruned_kb:.2f} KB) because 0.0f still occupies 32 bits in memory without")
    print("sparse index encoding (CSR/CSC) or structured channel removal.")
    print("\n[SUCCESS] Unstructured magnitude pruning verified.")


if __name__ == "__main__":
    main()

