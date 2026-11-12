"""
negative_transfer_demo.py
Demonstrasi fenomena Transfer Negatif (Negative Transfer):
Kasus di mana representasi pra-terlatih dari Source Task yang tidak relevan
atau memiliki korelasi terbalik (spurious/conflicting features) justru
menghasilkan performa lebih buruk daripada model yang dilatih dari awal (Scratch).
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

def generate_dissimilar_tasks(n_samples=120):
    """
    Source Task: Fitur 0 berkolerasi positif kuat dengan label Y (+1 -> Kelas 1)
    Target Task: Fitur 0 sengaja memiliki aturan terbalik (+1 -> Kelas 0),
                 namun fitur 1 dan 2 menentukan label sebenarnya.
    """
    # Source Data
    X_src = torch.randn(n_samples, 6)
    y_src = (X_src[:, 0] > 0).long()
    
    # Target Data (Aturan kontradiktif dengan source)
    X_tgt = torch.randn(n_samples, 6)
    # Aturan target: Kombinasi fitur 1 dan 2
    y_tgt = (X_tgt[:, 1] * X_tgt[:, 2] > 0).long()
    
    # Test target
    X_tgt_test = torch.randn(80, 6)
    y_tgt_test = (X_tgt_test[:, 1] * X_tgt_test[:, 2] > 0).long()
    
    return (X_src, y_src), (X_tgt, y_tgt), (X_tgt_test, y_tgt_test)

def make_backbone():
    return nn.Sequential(
        nn.Linear(6, 24),
        nn.ReLU(),
        nn.Linear(24, 16),
        nn.ReLU()
    )

def main():
    print("=== Simulasi Transfer Negatif (Negative Transfer) ===")
    (X_src, y_src), (X_tgt, y_tgt), (X_tgt_test, y_tgt_test) = generate_dissimilar_tasks()
    
    # 1. Pretraining pada Source Task yang Berlawanan Sifat
    print("[1] Pretraining Backbone pada Tugas Sumber...")
    src_backbone = make_backbone()
    src_head = nn.Linear(16, 2)
    src_model = nn.Sequential(src_backbone, src_head)
    
    opt_src = optim.Adam(src_model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    for _ in range(35):
        opt_src.zero_grad()
        loss = crit(src_model(X_src), y_src)
        loss.backward()
        opt_src.step()
        
    print(f"Pretraining selesai. Loss: {loss.item():.4f}")
    
    # 2. Transfer Learning (Linear Probing pada Backbone yang Bias)
    print("\n[2] Transfer Learning (Linear Probing dengan Backbone Sumber)...")
    for param in src_backbone.parameters():
        param.requires_grad = False
    transfer_model = nn.Sequential(src_backbone, nn.Linear(16, 2))
    opt_transfer = optim.Adam(transfer_model[1].parameters(), lr=0.01)
    
    for _ in range(25):
        opt_transfer.zero_grad()
        loss = crit(transfer_model(X_tgt), y_tgt)
        loss.backward()
        opt_transfer.step()
        
    acc_transfer = (transfer_model(X_tgt_test).argmax(dim=-1) == y_tgt_test).float().mean().item() * 100.0
    
    # 3. Model Scratch (Tanpa Pretraining)
    print("\n[3] Model Scratch (Pelatihan Mandiri dari Nol)...")
    scratch_backbone = make_backbone()
    scratch_model = nn.Sequential(scratch_backbone, nn.Linear(16, 2))
    opt_scratch = optim.Adam(scratch_model.parameters(), lr=0.01)
    
    for _ in range(25):
        opt_scratch.zero_grad()
        loss = crit(scratch_model(X_tgt), y_tgt)
        loss.backward()
        opt_scratch.step()
        
    acc_scratch = (scratch_model(X_tgt_test).argmax(dim=-1) == y_tgt_test).float().mean().item() * 100.0
    
    print("\n" + "=" * 60)
    print(f"{'Metode Pelatihan':<30} {'Akurasi Target Test Set'}")
    print("=" * 60)
    print(f"{'Model Scratch (Tanpa Prior)':<30} {acc_scratch:.2f}%")
    print(f"{'Transfer Learning (Prior Bias)':<30} {acc_transfer:.2f}%")
    print("=" * 60)
    delta = acc_transfer - acc_scratch
    print(f"Dampak Transfer: {delta:+.2f}%")
    if delta < 0:
        print("-> Terjadi Negative Transfer! Representasi dari source task merugikan")
        print("   karena fitur relevan target tertindas oleh representasi usang source task.")

if __name__ == "__main__":
    main()

