"""
full_fine_tuning.py
Demonstrasi Full Fine-Tuning dan Fenomena Catastrophic Forgetting:
- Memperbarui seluruh parameter model pada tugas target
- Membandingkan perolehan performa target task vs penurunan retensi pada source task (Forgetting)
- Strategi mitigasi: penggunaan conservative learning rate dan weight decay
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

def build_model(input_dim=8, hidden_dim=24, n_classes=2):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_classes)
    )

def main():
    print("=== Eksperimen Full Fine-Tuning & Catastrophic Forgetting ===")
    
    # 1. Dataset Source Task (Tugas A: Klasifikasi Biner Aturan Fitur 0 & 1)
    X_src_train = torch.randn(300, 8)
    y_src_train = ((X_src_train[:, 0] > 0) & (X_src_train[:, 1] > 0)).long()
    X_src_test  = torch.randn(100, 8)
    y_src_test  = ((X_src_test[:, 0] > 0) & (X_src_test[:, 1] > 0)).long()
    
    # 2. Dataset Target Task (Tugas B: Klasifikasi Biner Aturan Fitur 6 & 7)
    X_tgt_train = torch.randn(100, 8)
    y_tgt_train = ((X_tgt_train[:, 6] + X_tgt_train[:, 7] > 0)).long()
    X_tgt_test  = torch.randn(80, 8)
    y_tgt_test  = ((X_tgt_test[:, 6] + X_tgt_test[:, 7] > 0)).long()
    
    # 3. Pretraining pada Source Task (Tugas A)
    print("\n[Langkah 1] Melatih model pada Source Task (Tugas A)...")
    model = build_model(input_dim=8, hidden_dim=24, n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer_src = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(30):
        optimizer_src.zero_grad()
        loss = criterion(model(X_src_train), y_src_train)
        loss.backward()
        optimizer_src.step()
        
    acc_src_initial = (model(X_src_test).argmax(dim=-1) == y_src_test).float().mean().item() * 100.0
    print(f"Performa Awal pada Source Task (Tugas A) : {acc_src_initial:.2f}%")
    
    # 4. Skenario A: Full Fine-Tuning Agresif (LR Tinggi: 0.02)
    print("\n[Langkah 2] Skenario A: Full Fine-Tuning Agresif pada Target Task (LR = 0.02)...")
    model_aggressive = torch.load_state_dict if False else build_model(8, 24, 2)
    model_aggressive.load_state_dict(model.state_dict())
    opt_aggr = optim.Adam(model_aggressive.parameters(), lr=0.02)
    
    for _ in range(25):
        opt_aggr.zero_grad()
        loss = criterion(model_aggressive(X_tgt_train), y_tgt_train)
        loss.backward()
        opt_aggr.step()
        
    acc_tgt_aggr = (model_aggressive(X_tgt_test).argmax(dim=-1) == y_tgt_test).float().mean().item() * 100.0
    acc_src_after_aggr = (model_aggressive(X_src_test).argmax(dim=-1) == y_src_test).float().mean().item() * 100.0
    
    # 5. Skenario B: Full Fine-Tuning Konservatif (LR Rendah: 0.001 + Weight Decay)
    print("\n[Langkah 3] Skenario B: Full Fine-Tuning Konservatif (LR = 0.001)...")
    model_conservative = build_model(8, 24, 2)
    model_conservative.load_state_dict(model.state_dict())
    opt_cons = optim.AdamW(model_conservative.parameters(), lr=0.001, weight_decay=1e-4)
    
    for _ in range(25):
        opt_cons.zero_grad()
        loss = criterion(model_conservative(X_tgt_train), y_tgt_train)
        loss.backward()
        opt_cons.step()
        
    acc_tgt_cons = (model_conservative(X_tgt_test).argmax(dim=-1) == y_tgt_test).float().mean().item() * 100.0
    acc_src_after_cons = (model_conservative(X_src_test).argmax(dim=-1) == y_src_test).float().mean().item() * 100.0
    
    print("\n" + "=" * 65)
    print(f"{'Strategi Pelatihan':<28} {'Akurasi Target':<18} {'Akurasi Source (Retensi)'}")
    print("=" * 65)
    print(f"{'Sebelum Fine-Tuning':<28} {'-':<18} {acc_src_initial:.2f}%")
    print(f"{'Agresif (LR=0.02)':<28} {acc_tgt_aggr:.2f}%{' ':10} {acc_src_after_aggr:.2f}% (Forgetting: {acc_src_initial - acc_src_after_aggr:+.1f}%)")
    print(f"{'Konservatif (LR=0.001)':<28} {acc_tgt_cons:.2f}%{' ':10} {acc_src_after_cons:.2f}% (Forgetting: {acc_src_initial - acc_src_after_cons:+.1f}%)")
    print("=" * 65)
    print("Kesimpulan: Full fine-tuning tanpa kendali laju pembelajaran dapat memicu")
    print("Catastrophic Forgetting di mana representasi tugas asal runtuh secara drastis.")

if __name__ == "__main__":
    main()

