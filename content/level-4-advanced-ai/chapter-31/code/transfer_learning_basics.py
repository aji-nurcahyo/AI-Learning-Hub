"""
transfer_learning_basics.py
Demonstrasi konsep dasar Transfer Learning:
Perbedaan pelatihan dari awal (Training from Scratch) vs Pretrained Backbone,
definisi Source Task dan Target Task, serta representasi fitur terpakai ulang (Feature Reuse).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set deterministic random seed
torch.manual_seed(42)
np.random.seed(42)

def generate_source_data(n_samples=500, input_dim=8, n_classes=4):
    """
    Source Task: Klasifikasi 4-kelas pada ruang fitur 8-dimensi.
    Merepresentasikan domain umum dengan volume sampel lebih banyak.
    """
    X = torch.randn(n_samples, input_dim)
    # Pembagian kelas berdasarkan kombinasi linear
    y = ((X[:, 0] > 0).long() * 2 + (X[:, 1] > 0).long()) % n_classes
    return X, y

def generate_target_data(n_samples=80, input_dim=8, n_classes=2):
    """
    Target Task: Klasifikasi biner pada subset fitur serupa dengan sampel terbatas (low-data regime).
    """
    X = torch.randn(n_samples, input_dim)
    # Target task memanfaatkan sebagian representasi yang dipelajari source task
    y = ((X[:, 0] + X[:, 1] > 0)).long()
    return X, y

class PretrainedBackbone(nn.Module):
    """Backbone jaringan saraf untuk ekstraksi fitur generik."""
    def __init__(self, input_dim=8, hidden_dim=32, repr_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class SourceModel(nn.Module):
    """Model lengkap untuk Source Task."""
    def __init__(self, backbone, repr_dim=16, n_classes=4):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(repr_dim, n_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def pretrain_source_model(epochs=30):
    print("=== [1] Pre-training Model pada Source Task (Multi-Class) ===")
    X_src, y_src = generate_source_data(n_samples=600, input_dim=8, n_classes=4)
    backbone = PretrainedBackbone(input_dim=8, hidden_dim=32, repr_dim=16)
    model = SourceModel(backbone, repr_dim=16, n_classes=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_src)
        loss = criterion(logits, y_src)
        loss.backward()
        optimizer.step()
    
    preds = logits.argmax(dim=-1)
    acc = (preds == y_src).float().mean().item() * 100.0
    print(f"Pre-training selesai ({epochs} epoch). Akurasi Source Task: {acc:.2f}%, Final Loss: {loss.item():.4f}")
    return backbone

def main():
    print("Konsep Dasar Transfer Learning & Feature Reuse")
    print("-" * 50)
    
    # 1. Pretrain pada Source Task
    pretrained_backbone = pretrain_source_model(epochs=30)
    
    # 2. Definisikan Target Task (Low Data Regime: 80 sampel)
    X_tgt, y_tgt = generate_target_data(n_samples=80, input_dim=8, n_classes=2)
    X_val, y_val = generate_target_data(n_samples=50, input_dim=8, n_classes=2)
    
    # 3. Model A: Training from Scratch pada Target Task
    scratch_backbone = PretrainedBackbone(input_dim=8, hidden_dim=32, repr_dim=16)
    model_scratch = nn.Sequential(scratch_backbone, nn.Linear(16, 2))
    
    opt_scratch = optim.Adam(model_scratch.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    
    for _ in range(25):
        opt_scratch.zero_grad()
        loss = crit(model_scratch(X_tgt), y_tgt)
        loss.backward()
        opt_scratch.step()
        
    val_acc_scratch = (model_scratch(X_val).argmax(dim=-1) == y_val).float().mean().item() * 100.0
    
    # 4. Model B: Transfer Learning (Linear Probing pada Pretrained Backbone)
    # Freeze backbone
    for param in pretrained_backbone.parameters():
        param.requires_grad = False
        
    model_transfer = nn.Sequential(pretrained_backbone, nn.Linear(16, 2))
    opt_transfer = optim.Adam(model_transfer[1].parameters(), lr=0.01)
    
    for _ in range(25):
        opt_transfer.zero_grad()
        loss = crit(model_transfer(X_tgt), y_tgt)
        loss.backward()
        opt_transfer.step()
        
    val_acc_transfer = (model_transfer(X_val).argmax(dim=-1) == y_val).float().mean().item() * 100.0
    
    print("\n=== [2] Evaluasi Target Task pada Data Validasi ===")
    print(f"Akurasi Model Scratch (Tanpa Pretraining) : {val_acc_scratch:.2f}%")
    print(f"Akurasi Transfer Learning (Pretrained)     : {val_acc_transfer:.2f}%")
    gain = val_acc_transfer - val_acc_scratch
    print(f"Keuntungan Transfer Learning (Delta Acc)  : {gain:+.2f}%")
    print("\nKesimpulan: Pada regim data sedikit, representasi terpakai ulang (feature reuse)")
    print("dari source task memberikan priors yang kokoh dan mengurangi risiko overfitting.")

if __name__ == "__main__":
    main()

