"""
partial_fine_tuning.py
Implementasi Partial Fine-Tuning dengan Differential Learning Rates:
- Membekukan lapisan bawah (lapisan umum/general feature detectors)
- Membuka lapisan atas (lapisan spesifik tugas/task-specific representations)
- Menerapkan laju pembelajaran berbeda: lr lebih rendah untuk backbone, lr lebih tinggi untuk classifier head
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class HierarchicalFeatureNet(nn.Module):
    """Jaringan saraf bertingkat dengan 3 blok representasi + kepala."""
    def __init__(self, in_features=12, hidden_dim=32, num_classes=3):
        super().__init__()
        # Lapisan Bawah: Detektor Fitur Dasar (General Lower Features)
        self.block1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        # Lapisan Tengah: Representasi Menengah (Mid-level Features)
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        # Lapisan Atas: Representasi Spesifik (High-level Domain Features)
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        # Kepala Klasifikasi Baru
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

def main():
    print("=== Eksperimen Partial Fine-Tuning dengan Differential Learning Rates ===")
    
    model = HierarchicalFeatureNet(in_features=12, hidden_dim=32, num_classes=3)
    
    # 1. Konfigurasi Pembekuan Lapisan
    print("[1] Membekukan Block 1 dan Block 2 (General Lower Features)...")
    for param in model.block1.parameters():
        param.requires_grad = False
    for param in model.block2.parameters():
        param.requires_grad = False
        
    # Block 3 dan Classifier tetap dibuka (Trainable)
    for param in model.block3.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameter     : {total_params:,}")
    print(f"Parameter Trainable : {trainable_params:,} ({(trainable_params/total_params)*100:.1f}%)")
    
    # 2. Konfigurasi Differential Learning Rates via Optimizer Parameter Groups
    print("\n[2] Menetapkan Laju Pembelajaran Bertingkat (Differential LRs):")
    # Lapisan atas backbone dilatih dengan LR lambat agar tidak merusak representasi (1e-4)
    # Kepala baru dilatih dengan LR cepat (1e-2) agar cepat beradaptasi
    lr_backbone = 1e-4
    lr_head = 1e-2
    
    optimizer = optim.AdamW([
        {'params': model.block3.parameters(), 'lr': lr_backbone, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': lr_head, 'weight_decay': 1e-3}
    ])
    
    print(f"   - Block 3 (Backbone Top Layer) LR : {lr_backbone}")
    print(f"   - Classifier Head LR              : {lr_head}")
    
    # 3. Simulasi Data dan Langkah Optimasi
    X_train = torch.randn(150, 12)
    y_train = torch.randint(0, 3, (150,))
    criterion = nn.CrossEntropyLoss()
    
    print("\n[3] Memulai 15 Epoch Pelatihan...")
    for epoch in range(1, 16):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        
        # Gradient Clipping untuk stabilitas
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 5 == 0:
            acc = (logits.argmax(dim=-1) == y_train).float().mean().item() * 100.0
            print(f"   Epoch {epoch:2d}/15 -> Loss: {loss.item():.4f} | Train Acc: {acc:.2f}%")
            
    # Verifikasi bahwa bobot Block 1 dan 2 benar-benar tidak terupdate gradiennya
    b1_grad = any(p.grad is not None for p in model.block1.parameters())
    b3_grad = any(p.grad is not None for p in model.block3.parameters())
    print("\n[4] Verifikasi Aliran Gradien:")
    print(f"   - Gradien pada Block 1 (Beku) : {b1_grad} (Harus False)")
    print(f"   - Gradien pada Block 3 (Aktif): {b3_grad} (Harus True)")
    print("\nKesimpulan: Partial Fine-Tuning mempertahankan fitur fondasi sembari membiarkan")
    print("lapisan atas beradaptasi terhadap domain target secara terkontrol.")

if __name__ == "__main__":
    main()

