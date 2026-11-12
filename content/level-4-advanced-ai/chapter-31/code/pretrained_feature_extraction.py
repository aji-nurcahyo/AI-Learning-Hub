"""
pretrained_feature_extraction.py
Demonstrasi ekstraksi fitur (Feature Extraction):
- Pembekuan bobot backbone via `param.requires_grad = False`
- Penggantian kepala klasifikasi (Classification Head)
- Perbedaan krusial mode `train()` vs `eval()` pada lapisan Dropout dan BatchNorm
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class SyntheticPretrainedVisionBackbone(nn.Module):
    """
    Simulasi backbone penglihatan komputer (Vision Backbone)
    dengan lapisan Konvolusi, BatchNorm, ReLU, dan Dropout.
    """
    def __init__(self, in_channels=3, feature_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(16, feature_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(feature_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [Batch, Channels, Length]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x.squeeze(-1) # [Batch, feature_dim]

def main():
    print("=== Feature Extraction & Runtime Mode Analysis ===")
    
    # 1. Instansiasi Backbone Pra-terlatih
    backbone = SyntheticPretrainedVisionBackbone(in_channels=3, feature_dim=32)
    
    # Hitung total parameter awal
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"Total parameter backbone pra-terlatih: {total_params:,}")
    
    # 2. Pembekuan Bobot Backbone (Freezing Weights)
    print("\n[Langkah 1] Membekukan seluruh parameter backbone...")
    for name, param in backbone.named_parameters():
        param.requires_grad = False
        
    trainable_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Parameter trainable di backbone setelah pembekuan: {trainable_backbone}")
    
    # 3. Memasang Kepala Klasifikasi Baru (Target Task: 5 Kelas)
    print("\n[Langkah 2] Memasang kepala klasifikasi baru (Linear Probing Head)...")
    new_classifier_head = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 5) # 5 kelas target
    )
    
    full_model = nn.Sequential(backbone, new_classifier_head)
    
    trainable_total = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    frozen_total = sum(p.numel() for p in full_model.parameters() if not p.requires_grad)
    print(f"Parameter Beku (Frozen)    : {frozen_total:,}")
    print(f"Parameter Latih (Trainable): {trainable_total:,} (Hanya kepala baru)")
    
    # 4. Analisis Mode train() vs eval() pada Bobot Beku
    print("\n[Langkah 3] Pentingnya Pengaturan Mode eval() vs train():")
    dummy_input = torch.randn(4, 3, 10)
    
    # Mode train(): Dropout aktif dan BatchNorm menggunakan statistik batch berjalan
    backbone.train()
    out_train_1 = backbone(dummy_input)
    out_train_2 = backbone(dummy_input)
    diff_train = (out_train_1 - out_train_2).abs().max().item()
    print(f"Perbedaan 2 inferensi berturut-turut pada mode train(): {diff_train:.6f}")
    print("-> Pada mode train(), Dropout menghasilkan stokastisitas meskipun bobot dibekukan!")
    
    # Mode eval(): Dropout nonaktif dan BatchNorm menggunakan running mean/var beku
    backbone.eval()
    out_eval_1 = backbone(dummy_input)
    out_eval_2 = backbone(dummy_input)
    diff_eval = (out_eval_1 - out_eval_2).abs().max().item()
    print(f"Perbedaan 2 inferensi berturut-turut pada mode eval() : {diff_eval:.6f}")
    print("-> Pada mode eval(), representasi fitur bersifat deterministik murni.")
    
    print("\nKesimpulan: Membekukan bobot (requires_grad = False) TIDAK otomatis membekukan")
    print("perilaku runtime. Backbone wajib disetel ke mode eval() saat feature extraction")
    print("agar inferensi deterministik dan statistik BatchNorm tidak terkontaminasi target task.")

if __name__ == "__main__":
    main()

