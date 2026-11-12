"""
freeze_unfreeze_layers.py
Demonstrasi pembekuan dan pencairan lapisan secara progresif (Layer Freezing & Progressive Unfreezing),
inspeksi flag `requires_grad`, serta kalkulasi jumlah parameter terlatih per blok arsitektur.
"""

import torch
import torch.nn as nn

class MultiStageDeepNet(nn.Module):
    """Jaringan saraf 4-tahap bertingkat untuk studi pembekuan lapisan."""
    def __init__(self, in_features=16):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.stage3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        self.head = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

def print_parameter_status(model, stage_description):
    print(f"\n--- {stage_description} ---")
    print(f"{'Nama Lapisan / Parameter':<30} {'Bentuk Tensor':<18} {'Trainable?':<12} {'Jumlah Elemen'}")
    print("-" * 75)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_el = param.numel()
        total_params += num_el
        if param.requires_grad:
            trainable_params += num_el
        status = "YA (Aktif)" if param.requires_grad else "TIDAK (Beku)"
        print(f"{name:<30} {str(list(param.shape)):<18} {status:<12} {num_el:,}")
        
    pct = (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0
    print("-" * 75)
    print(f"Total Parameter      : {total_params:,}")
    print(f"Parameter Trainable  : {trainable_params:,} ({pct:.2f}%)")
    print(f"Parameter Frozen     : {total_params - trainable_params:,} ({100.0 - pct:.2f}%)")

def main():
    print("=== Progressive Unfreezing & Layer Inspection ===")
    model = MultiStageDeepNet(in_features=16)
    
    # Kondisi 1: Seluruh Lapisan Aktif (Full Fine-Tuning Default)
    print_parameter_status(model, "Kondisi 1: Inisialisasi Penuh (Semua Parameter Trainable)")
    
    # Kondisi 2: Feature Extraction Murni (Hanya Head Trainable)
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    print_parameter_status(model, "Kondisi 2: Linear Probing (Membekukan Stage 1, 2, 3)")
    
    # Kondisi 3: Progressive Unfreezing Tahap 1 (Cairkan Stage 3 + Head)
    for param in model.stage3.parameters():
        param.requires_grad = True
    print_parameter_status(model, "Kondisi 3: Partial Fine-Tuning (Mencairkan Stage 3 & Head)")
    
    # Kondisi 4: Progressive Unfreezing Tahap 2 (Cairkan Stage 2, 3 + Head)
    for param in model.stage2.parameters():
        param.requires_grad = True
    print_parameter_status(model, "Kondisi 4: Deep Partial Fine-Tuning (Mencairkan Stage 2, 3 & Head)")

if __name__ == "__main__":
    main()

