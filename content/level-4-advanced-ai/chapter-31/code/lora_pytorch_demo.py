"""
lora_pytorch_demo.py
Implementasi Modul Kustom LoRALinear di PyTorch:
- Merancang kelas turunan `nn.Module` yang mengintegrasikan lapisan linier beku dan adaptor rank rendah
- Memverifikasi aliran gradien (gradient flow) hanya mengalir ke matriks A dan B
- Demonstrasi penggabungan bobot (Weight Merging) untuk inferensi tanpa penalti latensi
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class LoRALinear(nn.Module):
    """
    Lapisan Linear Teradaptasi LoRA:
    y = x @ W_base.T + (alpha / r) * (x @ A.T @ B.T) + bias
    """
    def __init__(self, in_features, out_features, r=4, lora_alpha=8.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # 1. Base Linear Layer (Dibekukan)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False
        if bias:
            self.base_layer.bias.requires_grad = False
            
        # 2. Matriks Adaptor LoRA
        # A: Inisialisasi Kaiming/Normal
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        
        # B: Inisialisasi Nol Mutlak
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Flag status penggabungan
        self.merged = False

    def forward(self, x):
        if self.merged:
            # Jalur inferensi cepat setelah digabung
            return self.base_layer(x)
        
        # Jalur pelatihan modular
        base_out = self.base_layer(x)
        # Adaptor: x @ A.T -> [Batch, r] kemudian @ B.T -> [Batch, out_features]
        adapter_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + adapter_out
    
    def merge_weights(self):
        """Menggabungkan Delta W ke bobot dasar untuk zero latency inference."""
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True
            print("-> Bobot LoRA berhasil digabungkan ke base_layer (Merged Mode Aktif).")
            
    def unmerge_weights(self):
        """Memisahkan kembali bobot adaptor dari bobot dasar."""
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False
            print("-> Bobot LoRA dipisahkan kembali dari base_layer.")

def main():
    print("=== Demonstrasi LoRALinear dalam PyTorch ===")
    
    # Inisialisasi Lapisan
    lora_layer = LoRALinear(in_features=16, out_features=16, r=4, lora_alpha=8.0)
    
    print("\n[1] Verifikasi Status Gradien:")
    for name, param in lora_layer.named_parameters():
        print(f"   - {name:<25} | Shape: {str(list(param.shape)):<14} | requires_grad: {param.requires_grad}")
        
    trainable_params = [p for p in lora_layer.parameters() if p.requires_grad]
    print(f"Jumlah parameter aktif dalam optimizer: {sum(p.numel() for p in trainable_params)} (Hanya A dan B)")
    
    # 2. Simulasi Langkah Pelatihan
    print("\n[2] Menjalankan Langkah Propagasi Balik (Backward Pass)...")
    optimizer = optim.Adam(trainable_params, lr=0.01)
    criterion = nn.MSELoss()
    
    x_input = torch.randn(8, 16)
    target = torch.randn(8, 16)
    
    optimizer.zero_grad()
    output = lora_layer(x_input)
    loss = criterion(output, target)
    loss.backward()
    
    print("Verifikasi Keberadaan Gradien:")
    print(f"   - Gradien base_layer.weight : {lora_layer.base_layer.weight.grad is not None} (Harus False)")
    print(f"   - Gradien lora_A            : {lora_layer.lora_A.grad is not None} (Harus True)")
    print(f"   - Gradien lora_B            : {lora_layer.lora_B.grad is not None} (Harus True)")
    
    optimizer.step()
    print("Langkah optimasi berhasil dieksekusi tanpa menyentuh bobot dasar.")
    
    # 3. Uji Penggabungan Bobot (Zero Latency Merging)
    print("\n[3] Uji Penggabungan Bobot (Merging) untuk Produksi:")
    test_x = torch.randn(2, 16)
    
    # Output sebelum digabung
    lora_layer.eval()
    with torch.no_grad():
        out_unmerged = lora_layer(test_x)
        
    # Gabungkan bobot
    lora_layer.merge_weights()
    with torch.no_grad():
        out_merged = lora_layer(test_x)
        
    diff = (out_unmerged - out_merged).abs().max().item()
    print(f"Selisih Output Sebelum vs Sesudah Merging : {diff:.8e}")
    assert diff < 1e-6, "Output sebelum dan sesudah merging harus identik!"
    
    print("\nKesimpulan: LoRALinear memungkinkan pelatihan hemat memori saat fine-tuning,")
    print("dan dapat digabungkan langsung ke model utama untuk inferensi tanpa overhead komputasi.")

if __name__ == "__main__":
    main()

