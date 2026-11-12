"""
Chapter 29 - Large Language Models & Generative AI
Script 9: Parameter-Efficient Fine-Tuning & Low-Rank Adaptation (LoRA)
Mendemonstrasikan rancang bangun modul LoRA (Hu et al., 2021) menggunakan PyTorch:
Pembekuan bobot dasar (frozen W_0), pembaruan matriks dekomposisi rank rendah (B * A),
faktor penskalaan alpha/r, serta pembuktian penghematan parameter masif (>90%).
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Lapisan Linear yang diperkaya dengan adaptor Low-Rank Adaptation (LoRA).
    Rumus Maju: h = x * W_0^T + (alpha / r) * x * (B * A)^T
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 1. Bobot Dasar Terlatih (Pretrained Base Weight) - DIBEKUKAN (FROZEN)
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.base_layer.weight.requires_grad = False

        # 2. Matriks Adaptor LoRA Rank Rendah - DAPAT DILATIH (TRAINABLE)
        # Matriks A diinisialisasi Gaussian berdistribusi normal, B diinisialisasi 0
        # sehingga Delta W = B * A = 0 pada awal fine-tuning (tidak merusak model dasar)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Jalur utama model beku
        base_out = self.base_layer(x)
        # Jalur adaptor rank rendah
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


def main():
    print("=" * 70)
    print("CHAPTER 29: LOW-RANK ADAPTATION (LoRA) MECHANICS & VERIFICATION")
    print("=" * 70)

    d_in = 256
    d_out = 256
    rank = 4
    alpha = 8.0

    print(f"Dimensi Lapisan Linier : ({d_in} -> {d_out})")
    print(f"Peringkat Adaptor (r)  : {rank}")
    print(f"Faktor Skalasi (alpha) : {alpha} (Skala alpha/r = {alpha/rank:.1f})")

    # Inisialisasi modul
    lora_linear = LoRALinear(in_features=d_in, out_features=d_out, rank=rank, alpha=alpha)

    # 1. Audit Jumlah Parameter Terlatih vs Beku
    base_params = lora_linear.base_layer.weight.numel()
    lora_params = lora_linear.lora_A.numel() + lora_linear.lora_B.numel()
    total_params = base_params + lora_params
    trainable_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)

    saving_pct = (1.0 - (trainable_params / base_params)) * 100.0

    print("\n[AUDIT PARAMETER ALOKASI]")
    print(f"Parameter Model Dasar (Beku)   : {base_params:,} bobot")
    print(f"Parameter Matriks LoRA A (r x d): {lora_linear.lora_A.numel():,} bobot")
    print(f"Parameter Matriks LoRA B (d x r): {lora_linear.lora_B.numel():,} bobot")
    print(f"Total Parameter Adaptor Terlatih: {trainable_params:,} bobot")
    print(f"Persentase Penghematan Parameter: {saving_pct:.2f}%!")

    # 2. Uji Lintasan Maju (Forward Pass)
    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, d_in)
    out = lora_linear(x)
    print(f"\n[FORWARD PASS TEST]")
    print(f"Bentuk Tensor Masukan : {x.shape}")
    print(f"Bentuk Tensor Keluaran: {out.shape}")

    # 3. Uji Aliran Gradien (Backward Pass & Verification)
    print("\n[VERIFIKASI ALIRAN GRADIEN]")
    loss = out.sum()
    loss.backward()

    base_has_grad = lora_linear.base_layer.weight.grad is not None
    lora_A_has_grad = lora_linear.lora_A.grad is not None
    lora_B_has_grad = lora_linear.lora_B.grad is not None

    print(f"Gradien pada Model Dasar (W_0) : {base_has_grad} (Wajib False / Bebas Update)")
    print(f"Gradien pada Adaptor LoRA A    : {lora_A_has_grad} (Norm: {lora_linear.lora_A.grad.norm().item():.4f})")
    print(f"Gradien pada Adaptor LoRA B    : {lora_B_has_grad} (Norm: {lora_linear.lora_B.grad.norm().item():.4f})")

    assert not base_has_grad, "Error: Bobot model dasar tidak boleh menerima gradien!"
    assert lora_A_has_grad and lora_B_has_grad, "Error: Adaptor LoRA wajib menerima gradien!"

    print("\n[KESIMPULAN ILMIAH]")
    print("1. LoRA mempertahankan 100% bobot asli model dasar dalam kondisi beku (frozen).")
    print("2. Pelatihan hanya memperbarui matriks dekomposisi rank rendah Delta W = B * A.")
    print("3. Memori optimizer berkurang drastis karena hanya melacak momentum untuk 3.12% parameter.")
    print("4. Selama inferensi produksi, Delta W dapat langsung digabungkan (merged) ke W_0 tanpa overhead latensi.")


if __name__ == "__main__":
    main()

