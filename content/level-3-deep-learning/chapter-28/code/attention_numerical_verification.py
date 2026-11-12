"""
Chapter 28 - Attention & Transformer Architecture
Script 13: Rigorous Numerical Verification (Scratch NumPy vs PyTorch MultiheadAttention)

Melakukan pengujian kesetaraan numerik ketat antara implementasi Scaled Dot-Product
dan Multi-Head Attention dari dasar (NumPy) terhadap modul resmi PyTorch nn.MultiheadAttention:
- Menyamakan bobot proyeksi linier (W_Q, W_K, W_V, W_O) secara deterministik.
- Menonaktifkan dropout (dropout=0.0) dan memastikan konvensi scaling 1/sqrt(d_k) identik.
- Menguji kasus unmasked dan causal masked attention.
- Memverifikasi bahwa deviasi numerik absolut maksimum < 1e-6 (floating-point tolerance).
- Menyimpan hasil audit komputasi ke 'experiments/results_ch28.json'.
"""

import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def numpy_multi_head_attention_exact(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    b_q: np.ndarray,
    b_k: np.ndarray,
    b_v: np.ndarray,
    b_o: np.ndarray,
    num_heads: int,
    causal: bool = False,
):
    """Implementasi MHA NumPy dengan penyertaan bias persis setara PyTorch in_proj & out_proj."""
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads

    # 1. Proyeksi Linier dengan Bias
    q = np.matmul(x, w_q.T) + b_q  # (batch, seq, d_model)
    k = np.matmul(x, w_k.T) + b_k
    v = np.matmul(x, w_v.T) + b_v

    # 2. Reshape & Transpose ke (batch, num_heads, seq_len, d_k)
    q = q.reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)
    k = k.reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)
    v = v.reshape(batch_size, seq_len, num_heads, d_k).swapaxes(1, 2)

    # 3. Scaled Dot-Product
    scores = np.matmul(q, k.swapaxes(-2, -1)) / math.sqrt(d_k)

    if causal:
        # Causal mask: segitiga atas bernilai -inf
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores = scores + mask

    # Softmax stabil
    scores_max = np.max(scores, axis=-1, keepdims=True)
    # Jika baris berisi -inf (tidak terjadi pada segitiga bawah normal), hindari NaN
    exp_s = np.exp(scores - scores_max)
    attn_weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    # 4. Agregasi Context
    context = np.matmul(attn_weights, v)  # (batch, heads, seq, d_k)

    # 5. Gabungkan heads: (batch, seq, d_model)
    concat_context = context.swapaxes(1, 2).reshape(batch_size, seq_len, d_model)

    # 6. Proyeksi Luaran Akhir dengan Bias
    output = np.matmul(concat_context, w_o.T) + b_o
    return output, attn_weights


def main():
    print("=== Chapter 28: Rigorous Numerical Verification (NumPy vs PyTorch) ===")

    d_model = 16
    num_heads = 4
    batch_size = 2
    seq_len = 5

    # 1. Inisialisasi PyTorch nn.MultiheadAttention
    torch.manual_seed(42)
    mha_torch = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=0.0,
        bias=True,
        batch_first=True,
    )
    mha_torch.eval()

    # 2. Ekstrak Bobot dan Bias PyTorch ke NumPy
    # in_proj_weight: (3*d_model, d_model) -> pecah ke W_q, W_k, W_v
    in_w = mha_torch.in_proj_weight.detach().numpy()
    in_b = mha_torch.in_proj_bias.detach().numpy()
    out_w = mha_torch.out_proj.weight.detach().numpy()
    out_b = mha_torch.out_proj.bias.detach().numpy()

    w_q, w_k, w_v = in_w[:d_model], in_w[d_model:2*d_model], in_w[2*d_model:]
    b_q, b_k, b_v = in_b[:d_model], in_b[d_model:2*d_model], in_b[2*d_model:]

    # Tensor Masukan Identik
    rng = np.random.default_rng(42)
    x_numpy = rng.standard_normal((batch_size, seq_len, d_model), dtype=np.float32)
    x_torch = torch.from_numpy(x_numpy)

    # 3. Uji Kasus A: Unmasked Attention
    print("\n--- Uji Kasus A: Unmasked Attention ---")
    out_np_a, weights_np_a = numpy_multi_head_attention_exact(
        x_numpy, w_q, w_k, w_v, out_w, b_q, b_k, b_v, out_b, num_heads, causal=False
    )

    with torch.no_grad():
        out_torch_a, weights_torch_a = mha_torch(
            x_torch, x_torch, x_torch, need_weights=True, average_attn_weights=False
        )

    out_torch_np_a = out_torch_a.numpy()
    weights_torch_np_a = weights_torch_a.numpy()

    dev_out_a = float(np.max(np.abs(out_np_a - out_torch_np_a)))
    dev_weights_a = float(np.max(np.abs(weights_np_a - weights_torch_np_a)))

    print(f"  Deviasi Maksimum Output Tensor : {dev_out_a:.2e}")
    print(f"  Deviasi Maksimum Bobot Atensi  : {dev_weights_a:.2e}")
    assert dev_out_a < 1e-6, f"Deviasi output A ({dev_out_a}) melampaui toleransi 1e-6!"
    assert dev_weights_a < 1e-6, f"Deviasi weights A ({dev_weights_a}) melampaui toleransi 1e-6!"
    print("  -> Kasus Unmasked: MATCH IDENTIK 100%!")

    # 4. Uji Kasus B: Causal Masked Attention (Autoregressive Decoder)
    print("\n--- Uji Kasus B: Causal Masked Attention ---")
    out_np_b, weights_np_b = numpy_multi_head_attention_exact(
        x_numpy, w_q, w_k, w_v, out_w, b_q, b_k, b_v, out_b, num_heads, causal=True
    )

    causal_mask_torch = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    with torch.no_grad():
        out_torch_b, weights_torch_b = mha_torch(
            x_torch, x_torch, x_torch, attn_mask=causal_mask_torch, need_weights=True, average_attn_weights=False
        )

    out_torch_np_b = out_torch_b.numpy()
    weights_torch_np_b = weights_torch_b.numpy()

    dev_out_b = float(np.max(np.abs(out_np_b - out_torch_np_b)))
    dev_weights_b = float(np.max(np.abs(weights_np_b - weights_torch_np_b)))

    print(f"  Deviasi Maksimum Output Tensor : {dev_out_b:.2e}")
    print(f"  Deviasi Maksimum Bobot Atensi  : {dev_weights_b:.2e}")
    assert dev_out_b < 1e-6, f"Deviasi output B ({dev_out_b}) melampaui toleransi 1e-6!"
    assert dev_weights_b < 1e-6, f"Deviasi weights B ({dev_weights_b}) melampaui toleransi 1e-6!"
    print("  -> Kasus Causal Masked: MATCH IDENTIK 100%!")

    # 5. Rekam Hasil Eksperimen ke JSON
    verification_results = {
        "status": "PASSED",
        "d_model": d_model,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "unmasked_output_max_deviation": dev_out_a,
        "unmasked_weights_max_deviation": dev_weights_a,
        "causal_output_max_deviation": dev_out_b,
        "causal_weights_max_deviation": dev_weights_b,
        "tolerance": 1e-6,
        "conclusion": "Perhitungan Multi-Head Attention NumPy murni terbukti identik sempurna secara numerik terhadap PyTorch nn.MultiheadAttention."
    }

    out_dir = Path(__file__).resolve().parent.parent / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results_ch28.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(verification_results, f, indent=2)

    print(f"\nLog hasil verifikasi tersimpan di: {out_file}")
    print("\nStatus: attention_numerical_verification.py berhasil dieksekusi 100%!")


if __name__ == "__main__":
    main()

