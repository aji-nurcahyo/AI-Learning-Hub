"""
Chapter 29 - Large Language Models & Generative AI
Script 14: Tiny Autoregressive Transformer Generation
Implementasi mandiri model Decoder-Only Transformer mini (arsitektur ala GPT) menggunakan PyTorch pada CPU:
Pelatihan cepat pada korpus kecil, pembangkitan autoregresif, serta perbandingan Greedy vs Sampling.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TinyCausalSelfAttention(nn.Module):
    """Causal Self-Attention dengan Masker Segitiga Bawah."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot product
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TinyTransformerBlock(nn.Module):
    """Satu blok Transformer Decoder: Causal Attention + LayerNorm + FFN."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyCausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Model Mini Decoder-Only Transformer untuk Demonstrasi Pembangkitan Teks."""
    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 2, n_layers: int = 2, max_seq_len: int = 32):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, n_heads, d_ff=d_model * 2) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Batasi ke konteks maksimum
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            # Ambil logit dari posisi token terakhir
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


def main():
    print("=" * 70)
    print("CHAPTER 29: TINY TRANSFORMER GENERATION (DECODER-ONLY)")
    print("=" * 70)

    # Korpus Mini Bahasa Indonesia
    corpus = [
        "kecerdasan buatan membantu manusia belajar",
        "model bahasa menghasilkan teks cerdas",
        "pembelajaran mesin memproses data cepat",
        "jaringan saraf meniru cara kerja otak",
        "komputer cerdas memahami bahasa manusia"
    ]

    # Bangun Kosakata
    tokens = [w for sentence in corpus for w in sentence.split()]
    vocab = ["<pad>", "<eos>"] + sorted(list(set(tokens)))
    token2id = {w: i for i, w in enumerate(vocab)}
    id2token = {i: w for i, w in enumerate(vocab)}
    V = len(vocab)

    print(f"Jumlah Kalimat Latih: {len(corpus)}")
    print(f"Ukuran Kosakata (V) : {V} kata")

    # Inisialisasi Model TinyGPT
    torch.manual_seed(42)
    d_model = 32
    model = TinyGPT(vocab_size=V, d_model=d_model, n_heads=2, n_layers=2, max_seq_len=16)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Konfigurasi Model   : d_model={d_model}, n_heads=2, n_layers=2")
    print(f"Total Parameter     : {total_params:,} bobot (Sangat Ringan, CPU-Friendly)")

    # Pelatihan Cepat Mini (50 Iterasi Epoch pada CPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Siapkan data latih (Next-Token Prediction Target)
    data = []
    for sentence in corpus:
        ids = [token2id[w] for w in sentence.split()] + [token2id["<eos>"]]
        data.append(ids)

    print("\n[PELATIHAN SINGKAT MODEL PADA CPU]")
    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, 41):
        total_loss = 0.0
        for seq in data:
            inputs = torch.tensor([seq[:-1]], dtype=torch.long)
            targets = torch.tensor([seq[1:]], dtype=torch.long)
            logits = model(inputs)
            loss = criterion(logits.view(-1, V), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / len(data)
            print(f"  Epoch {epoch:02d}/40 | Loss: {avg_loss:.4f} | Perplexity: {math.exp(avg_loss):.2f}")
    train_time = time.perf_counter() - t0
    print(f"Pelatihan Selesai dalam: {train_time:.3f} detik pada CPU.")

    # Pembangkitan Teks Autoregresif
    model.eval()
    prompt_words = ["kecerdasan", "buatan"]
    prompt_ids = torch.tensor([[token2id[w] for w in prompt_words]], dtype=torch.long)

    print(f"\n[DEMONSTRASI GENERASI TEKS AUTOREGRESIF]")
    print(f"Prompt Masukan: '{' '.join(prompt_words)}'")

    # Skenario 1: Suhu Dingin (T = 0.3, Top-k = 2)
    gen_ids_cold = model.generate(prompt_ids.clone(), max_new_tokens=5, temperature=0.3, top_k=2)
    gen_text_cold = " ".join([id2token[i.item()] for i in gen_ids_cold[0]])
    print(f"1. Suhu Rendah (T=0.3, k=2): '{gen_text_cold}'")

    # Skenario 2: Suhu Standar (T = 1.0, Top-k = 5)
    gen_ids_std = model.generate(prompt_ids.clone(), max_new_tokens=5, temperature=1.0, top_k=5)
    gen_text_std = " ".join([id2token[i.item()] for i in gen_ids_std[0]])
    print(f"2. Suhu Netral (T=1.0, k=5): '{gen_text_std}'")

    print("\n[KESIMPULAN ILMIAH]")
    print("1. TinyGPT berhasil memodelkan transisi bahasa alami melalui lapisan Causal Self-Attention.")
    print("2. Model menghasilkan token berikutnya secara autoregresif di mana setiap luaran menjadi masukan berikutnya.")
    print("3. Seluruh inferensi dan pelatihan berlangsung secara instan pada CPU komputer standar.")


if __name__ == "__main__":
    main()

