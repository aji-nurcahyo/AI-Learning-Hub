"""Lightweight Text Encoder for Multimodal Systems.

Maps discrete caption token sequences (B, L) using an Embedding layer + projection:
1. forward_tokens(x) -> sequence representations (B, L, embed_dim) for Cross-Attention.
2. forward(x) -> masked mean-pooled sentence representation (B, embed_dim).
"""

import torch
import torch.nn as nn
from paired_dataset_generator import VOCAB, generate_paired_splits, PairedMultimodalDataset


class TinyTextEncoder(nn.Module):
    """Compact token embedding + linear encoder mapping (B, L) -> (B, embed_dim)."""

    def __init__(self, vocab_size: int = len(VOCAB), embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns sequence token embeddings: (B, L, embed_dim)."""
        emb = self.embedding(tokens)
        return self.proj(emb)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns masked average pooled text vector: (B, embed_dim)."""
        seq = self.forward_tokens(tokens)
        mask = (tokens != 0).unsqueeze(-1).float()
        summed = torch.sum(seq * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / counts


def main():
    print("=" * 75)
    print("CHAPTER 38: LIGHTWEIGHT TEXT ENCODER VERIFICATION")
    print("=" * 75)

    _, test_items = generate_paired_splits(seed=42)
    ds = PairedMultimodalDataset(test_items[:4])
    encoder = TinyTextEncoder(embed_dim=64)

    with torch.no_grad():
        seq_tokens = encoder.forward_tokens(ds.tokens)
        pooled_vec = encoder(ds.tokens)

    params = sum(p.numel() for p in encoder.parameters())
    print(f"TinyTextEncoder Parameters  : {params:,}")
    print(f"Input Tokens Batch Shape    : {tuple(ds.tokens.shape)}")
    print(f"Token Embeddings Shape      : {tuple(seq_tokens.shape)}")
    print(f"Pooled Sentence Vector Shape: {tuple(pooled_vec.shape)}")
    print("\n[SUCCESS] TinyTextEncoder verified.")


if __name__ == "__main__":
    main()

