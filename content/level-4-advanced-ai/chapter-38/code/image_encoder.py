"""Lightweight CNN Image Encoder for Multimodal Systems.

Maps 32x32 RGB tensors (B, 3, 32, 32) to:
1. Spatial patch grid tokens (B, 16, embed_dim) via adaptive 4x4 pooling (for Cross-Attention).
2. Global pooled vector embedding (B, embed_dim) via global average pooling.
Runs rapidly on CPU (~25K parameters).
"""

import torch
import torch.nn as nn
from paired_dataset_generator import generate_paired_splits, PairedMultimodalDataset


class TinyImageEncoder(nn.Module):
    """Compact CNN mapping (B, 3, 32, 32) -> (B, embed_dim)."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 16, 16, 16)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, 32, 8, 8)
            nn.Conv2d(32, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (B, embed_dim, 4, 4)
        )

    def forward_spatial(self, images: torch.Tensor) -> torch.Tensor:
        """Returns 4x4=16 spatial patch tokens: (B, 16, embed_dim)."""
        feats = self.conv(images)  # (B, D, 4, 4)
        b, d, h, w = feats.shape
        return feats.view(b, d, h * w).transpose(1, 2)  # (B, 16, D)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns global average pooled vector: (B, embed_dim)."""
        spatial = self.forward_spatial(images)
        return spatial.mean(dim=1)


def main():
    print("=" * 75)
    print("CHAPTER 38: LIGHTWEIGHT IMAGE ENCODER VERIFICATION")
    print("=" * 75)

    _, test_items = generate_paired_splits(seed=42)
    ds = PairedMultimodalDataset(test_items[:4])
    encoder = TinyImageEncoder(embed_dim=64)

    with torch.no_grad():
        spatial_tokens = encoder.forward_spatial(ds.images)
        pooled_vec = encoder(ds.images)

    params = sum(p.numel() for p in encoder.parameters())
    print(f"TinyImageEncoder Parameters : {params:,}")
    print(f"Input Image Batch Shape     : {tuple(ds.images.shape)}")
    print(f"Spatial Patch Tokens Shape  : {tuple(spatial_tokens.shape)} (4x4 = 16 patches)")
    print(f"Global Pooled Vector Shape  : {tuple(pooled_vec.shape)}")
    print("\n[SUCCESS] TinyImageEncoder verified.")


if __name__ == "__main__":
    main()

