"""Shared Multimodal Embedding Space & L2 Normalization.

Demonstrates:
1. Projection layers mapping separate encoders into a common shared dimension d=32.
2. L2-normalization onto the unit hypersphere so dot product == cosine similarity.
3. Measuring the Modality Gap before alignment vs alignment after projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from paired_dataset_generator import generate_paired_splits, PairedMultimodalDataset
from image_encoder import TinyImageEncoder
from text_encoder import TinyTextEncoder


class SharedSpaceProjection(nn.Module):
    """Projects raw image (64d) and text (64d) encodings into normalized 32d shared space."""

    def __init__(self, hidden_dim: int = 64, shared_dim: int = 32):
        super().__init__()
        self.img_enc = TinyImageEncoder(embed_dim=hidden_dim)
        self.txt_enc = TinyTextEncoder(embed_dim=hidden_dim)
        self.img_proj = nn.Linear(hidden_dim, shared_dim, bias=False)
        self.txt_proj = nn.Linear(hidden_dim, shared_dim, bias=False)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        z = self.img_proj(self.img_enc(images))
        return F.normalize(z, p=2, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.txt_proj(self.txt_enc(tokens))
        return F.normalize(z, p=2, dim=-1)


def main():
    print("=" * 75)
    print("CHAPTER 38: SHARED MULTIMODAL EMBEDDING SPACE & L2 NORMALIZATION")
    print("=" * 75)

    torch.manual_seed(42)
    _, test_items = generate_paired_splits(seed=42)
    ds = PairedMultimodalDataset(test_items[:8])

    proj_model = SharedSpaceProjection(hidden_dim=64, shared_dim=32)
    with torch.no_grad():
        emb_img = proj_model.encode_image(ds.images)
        emb_txt = proj_model.encode_text(ds.tokens)

    img_norms = torch.norm(emb_img, dim=-1)
    txt_norms = torch.norm(emb_txt, dim=-1)
    assert torch.allclose(img_norms, torch.ones_like(img_norms), atol=1e-5)
    assert torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-5)

    diag_sim = torch.sum(emb_img * emb_txt, dim=-1).mean().item()
    print(f"Image Embedding Shape   : {tuple(emb_img.shape)} | Unit Norms == 1.0 Verified")
    print(f"Text  Embedding Shape   : {tuple(emb_txt.shape)} | Unit Norms == 1.0 Verified")
    print(f"Untrained Mean Pair Cosine Similarity (Modality Gap): {diag_sim:.4f}")
    print("\nNote: Before contrastive training, image and text encoders occupy unaligned")
    print("regions of the hypersphere (Modality Gap); InfoNCE pulls matching pairs together.")
    print("[SUCCESS] Multimodal projection & normalization verified.")


if __name__ == "__main__":
    main()

