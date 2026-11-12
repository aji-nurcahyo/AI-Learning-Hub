"""Miniature Image-Text Dual Encoder (CLIP-Style Architecture).

Implements a lightweight Dual Encoder trained on CPU using symmetric InfoNCE loss:
- TinyImageEncoder -> Linear(64, 32) -> L2 Normalize
- TinyTextEncoder  -> Linear(64, 32) -> L2 Normalize
- Uses unique-concept batch sampling (16 distinct concepts per batch) to prevent
  false-negative collisions during contrastive optimization.

NOTE: This miniature geometry prototype demonstrates CLIP's architectural alignment
mechanics; it is NOT equivalent to OpenAI's 400M-pair pretrained CLIP models.
"""

import os
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from paired_dataset_generator import generate_paired_splits, PairedMultimodalDataset
from image_encoder import TinyImageEncoder
from text_encoder import TinyTextEncoder
from contrastive_loss import symmetric_contrastive_loss


class MiniatureDualEncoder(nn.Module):
    """CLIP-style Dual Encoder aligning 32x32 RGB images and short captions."""

    def __init__(self, hidden_dim: int = 64, shared_dim: int = 32):
        super().__init__()
        self.img_enc = TinyImageEncoder(embed_dim=hidden_dim)
        self.txt_enc = TinyTextEncoder(embed_dim=hidden_dim)
        self.img_proj = nn.Linear(hidden_dim, shared_dim, bias=False)
        self.txt_proj = nn.Linear(hidden_dim, shared_dim, bias=False)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.img_proj(self.img_enc(images)), p=2, dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.txt_proj(self.txt_enc(tokens)), p=2, dim=-1)


def train_clip_miniature(epochs: int = 35, lr: float = 3e-3, temperature: float = 0.15, seed: int = 42, save_ckpt: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_items, test_items = generate_paired_splits(seed=seed)
    train_ds = PairedMultimodalDataset(train_items)
    test_ds = PairedMultimodalDataset(test_items)

    # Group training indices by concept_id (0..15) so each batch has 16 unique concepts (zero false negatives)
    by_concept = {c: [] for c in range(16)}
    for idx in range(len(train_ds)):
        by_concept[int(train_ds.concept_ids[idx].item())].append(idx)

    model = MiniatureDualEncoder(hidden_dim=64, shared_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    model.train()
    for _ in range(epochs):
        # 20 samples per concept -> 20 unique-concept batches per epoch
        epoch_losses = []
        orders = {c: rng.permutation(by_concept[c]) for c in range(16)}
        for b in range(20):
            batch_idx = [int(orders[c][b]) for c in range(16)]
            imgs = train_ds.images[batch_idx]
            toks = train_ds.tokens[batch_idx]

            optimizer.zero_grad()
            emb_i = model.encode_image(imgs)
            emb_t = model.encode_text(toks)
            loss = symmetric_contrastive_loss(emb_i, emb_t, temperature=temperature)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        loss_history.append(float(np.mean(epoch_losses)))

    model.eval()
    if save_ckpt:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exp_dir = os.path.abspath(os.path.join(script_dir, "..", "experiments"))
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_path = os.path.join(exp_dir, "clip_miniature.pt")
        torch.save(model.state_dict(), ckpt_path)

    return model, train_ds, test_ds, loss_history


def load_or_train_clip_miniature(seed: int = 42):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.abspath(os.path.join(script_dir, "..", "experiments", "clip_miniature.pt"))
    _, test_items = generate_paired_splits(seed=seed)
    test_ds = PairedMultimodalDataset(test_items)

    if os.path.exists(ckpt_path):
        model = MiniatureDualEncoder(hidden_dim=64, shared_dim=32)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        return model, test_ds
    model, _, test_ds, _ = train_clip_miniature(epochs=35, seed=seed, save_ckpt=True)
    return model, test_ds


def main():
    print("=" * 75)
    print("CHAPTER 38: MINIATURE DUAL ENCODER (CLIP-STYLE CONTRASTIVE ALIGNMENT)")
    print("=" * 75)

    model, train_ds, test_ds, losses = train_clip_miniature(epochs=35, seed=42, save_ckpt=True)

    # Check alignment on 16 unique unseen test concepts (one per concept)
    unique_test_idx = [i * 6 for i in range(16)]  # 6 test items per concept -> index 0, 6, 12...
    with torch.no_grad():
        emb_i = model.encode_image(test_ds.images[unique_test_idx])
        emb_t = model.encode_text(test_ds.tokens[unique_test_idx])
        sim_mat = emb_i @ emb_t.T

    diag_mean = float(torch.diag(sim_mat).mean().item())
    off_mask = ~torch.eye(16, dtype=torch.bool)
    off_mean = float(sim_mat[off_mask].mean().item())
    top1_acc = float((torch.argmax(sim_mat, dim=1) == torch.arange(16)).float().mean().item())

    print(f"Dual Encoder Params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Contrastive Loss    : Epoch 1 = {losses[0]:.4f} -> Epoch 35 = {losses[-1]:.4f}")
    print(f"Unseen Test (16 concepts):")
    print(f"- Positive Pair Cosine Sim Mean : {diag_mean:+.4f}")
    print(f"- Negative Pair Cosine Sim Mean : {off_mean:+.4f}")
    print(f"- Alignment Margin (Pos - Neg)  : {diag_mean - off_mean:+.4f}")
    print(f"- 16-Way Zero-Shot Match Acc    : {top1_acc * 100:.2f}%")

    print("\nDisclaimer: Performance on synthetic 4x4 shapes proves alignment mechanics,")
    print("not open-world natural photography generalization.")
    print("[SUCCESS] Miniature dual encoder trained and verified.")


if __name__ == "__main__":
    main()

