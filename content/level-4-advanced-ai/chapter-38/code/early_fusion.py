"""Early Fusion Architecture (Feature-Level Concatenation).

Concatenates visual and text representations [z_img; z_txt] into a joint
embedding vector (B, 128) passed into a unified classification head (16 concepts).
Tests unseen test accuracy and missing-modality degradation.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from paired_dataset_generator import generate_paired_splits, PairedMultimodalDataset
from image_encoder import TinyImageEncoder
from text_encoder import TinyTextEncoder


class EarlyFusionClassifier(nn.Module):
    """Fuses image and text features by early concatenation prior to joint MLP."""

    def __init__(self, embed_dim: int = 64, num_classes: int = 16):
        super().__init__()
        self.img_enc = TinyImageEncoder(embed_dim=embed_dim)
        self.txt_enc = TinyTextEncoder(embed_dim=embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        z_img = self.img_enc(images)
        z_txt = self.txt_enc(tokens)
        z_joint = torch.cat([z_img, z_txt], dim=-1)  # (B, 128)
        return self.fusion_mlp(z_joint)


def train_and_evaluate_early_fusion(epochs: int = 20, seed: int = 42):
    torch.manual_seed(seed)
    train_items, test_items = generate_paired_splits(seed=seed)
    train_ds = PairedMultimodalDataset(train_items)
    test_ds = PairedMultimodalDataset(test_items)

    model = EarlyFusionClassifier()
    opt = optim.Adam(model.parameters(), lr=2e-3)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(train_ds))
        for i in range(0, len(train_ds), 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            logits = model(train_ds.images[idx], train_ds.tokens[idx])
            crit(logits, train_ds.concept_ids[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        acc_both = float((torch.argmax(model(test_ds.images, test_ds.tokens), dim=1) == test_ds.concept_ids).float().mean())
        acc_no_txt = float((torch.argmax(model(test_ds.images, torch.zeros_like(test_ds.tokens)), dim=1) == test_ds.concept_ids).float().mean())
        acc_no_img = float((torch.argmax(model(torch.zeros_like(test_ds.images), test_ds.tokens), dim=1) == test_ds.concept_ids).float().mean())

    return model, round(acc_both * 100, 2), round(acc_no_txt * 100, 2), round(acc_no_img * 100, 2)


def main():
    print("=" * 75)
    print("CHAPTER 38: EARLY FUSION (FEATURE-LEVEL CONCATENATION)")
    print("=" * 75)

    _, acc_both, acc_no_txt, acc_no_img = train_and_evaluate_early_fusion(epochs=20, seed=42)
    print(f"Concatenated Joint Feature Vector: [z_img (64) ; z_txt (64)] -> 128-dim MLP")
    print(f"- Test Accuracy (Both Modalities Present) : {acc_both:.2f}%")
    print(f"- Test Accuracy (Missing Text Modality)   : {acc_no_txt:.2f}%")
    print(f"- Test Accuracy (Missing Image Modality)  : {acc_no_img:.2f}%")

    print("\nNote: Because concatenated MLP weights couple both modalities tightly during training,")
    print("missing one modality at test time can cause severe degradation if no modality dropout was trained.")
    print("\n[SUCCESS] Early Fusion evaluation completed.")


if __name__ == "__main__":
    main()

