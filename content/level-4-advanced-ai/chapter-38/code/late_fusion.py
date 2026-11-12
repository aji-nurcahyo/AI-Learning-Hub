"""Late Fusion Architecture (Decision-Level Weighted Ensemble).

Trains independent unimodal ImageClassifier and TextClassifier branches,
then combines predicted probability distributions via weighted fusion:
   P_fused(y) = w * P_img(y|x_img) + (1 - w) * P_txt(y|x_txt)
Demonstrates graceful fallback resilience when one modality is completely missing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from paired_dataset_generator import generate_paired_splits, PairedMultimodalDataset
from image_encoder import TinyImageEncoder
from text_encoder import TinyTextEncoder


class LateFusionSystem(nn.Module):
    """Independent unimodal classifiers fused only at decision/probability level."""

    def __init__(self, embed_dim: int = 64, num_classes: int = 16):
        super().__init__()
        self.img_branch = nn.Sequential(
            TinyImageEncoder(embed_dim=embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        self.txt_branch = nn.Sequential(
            TinyTextEncoder(embed_dim=embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def predict_fused(
        self,
        images: torch.Tensor = None,
        tokens: torch.Tensor = None,
        w_img: float = 0.5,
    ) -> torch.Tensor:
        if images is not None and tokens is not None:
            p_img = F.softmax(self.img_branch(images), dim=-1)
            p_txt = F.softmax(self.txt_branch(tokens), dim=-1)
            return w_img * p_img + (1.0 - w_img) * p_txt
        elif images is not None:
            return F.softmax(self.img_branch(images), dim=-1)
        elif tokens is not None:
            return F.softmax(self.txt_branch(tokens), dim=-1)
        else:
            raise ValueError("At least one modality must be provided.")


def train_and_evaluate_late_fusion(epochs: int = 20, seed: int = 42):
    torch.manual_seed(seed)
    train_items, test_items = generate_paired_splits(seed=seed)
    train_ds = PairedMultimodalDataset(train_items)
    test_ds = PairedMultimodalDataset(test_items)

    model = LateFusionSystem()
    opt = optim.Adam(model.parameters(), lr=2e-3)
    crit = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(train_ds))
        for i in range(0, len(train_ds), 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            # Train each unimodal branch independently
            loss_i = crit(model.img_branch(train_ds.images[idx]), train_ds.concept_ids[idx])
            loss_t = crit(model.txt_branch(train_ds.tokens[idx]), train_ds.concept_ids[idx])
            (loss_i + loss_t).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        p_both = model.predict_fused(test_ds.images, test_ds.tokens, w_img=0.5)
        p_img_only = model.predict_fused(images=test_ds.images, tokens=None)
        p_txt_only = model.predict_fused(images=None, tokens=test_ds.tokens)

        acc_both = float((torch.argmax(p_both, dim=1) == test_ds.concept_ids).float().mean())
        acc_img_only = float((torch.argmax(p_img_only, dim=1) == test_ds.concept_ids).float().mean())
        acc_txt_only = float((torch.argmax(p_txt_only, dim=1) == test_ds.concept_ids).float().mean())

    return model, round(acc_both * 100, 2), round(acc_img_only * 100, 2), round(acc_txt_only * 100, 2)


def main():
    print("=" * 75)
    print("CHAPTER 38: LATE FUSION (DECISION-LEVEL ENSEMBLE)")
    print("=" * 75)

    _, acc_both, acc_img, acc_txt = train_and_evaluate_late_fusion(epochs=20, seed=42)
    print(f"Decision Probability Ensemble (0.5 * P_img + 0.5 * P_txt):")
    print(f"- Test Accuracy (Both Modalities Present) : {acc_both:.2f}%")
    print(f"- Test Accuracy (Image Modality Only)     : {acc_img:.2f}% (Text dropped cleanly)")
    print(f"- Test Accuracy (Text Modality Only)      : {acc_txt:.2f}% (Image dropped cleanly)")

    print("\nAdvantage: Late Fusion handles missing modalities gracefully without re-training,")
    print("though it cannot model fine-grained token-to-pixel interactions like Cross-Attention.")
    print("\n[SUCCESS] Late Fusion evaluation completed.")


if __name__ == "__main__":
    main()

