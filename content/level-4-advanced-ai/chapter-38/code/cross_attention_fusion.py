"""Cross-Attention Fusion (Query=Text Tokens, Key/Value=Image Spatial Patches).

Demonstrates:
1. Worked tensor verification in pure NumPy vs PyTorch Scaled Dot-Product Cross-Attention:
   Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V
2. Clear origin of Q (Text modality, shape L_t x d_k) and K, V (Image patches, shape N_p x d_k).
3. Verifies row-wise normalization sum_j alpha_{i, j} == 1.0.
4. Full PyTorch CrossAttentionFusionClassifier trained on paired geometric images & text.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from paired_dataset_generator import generate_paired_splits, PairedMultimodalDataset
from image_encoder import TinyImageEncoder
from text_encoder import TinyTextEncoder


def numpy_cross_attention_worked_example():
    """Exact small-tensor worked verification contrasting NumPy vs PyTorch."""
    # Q: 2 text tokens x d_k=4,  K/V: 3 image patches x d_k=4
    Q_np = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    K_np = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.5, 0.5, 0.0, 0.0]], dtype=np.float32)
    V_np = np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0], [1.0, 1.0, 5.0, 5.0]], dtype=np.float32)

    d_k = Q_np.shape[-1]  # 4 -> sqrt(4) = 2.0
    scores_np = (Q_np @ K_np.T) / np.sqrt(d_k)
    exp_scores = np.exp(scores_np - np.max(scores_np, axis=-1, keepdims=True))
    weights_np = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    out_np = weights_np @ V_np

    # Verify sum of weights == 1.0 for every query token
    assert np.allclose(weights_np.sum(axis=-1), np.ones(2), atol=1e-6)

    # PyTorch exact parity verification
    out_pt, weights_pt = torch_scaled_cross_attention(
        torch.tensor(Q_np).unsqueeze(0),
        torch.tensor(K_np).unsqueeze(0),
        torch.tensor(V_np).unsqueeze(0),
    )
    assert np.allclose(out_np, out_pt.squeeze(0).numpy(), atol=1e-5)

    print("1. Worked Tensor Cross-Attention Verification (NumPy == PyTorch Parity):")
    print(f"   Q (Text Tokens)  shape: {Q_np.shape}   | K, V (Image Patches) shape: {K_np.shape}")
    print(f"   Attention Weights (2x3): {np.round(weights_np, 4).tolist()}")
    print(f"   Row Sums == 1.0 Check  : {weights_np.sum(axis=-1).tolist()} (PASS)")
    print(f"   Fused Output     shape: {out_np.shape}   (PASS: Not mere concatenation!)\n")


def torch_scaled_cross_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    d_k = Q.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)
    context = torch.bmm(attn_weights, V)
    return context, attn_weights


class CrossAttentionFusionClassifier(nn.Module):
    """Cross-Attention where Text Queries attend over 16 Image Spatial Patches."""

    def __init__(self, embed_dim: int = 64, num_classes: int = 16):
        super().__init__()
        self.img_enc = TinyImageEncoder(embed_dim=embed_dim)
        self.txt_enc = TinyTextEncoder(embed_dim=embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        img_patches = self.img_enc.forward_spatial(images)   # (B, 16, D)
        txt_seq = self.txt_enc.forward_tokens(tokens)        # (B, L_t, D)

        Q = self.W_q(txt_seq)      # (B, L_t, D)
        K = self.W_k(img_patches)  # (B, 16, D)
        V = self.W_v(img_patches)  # (B, 16, D)

        attended_visual, _ = torch_scaled_cross_attention(Q, K, V)  # (B, L_t, D)
        # Pool across text token positions and combine with text query summary
        pooled_attn = attended_visual.mean(dim=1)
        pooled_txt = txt_seq.mean(dim=1)
        return self.head(torch.cat([pooled_attn, pooled_txt], dim=-1))


def train_and_evaluate_cross_attention(epochs: int = 20, seed: int = 42):
    torch.manual_seed(seed)
    train_items, test_items = generate_paired_splits(seed=seed)
    train_ds = PairedMultimodalDataset(train_items)
    test_ds = PairedMultimodalDataset(test_items)

    model = CrossAttentionFusionClassifier()
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
    print("CHAPTER 38: CROSS-ATTENTION MULTIMODAL FUSION")
    print("=" * 75)

    numpy_cross_attention_worked_example()
    _, acc_both, acc_no_txt, acc_no_img = train_and_evaluate_cross_attention(epochs=20, seed=42)

    print("2. CrossAttentionFusionClassifier Evaluation (Q=Text, K/V=Image Patches):")
    print(f"   - Test Accuracy (Both Modalities Present) : {acc_both:.2f}%")
    print(f"   - Test Accuracy (Missing Text Modality)   : {acc_no_txt:.2f}%")
    print(f"   - Test Accuracy (Missing Image Modality)  : {acc_no_img:.2f}%")
    print("\n[SUCCESS] Cross-Attention fusion and worked example verified.")


if __name__ == "__main__":
    main()

