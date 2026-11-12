"""Symmetric InfoNCE / CLIP Contrastive Loss with Manual Verification.

Verifies exact mathematical equivalence between:
1. Manual InfoNCE cross-entropy over rows (Image->Text) and columns (Text->Image).
2. PyTorch F.cross_entropy implementation on a 3x3 similarity matrix.
Also illustrates how false negatives in a batch affect contrastive gradients.
"""

import numpy as np
import torch
import torch.nn.functional as F


def symmetric_contrastive_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Computes symmetric InfoNCE loss for L2-normalized image and text batch."""
    logits = (img_emb @ txt_emb.T) / temperature
    labels = torch.arange(len(img_emb), device=img_emb.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def manual_infonce_worked_example():
    """Calculates symmetric InfoNCE manually on 3 pairs and checks PyTorch parity."""
    # Suppose 3 L2-normalized pairs with temperature tau = 0.5
    I = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.70710678, 0.70710678],
    ], dtype=np.float32)
    T = np.array([
        [0.9, 0.43588989],   # close to I[0]
        [0.1, 0.99498744],   # close to I[1]
        [0.70710678, 0.70710678],  # identical to I[2]
    ], dtype=np.float32)
    tau = 0.5

    logits_np = (I @ T.T) / tau

    # Manual row-wise softmax cross-entropy
    def cross_entropy_rows(mat):
        losses = []
        for i in range(mat.shape[0]):
            row = mat[i] - np.max(mat[i])
            probs = np.exp(row) / np.sum(np.exp(row))
            losses.append(-np.log(probs[i]))
        return float(np.mean(losses))

    manual_i2t = cross_entropy_rows(logits_np)
    manual_t2i = cross_entropy_rows(logits_np.T)
    manual_total = 0.5 * (manual_i2t + manual_t2i)

    # Verify with PyTorch
    pt_total = float(symmetric_contrastive_loss(torch.tensor(I), torch.tensor(T), temperature=tau).item())
    assert np.isclose(manual_total, pt_total, atol=1e-5)

    print("1. Worked Manual 3x3 Contrastive Matrix Verification (tau=0.5):")
    print("   Cosine Similarity Matrix (I @ T^T):")
    sim = I @ T.T
    for r in range(3):
        print(f"     Img {r}: {[round(float(v), 4) for v in sim[r]]}")
    print(f"   - Manual Image->Text Loss : {manual_i2t:.5f}")
    print(f"   - Manual Text->Image Loss : {manual_t2i:.5f}")
    print(f"   - Symmetric Total Loss    : {manual_total:.5f} (Matches PyTorch {pt_total:.5f}!)")


def main():
    print("=" * 75)
    print("CHAPTER 38: SYMMETRIC INFONCE / CLIP CONTRASTIVE LOSS")
    print("=" * 75)

    manual_infonce_worked_example()
    print("\nNote on Batch Negatives: If a mini-batch contains two pairs with the exact same")
    print("concept (e.g. two 'red circle' samples), standard InfoNCE pushes them apart as")
    print("'false negatives'. Keeping diverse unique concepts per batch stabilizes alignment!")
    print("\n[SUCCESS] Contrastive loss derivation and PyTorch parity verified.")


if __name__ == "__main__":
    main()

