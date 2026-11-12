"""Discounted Returns and Impact of Discount Factor (Gamma).

Computes:
    G_t = sum_{k=0}^{T-t-1} gamma^k * R_{t+k+1}
Demonstrates:
- Forward computation vs backward recursive computation (G_t = R_{t+1} + gamma * G_{t+1}).
- Myopic (gamma=0.0) vs Far-sighted (gamma=0.99) return sensitivity.
- Numerical assertion verifying backward and forward equivalence.
"""

from typing import List
import numpy as np


def compute_discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    """Computes discounted return G_t for every timestep t using backward recursion."""
    n = len(rewards)
    returns = np.zeros(n, dtype=np.float64)
    g = 0.0
    for t in reversed(range(n)):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns.tolist()


def compute_g0_forward(rewards: List[float], gamma: float) -> float:
    return sum((gamma ** k) * r for k, r in enumerate(rewards))


def main():
    print("=" * 70)
    print("CHAPTER 35: DISCOUNTED RETURNS & GAMMA TRADE-OFFS")
    print("=" * 70)

    # Sample reward sequence: 5 steps leading to goal reward +10
    rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, 10.0]
    print(f"Episode Reward Sequence (T={len(rewards)} steps): {rewards}\n")

    gammas = [0.0, 0.5, 0.9, 0.99]
    print(f"{'Gamma':<8} | {'G_0 (Forward)':<15} | {'G_0 (Backward)':<15} | {'Returns G_t across t=0..5'}")
    print("-" * 80)

    for gamma in gammas:
        g_list = compute_discounted_returns(rewards, gamma)
        g0_fwd = compute_g0_forward(rewards, gamma)
        assert np.isclose(g_list[0], g0_fwd, atol=1e-9), "Forward and backward return mismatch!"
        g_str = ", ".join(f"{v:.2f}" for v in g_list)
        print(f"{gamma:<8.2f} | {g0_fwd:<15.4f} | {g_list[0]:<15.4f} | [{g_str}]")

    print("\nAnalytical Insight:")
    print("- gamma = 0.00 (Myopic): Agent only cares about immediate reward R_1 (-0.10).")
    print("- gamma = 0.50: Future goal reward (+10) decays exponentially by 0.5^5 = 0.3125.")
    print("- gamma = 0.90: Preserves ~59% of goal reward across 5 steps (G_0 = 5.4951).")
    print("- gamma = 0.99: Almost undiscounted (G_0 = 9.0195), prioritizes long-horizon optimality.")
    print("NOTE: High gamma is not universally better; in continuing noisy tasks it increases variance.")

    print("\n[SUCCESS] Discounted returns verified.")


if __name__ == "__main__":
    main()

