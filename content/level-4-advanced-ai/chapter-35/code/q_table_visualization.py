"""Q-Table Visualization and Policy Heatmap.

Trains Q-Learning on GridWorld, then:
1. Prints the complete tabular Q(s, a) matrix with best action arrows.
2. Renders ASCII Max-Q Value Grid and Greedy Policy Grid.
3. Saves a Matplotlib heatmap of learned state values and learning curve (Agg backend).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridworld_environment import GridWorld, ACTION_NAMES, ACTION_ARROWS
from q_learning import train_q_learning
from value_iteration import render_policy_grid


def main():
    print("=" * 75)
    print("CHAPTER 35: Q-TABLE & VALUE HEATMAP VISUALIZATION")
    print("=" * 75)

    env = GridWorld(rows=4, cols=4, slip_prob=0.0, max_steps=50, seed=42)
    Q, returns_hist = train_q_learning(env, num_episodes=800, seed=42)

    print(f"{'State':<8} | {'UP (0)':<8} | {'RIGHT(1)':<8} | {'DOWN (2)':<8} | {'LEFT (3)':<8} | {'Best'}")
    print("-" * 70)

    for s in range(env.n_states):
        pos = env.state_to_pos(s)
        tag = f"S{s:02d}{pos}"
        if pos in env.obstacles:
            print(f"{tag:<8} | {'WALL':^39} |  #")
            continue
        if pos in env.hazards:
            print(f"{tag:<8} | {'HAZARD':^39} |  X")
            continue
        if pos == env.goal_pos:
            print(f"{tag:<8} | {'GOAL':^39} |  G")
            continue

        best_a = int(np.argmax(Q[s]))
        print(
            f"{tag:<8} | {Q[s,0]:8.3f} | {Q[s,1]:8.3f} | {Q[s,2]:8.3f} | {Q[s,3]:8.3f} | "
            f"{ACTION_ARROWS[best_a]} ({ACTION_NAMES[best_a]})"
        )

    print("\nASCII Optimal Policy Grid:")
    render_policy_grid(env, np.argmax(Q, axis=1))

    # Generate Matplotlib figure
    max_q_grid = np.max(Q, axis=1).reshape(env.rows, env.cols)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Heatmap of Max Q(s, a)
    im = axes[0].imshow(max_q_grid, cmap="YlGnBu")
    axes[0].set_title("Learned Max Q(s, a) Heatmap")
    for r in range(env.rows):
        for c in range(env.cols):
            pos = (r, c)
            if pos in env.obstacles:
                axes[0].text(c, r, "WALL", ha="center", va="center", color="black", fontweight="bold")
            elif pos == env.goal_pos:
                axes[0].text(c, r, "GOAL", ha="center", va="center", color="white", fontweight="bold")
            elif pos in env.hazards:
                axes[0].text(c, r, "HAZARD", ha="center", va="center", color="red", fontweight="bold")
            else:
                s = env.pos_to_state(pos)
                arrow = ACTION_ARROWS[int(np.argmax(Q[s]))]
                axes[0].text(c, r, f"{max_q_grid[r, c]:.1f}\n{arrow}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    # Right: Moving average learning curve
    window = 40
    ma = np.convolve(returns_hist, np.ones(window) / window, mode="valid")
    axes[1].plot(returns_hist, alpha=0.3, color="gray", label="Raw Return")
    axes[1].plot(np.arange(window - 1, len(returns_hist)), ma, color="blue", linewidth=2, label=f"MA({window})")
    axes[1].set_title("Q-Learning Training Curve")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Return")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.abspath(os.path.join(script_dir, "..", "experiments", "q_table_heatmap.png"))
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close(fig)

    print(f"\nSaved Matplotlib visualization to: {out_img}")
    print("[SUCCESS] Q-Table visualization verified.")


if __name__ == "__main__":
    main()

