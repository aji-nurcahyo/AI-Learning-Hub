"""Comparative RL Policy Benchmark on GridWorld.

Evaluates 4 distinct policy paradigms across identical evaluation seeds (50 runs):
1. Uniform Random Policy (Baseline).
2. Value Iteration Policy (Model-Based Exact Optimal Reference).
3. Q-Learning Policy (Model-Free Off-Policy TD Control).
4. REINFORCE Policy (Model-Free On-Policy PyTorch Policy Gradient).

Reports Success Rate, Average Return, and Average Episode Length.
"""

import os
from typing import Callable, Dict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridworld_environment import GridWorld
from value_iteration import value_iteration
from q_learning import train_q_learning
from policy_gradient_reinforce import train_reinforce


def evaluate_policy_fn(env: GridWorld, action_fn: Callable[[int, np.random.Generator], int], n_eval: int = 50) -> Dict[str, float]:
    successes = 0
    returns = []
    lengths = []

    for i in range(n_eval):
        eval_rng = np.random.default_rng(5000 + i)
        s = env.reset(seed=5000 + i)
        done = False
        ep_ret = 0.0
        steps = 0

        while not done:
            a = action_fn(s, eval_rng)
            ns, r, done, info = env.step(a)
            ep_ret += r
            steps += 1
            s = ns
            if info["reached_goal"]:
                successes += 1

        returns.append(ep_ret)
        lengths.append(steps)

    return {
        "success_rate_pct": round((successes / n_eval) * 100, 1),
        "avg_return": round(float(np.mean(returns)), 3),
        "avg_length": round(float(np.mean(lengths)), 2),
    }


def main():
    print("=" * 75)
    print("CHAPTER 35: COMPARATIVE RL POLICY EVALUATION BENCHMARK")
    print("=" * 75)

    # Use stochastic slip=0.1 to make policy evaluation rigorous
    env = GridWorld(rows=4, cols=4, slip_prob=0.1, max_steps=40, seed=42)

    print("Training agents (Q-Learning & REINFORCE) and solving DP (Value Iteration)...")
    _, vi_policy, _ = value_iteration(env, gamma=0.9, theta=1e-6)
    Q_table, q_hist = train_q_learning(env, num_episodes=800, alpha=0.1, gamma=0.9, seed=42)
    reinforce_net, rf_hist = train_reinforce(env, num_episodes=600, lr=0.05, gamma=0.9, seed=42)

    # Evaluate all 4 policies
    res_random = evaluate_policy_fn(env, lambda s, rng: int(rng.integers(0, env.n_actions)))
    res_vi = evaluate_policy_fn(env, lambda s, rng: int(vi_policy[s]))
    res_q = evaluate_policy_fn(env, lambda s, rng: int(np.argmax(Q_table[s])))
    with torch.no_grad():
        rf_greedy = np.array([int(torch.argmax(reinforce_net(torch.tensor(s))).item()) for s in range(env.n_states)])
    res_rf = evaluate_policy_fn(env, lambda s, rng: int(rf_greedy[s]))

    benchmarks = [
        ("1. Uniform Random (Baseline)", res_random),
        ("2. Value Iteration (DP Ref)", res_vi),
        ("3. Q-Learning (Model-Free)", res_q),
        ("4. REINFORCE (Policy Grad)", res_rf),
    ]

    print(f"\nEvaluation over 50 identical stochastic runs (slip_prob=0.1):\n")
    print(f"{'Policy Paradigm':<30} | {'Success %':<10} | {'Avg Return':<12} | {'Avg Steps'}")
    print("-" * 70)
    for name, m in benchmarks:
        print(f"{name:<30} | {m['success_rate_pct']:<9.1f}% | {m['avg_return']:<12.3f} | {m['avg_length']}")

    # Plot Learning Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 30
    q_ma = np.convolve(q_hist, np.ones(w) / w, mode="valid")
    rf_ma = np.convolve(rf_hist, np.ones(w) / w, mode="valid")
    ax.plot(np.arange(w - 1, len(q_hist)), q_ma, label="Q-Learning (MA-30)", color="blue")
    ax.plot(np.arange(w - 1, len(rf_hist)), rf_ma, label="REINFORCE (MA-30)", color="green")
    ax.axhline(res_vi["avg_return"], color="red", linestyle="--", label=f"Value Iteration Ref ({res_vi['avg_return']:.2f})")
    ax.set_title("RL Learning Curves Comparison (GridWorld slip=0.1)")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Smoothed Episode Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_img = os.path.abspath(os.path.join(script_dir, "..", "experiments", "rl_policy_comparison.png"))
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close(fig)

    print(f"\nSaved learning curve comparison plot to: {out_img}")
    print("[SUCCESS] Comparative RL benchmark completed.")


if __name__ == "__main__":
    main()

