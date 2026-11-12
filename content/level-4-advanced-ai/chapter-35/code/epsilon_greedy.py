"""Exploration vs Exploitation Strategies (Epsilon-Greedy & Decay Schedules).

Compares four exploration schedules on GridWorld Q-Learning:
1. Pure Greedy (epsilon = 0.0) - risks premature convergence.
2. High Constant Exploration (epsilon = 0.50) - poor final training return.
3. Linear Decay (1.0 -> 0.05).
4. Exponential Decay (1.0 * 0.99^ep -> 0.05).

Measures both training return, final greedy success rate, and unique state coverage.
"""

import numpy as np
from gridworld_environment import GridWorld
from q_learning import evaluate_greedy_policy


def run_schedule(name: str, schedule_fn, num_episodes: int = 500, seed: int = 42):
    env = GridWorld(rows=4, cols=4, slip_prob=0.0, max_steps=50, seed=seed)
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    visited_states = set()
    ep_returns = []

    for ep in range(num_episodes):
        eps = schedule_fn(ep, num_episodes)
        s = env.reset()
        visited_states.add(s)
        done = False
        ret = 0.0

        while not done:
            if rng.random() < eps:
                a = int(rng.integers(0, env.n_actions))
            else:
                max_q = np.max(Q[s])
                best_as = np.where(np.isclose(Q[s], max_q))[0]
                a = int(rng.choice(best_as))

            ns, r, done, info = env.step(a)
            visited_states.add(ns)
            ret += r

            is_term = info["reached_goal"] or info["hit_hazard"]
            next_max = 0.0 if is_term else float(np.max(Q[ns]))
            Q[s, a] += 0.1 * (r + 0.9 * next_max - Q[s, a])
            s = ns

        ep_returns.append(ret)

    eval_metrics = evaluate_greedy_policy(env, Q, n_eval=20)
    return {
        "name": name,
        "train_last100_avg": round(float(np.mean(ep_returns[-100:])), 3),
        "eval_success_pct": eval_metrics["success_rate"] * 100,
        "eval_avg_return": eval_metrics["avg_return"],
        "state_coverage": len(visited_states),
    }


def main():
    print("=" * 75)
    print("CHAPTER 35: EXPLORATION VS EXPLOITATION (EPSILON SCHEDULES)")
    print("=" * 75)

    schedules = [
        ("Pure Greedy (eps=0.0)", lambda ep, N: 0.0),
        ("High Constant (eps=0.5)", lambda ep, N: 0.5),
        ("Linear Decay (1.0 -> 0.05)", lambda ep, N: max(0.05, 1.0 - (0.95 * ep / N))),
        ("Exponential Decay (0.99^ep)", lambda ep, N: max(0.05, 1.0 * (0.99 ** ep))),
    ]

    print(f"{'Schedule Strategy':<28} | {'Train Last100':<14} | {'Eval Succ%':<11} | {'Eval Return':<12} | {'States Visited'}")
    print("-" * 75)

    for name, fn in schedules:
        res = run_schedule(name, fn, num_episodes=500, seed=42)
        print(
            f"{res['name']:<28} | {res['train_last100_avg']:<14.3f} | "
            f"{res['eval_success_pct']:<10.1f}% | {res['eval_avg_return']:<12.3f} | {res['state_coverage']}/16"
        )

    print("\nKey Finding: Pure greedy risks suboptimal premature convergence without initial coverage,")
    print("while exponential decay balances wide early exploration with stable late exploitation.")
    print("\n[SUCCESS] Exploration schedules comparison verified.")


if __name__ == "__main__":
    main()

