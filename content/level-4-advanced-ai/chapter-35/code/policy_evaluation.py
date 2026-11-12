"""Iterative Policy Evaluation (Dynamic Programming).

Computes V^pi(s) for a fixed policy pi given complete knowledge of MDP transitions:
   V_{k+1}(s) <- sum_a pi(a|s) sum_{s'} P(s'|s,a) [r + gamma * V_k(s')]
Tracks delta = max_s |V_{k+1}(s) - V_k(s)| until convergence tolerance theta.
"""

from typing import Tuple
import numpy as np
from gridworld_environment import GridWorld


def iterative_policy_evaluation(
    env: GridWorld,
    pi: np.ndarray,
    gamma: float = 0.9,
    theta: float = 1e-5,
    max_iters: int = 1000,
) -> Tuple[np.ndarray, int]:
    """Evaluates policy pi on env, returning converged V and iteration count."""
    V = np.zeros(env.n_states, dtype=np.float64)
    iters = 0

    for k in range(1, max_iters + 1):
        iters = k
        delta = 0.0
        V_new = np.zeros_like(V)

        for s in range(env.n_states):
            pos = env.state_to_pos(s)
            if env.is_terminal(pos) or pos in env.obstacles:
                continue

            v_val = 0.0
            for a in range(env.n_actions):
                if pi[s, a] == 0.0:
                    continue
                for prob, ns, reward, done in env.get_transition_outcomes(s, a):
                    v_val += pi[s, a] * prob * (reward + gamma * (0.0 if done else V[ns]))

            V_new[s] = v_val
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if k in (1, 2, 5, 10) or delta < theta:
            print(f"  Sweep {k:3d} | max_delta = {delta:.6f} | V(start) = {V[env.pos_to_state(env.start_pos)]:7.4f}")

        if delta < theta:
            break

    return V, iters


def main():
    print("=" * 70)
    print("CHAPTER 35: ITERATIVE POLICY EVALUATION (UNIFORM RANDOM POLICY)")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, slip_prob=0.1)
    uniform_pi = np.full((env.n_states, env.n_actions), 0.25)

    print("Evaluating Uniform Random Policy (pi=0.25 each action, gamma=0.9, slip=0.1):")
    V_conv, total_iters = iterative_policy_evaluation(env, uniform_pi, gamma=0.9, theta=1e-5)

    print(f"\nConverged in {total_iters} sweeps.")
    print("Final Value Function Grid V^pi(s):")
    for r in range(env.rows):
        row_str = []
        for c in range(env.cols):
            pos = (r, c)
            if pos in env.obstacles:
                row_str.append("  WALL  ")
            elif pos in env.hazards:
                row_str.append(" HAZARD ")
            elif pos == env.goal_pos:
                row_str.append("  GOAL  ")
            else:
                row_str.append(f"{V_conv[env.pos_to_state(pos)]:7.3f} ")
        print("  " + "|".join(row_str))

    print("\n[SUCCESS] Policy Evaluation completed.")


if __name__ == "__main__":
    main()

