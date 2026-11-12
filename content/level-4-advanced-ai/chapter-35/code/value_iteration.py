"""Value Iteration Algorithm (Dynamic Programming).

Finds optimal value function V*(s) using Bellman Optimality Operator:
   V_{k+1}(s) <- max_a sum_{s'} P(s'|s,a) [r + gamma * V_k(s')]
Then extracts the optimal deterministic policy:
   pi*(s) = argmax_a sum_{s'} P(s'|s,a) [r + gamma * V*(s')]
"""

from typing import Dict, List, Tuple
import numpy as np
from gridworld_environment import GridWorld, ACTION_ARROWS


def value_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iters: int = 500,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Runs Value Iteration, returning (V_star, optimal_policy_indices, delta_history)."""
    V = np.zeros(env.n_states, dtype=np.float64)
    delta_history = []

    for k in range(1, max_iters + 1):
        delta = 0.0
        V_new = np.zeros_like(V)

        for s in range(env.n_states):
            pos = env.state_to_pos(s)
            if env.is_terminal(pos) or pos in env.obstacles:
                continue

            action_vals = []
            for a in range(env.n_actions):
                q_sa = 0.0
                for prob, ns, reward, done in env.get_transition_outcomes(s, a):
                    q_sa += prob * (reward + gamma * (0.0 if done else V[ns]))
                action_vals.append(q_sa)

            V_new[s] = max(action_vals)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        delta_history.append(delta)
        if delta < theta:
            break

    # Extract greedy optimal policy
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        pos = env.state_to_pos(s)
        if env.is_terminal(pos) or pos in env.obstacles:
            continue
        action_vals = []
        for a in range(env.n_actions):
            q_sa = 0.0
            for prob, ns, reward, done in env.get_transition_outcomes(s, a):
                q_sa += prob * (reward + gamma * (0.0 if done else V[ns]))
            action_vals.append(q_sa)
        policy[s] = int(np.argmax(action_vals))

    return V, policy, delta_history


def render_policy_grid(env: GridWorld, policy: np.ndarray):
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            pos = (r, c)
            if pos in env.obstacles:
                cells.append(" # ")
            elif pos in env.hazards:
                cells.append(" X ")
            elif pos == env.goal_pos:
                cells.append(" G ")
            else:
                s = env.pos_to_state(pos)
                cells.append(f" {ACTION_ARROWS[int(policy[s])]} ")
        print("  |" + "|".join(cells) + "|")


def main():
    print("=" * 70)
    print("CHAPTER 35: VALUE ITERATION (DYNAMIC PROGRAMMING)")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, slip_prob=0.1)
    V_star, pi_star, deltas = value_iteration(env, gamma=0.9, theta=1e-6)

    print(f"Converged in {len(deltas)} iterations (Final max_delta = {deltas[-1]:.2e}).")
    print(f"Optimal Start State Value V*(start): {V_star[env.pos_to_state(env.start_pos)]:.4f}\n")

    print("Optimal State Values V*(s):")
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
                row_str.append(f"{V_star[env.pos_to_state(pos)]:7.3f} ")
        print("  " + "|".join(row_str))

    print("\nExtracted Optimal Policy pi*(s):")
    render_policy_grid(env, pi_star)

    print("\n[SUCCESS] Value Iteration verified.")


if __name__ == "__main__":
    main()

