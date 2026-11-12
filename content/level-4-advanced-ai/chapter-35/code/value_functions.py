"""State Value Function V(s) and Action Value Function Q(s, a).

Demonstrates:
1. Relation: Q^pi(s, a) = sum_{s', r} P(s', r|s, a) * [r + gamma * V^pi(s')]
2. Relation: V^pi(s) = sum_a pi(a|s) * Q^pi(s, a)
3. Exact computation of Q^pi(s, a) given a converged V^pi(s) on GridWorld.
"""

import numpy as np
from gridworld_environment import GridWorld, ACTION_NAMES


def compute_q_from_v(env: GridWorld, V: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """Computes action-value matrix Q(s, a) from state-value vector V(s)."""
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    for s in range(env.n_states):
        pos = env.state_to_pos(s)
        if env.is_terminal(pos) or pos in env.obstacles:
            continue
        for a in range(env.n_actions):
            q_val = 0.0
            for prob, ns, reward, done in env.get_transition_outcomes(s, a):
                next_val = 0.0 if done else V[ns]
                q_val += prob * (reward + gamma * next_val)
            Q[s, a] = q_val
    return Q


def main():
    print("=" * 70)
    print("CHAPTER 35: STATE & ACTION VALUE FUNCTIONS (V AND Q)")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, slip_prob=0.0)
    gamma = 0.9

    # Initialize uniform random policy pi(a|s) = 0.25
    pi = np.full((env.n_states, env.n_actions), 0.25)

    # First evaluate V^pi iteratively
    V = np.zeros(env.n_states, dtype=np.float64)
    for _ in range(100):
        V_new = np.zeros_like(V)
        for s in range(env.n_states):
            pos = env.state_to_pos(s)
            if env.is_terminal(pos) or pos in env.obstacles:
                continue
            for a in range(env.n_actions):
                for prob, ns, reward, done in env.get_transition_outcomes(s, a):
                    V_new[s] += pi[s, a] * prob * (reward + gamma * (0.0 if done else V[ns]))
        if np.max(np.abs(V_new - V)) < 1e-6:
            V = V_new
            break
        V = V_new

    Q = compute_q_from_v(env, V, gamma=gamma)

    # Verify V^pi(s) == sum_a pi(a|s) * Q^pi(s, a)
    reconstructed_V = np.sum(pi * Q, axis=1)
    assert np.allclose(V, reconstructed_V, atol=1e-6), "V != sum(pi * Q)"

    print("State Value Grid V^pi(s) under Uniform Random Policy (gamma=0.9):")
    for r in range(env.rows):
        row_vals = [f"{V[env.pos_to_state((r, c))]:6.2f}" for c in range(env.cols)]
        print("  " + " | ".join(row_vals))

    # Inspect Q-values at a non-terminal state (3, 2), adjacent to Goal (3, 3)
    s_adj = env.pos_to_state((3, 2))
    print(f"\nAction Values Q^pi(s, a) at State {s_adj} {(3, 2)} (Adjacent to Goal (3, 3)):")
    for a in range(env.n_actions):
        print(f"  Q(s={s_adj}, a={ACTION_NAMES[a]:<5}) = {Q[s_adj, a]:7.3f}")
    print(f"  Expected Value V^pi(s={s_adj}) = {V[s_adj]:7.3f} (Verified = sum pi*Q)")

    print("\n[SUCCESS] Value functions V(s) and Q(s, a) consistency verified.")


if __name__ == "__main__":
    main()

