"""Bellman Expectation vs Bellman Optimality Equations.

Demonstrates the mathematical distinction between:
1. Bellman Expectation Equation (for evaluating a fixed policy pi):
   V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [R + gamma * V^pi(s')]
2. Bellman Optimality Equation (for finding optimal value V*):
   V*(s) = max_a sum_s' P(s'|s,a) [R + gamma * V*(s')]

Includes a manual step-by-step numerical verification.
"""

import numpy as np
from gridworld_environment import GridWorld, ACTION_NAMES


def bellman_expectation_backup(env: GridWorld, V: np.ndarray, pi: np.ndarray, s: int, gamma: float) -> float:
    pos = env.state_to_pos(s)
    if env.is_terminal(pos) or pos in env.obstacles:
        return 0.0
    val = 0.0
    for a in range(env.n_actions):
        action_val = 0.0
        for prob, ns, reward, done in env.get_transition_outcomes(s, a):
            action_val += prob * (reward + gamma * (0.0 if done else V[ns]))
        val += pi[s, a] * action_val
    return float(val)


def bellman_optimality_backup(env: GridWorld, V: np.ndarray, s: int, gamma: float) -> float:
    pos = env.state_to_pos(s)
    if env.is_terminal(pos) or pos in env.obstacles:
        return 0.0
    action_values = []
    for a in range(env.n_actions):
        action_val = 0.0
        for prob, ns, reward, done in env.get_transition_outcomes(s, a):
            action_val += prob * (reward + gamma * (0.0 if done else V[ns]))
        action_values.append(action_val)
    return float(max(action_values))


def main():
    print("=" * 70)
    print("CHAPTER 35: BELLMAN EXPECTATION VS BELLMAN OPTIMALITY")
    print("=" * 70)

    # Small 3x3 GridWorld with Goal at (2, 2)
    env = GridWorld(rows=3, cols=3, start_pos=(0, 0), goal_pos=(2, 2), obstacles={(1, 1)}, hazards=set(), slip_prob=0.0)
    gamma = 0.9

    # Suppose at iteration k, state (2, 1) adjacent to Goal (2, 2) has neighbors:
    # RIGHT -> (2, 2) Terminal (reward +10.0, done=True -> V=0)
    # LEFT  -> (2, 0) (suppose V_k(2,0) = 6.0, reward -0.1)
    # UP    -> (1, 1) Obstacle (bounces back to (2, 1), suppose V_k(2,1) = 8.0, reward -0.1)
    # DOWN  -> Wall (bounces back to (2, 1), suppose V_k(2,1) = 8.0, reward -0.1)
    V_k = np.zeros(env.n_states, dtype=np.float64)
    V_k[env.pos_to_state((2, 0))] = 6.0
    V_k[env.pos_to_state((2, 1))] = 8.0

    s_target = env.pos_to_state((2, 1))
    uniform_pi = np.full((env.n_states, env.n_actions), 0.25)

    exp_val = bellman_expectation_backup(env, V_k, uniform_pi, s_target, gamma)
    opt_val = bellman_optimality_backup(env, V_k, s_target, gamma)

    # Independent manual calculation verification:
    # UP    (bounce to 2,1): -0.1 + 0.9 * 8.0 = 7.1
    # RIGHT (reach goal 2,2): +10.0 + 0.9 * 0.0 = 10.0
    # DOWN  (bounce to 2,1): -0.1 + 0.9 * 8.0 = 7.1
    # LEFT  (move to 2,0): -0.1 + 0.9 * 6.0 = 5.3
    manual_action_vals = [7.1, 10.0, 7.1, 5.3]
    manual_exp = 0.25 * sum(manual_action_vals)  # = 7.375
    manual_opt = max(manual_action_vals)         # = 10.0

    assert np.isclose(exp_val, manual_exp, atol=1e-6)
    assert np.isclose(opt_val, manual_opt, atol=1e-6)

    print(f"Worked Numerical Example at State {s_target} {(2, 1)} (gamma={gamma}):")
    print("  Action Q-estimates:")
    for a, q in enumerate(manual_action_vals):
        print(f"    {ACTION_NAMES[a]:<5}: {q:6.2f}")

    print("\n1. Bellman Expectation Backup (Uniform Policy pi=0.25):")
    print(f"   Formula : 0.25 * (7.10 + 10.00 + 7.10 + 5.30)")
    print(f"   Computed: {exp_val:.4f} (Matches Manual: {manual_exp:.4f})")

    print("\n2. Bellman Optimality Backup (Max Over Actions):")
    print(f"   Formula : max(7.10, 10.00, 7.10, 5.30)")
    print(f"   Computed: {opt_val:.4f} (Matches Manual: {manual_opt:.4f})")

    print("\nKey Distinction: Expectation computes the weighted average under a fixed policy;")
    print("Optimality takes the greedy maximum over actions, producing a non-linear operator.")
    print("\n[SUCCESS] Bellman equations numerical distinction verified.")


if __name__ == "__main__":
    main()

