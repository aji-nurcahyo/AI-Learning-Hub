"""Policy Iteration Algorithm (Dynamic Programming).

Alternates between two exact phases until convergence:
1. Policy Evaluation: Compute V^pi(s) for current deterministic policy pi.
2. Policy Improvement: Update pi(s) <- argmax_a Q^pi(s, a).
Verifies parity with Value Iteration optimal solution.
"""

from typing import Tuple
import numpy as np
from gridworld_environment import GridWorld, ACTION_ARROWS
from value_iteration import value_iteration, render_policy_grid


def evaluate_deterministic_policy(
    env: GridWorld,
    policy: np.ndarray,
    gamma: float = 0.9,
    theta: float = 1e-6,
) -> np.ndarray:
    V = np.zeros(env.n_states, dtype=np.float64)
    while True:
        delta = 0.0
        V_new = np.zeros_like(V)
        for s in range(env.n_states):
            pos = env.state_to_pos(s)
            if env.is_terminal(pos) or pos in env.obstacles:
                continue
            a = int(policy[s])
            v_val = 0.0
            for prob, ns, reward, done in env.get_transition_outcomes(s, a):
                v_val += prob * (reward + gamma * (0.0 if done else V[ns]))
            V_new[s] = v_val
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < theta:
            break
    return V


def policy_iteration(env: GridWorld, gamma: float = 0.9, max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
    """Runs Policy Iteration returning (V_star, optimal_policy, outer_iterations)."""
    # Initialize arbitrary policy (e.g. always action 0 UP)
    policy = np.zeros(env.n_states, dtype=int)
    outer_iters = 0

    for k in range(1, max_iters + 1):
        outer_iters = k
        # 1. Policy Evaluation
        V = evaluate_deterministic_policy(env, policy, gamma=gamma)

        # 2. Policy Improvement
        policy_stable = True
        for s in range(env.n_states):
            pos = env.state_to_pos(s)
            if env.is_terminal(pos) or pos in env.obstacles:
                continue

            old_action = int(policy[s])
            action_vals = []
            for a in range(env.n_actions):
                q_sa = 0.0
                for prob, ns, reward, done in env.get_transition_outcomes(s, a):
                    q_sa += prob * (reward + gamma * (0.0 if done else V[ns]))
                action_vals.append(q_sa)

            best_action = int(np.argmax(action_vals))
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        print(f"  Policy Iteration Step {k}: V(start) = {V[env.pos_to_state(env.start_pos)]:7.4f} | Stable = {policy_stable}")
        if policy_stable:
            break

    return V, policy, outer_iters


def main():
    print("=" * 70)
    print("CHAPTER 35: POLICY ITERATION (DYNAMIC PROGRAMMING)")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, slip_prob=0.1)
    V_pi, pi_opt, total_steps = policy_iteration(env, gamma=0.9)

    # Verify parity with Value Iteration
    V_vi, pi_vi, _ = value_iteration(env, gamma=0.9, theta=1e-7)
    max_diff = np.max(np.abs(V_pi - V_vi))
    assert max_diff < 1e-4, f"Mismatch between Policy Iteration and Value Iteration: {max_diff}"

    print(f"\nPolicy Iteration converged in {total_steps} outer policy improvements.")
    print(f"Max Value Function Difference vs Value Iteration: {max_diff:.2e} (PASS)")

    print("\nOptimal Policy Grid:")
    render_policy_grid(env, pi_opt)

    print("\n[SUCCESS] Policy Iteration verified and matches Value Iteration.")


if __name__ == "__main__":
    main()

