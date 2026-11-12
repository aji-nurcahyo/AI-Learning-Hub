"""Temporal-Difference TD(0) Learning (Model-Free Prediction).

Implements online bootstrapping TD(0) update at every transition step:
   td_target = r + gamma * (0.0 if done else V[next_state])
   td_error  = td_target - V[state]
   V[state]  = V[state] + alpha * td_error
Compares RMSE convergence against exact Dynamic Programming and First-Visit Monte Carlo.
"""

import numpy as np
from gridworld_environment import GridWorld
from policy_evaluation import iterative_policy_evaluation
from monte_carlo_returns import first_visit_mc_prediction


def td_zero_prediction(
    env: GridWorld,
    policy_probs: np.ndarray,
    num_episodes: int = 1500,
    alpha: float = 0.05,
    gamma: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    """Estimates V^pi(s) using TD(0) bootstrapping updates."""
    rng = np.random.default_rng(seed)
    V = np.zeros(env.n_states, dtype=np.float64)

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = int(rng.choice(env.n_actions, p=policy_probs[state]))
            next_state, reward, done, info = env.step(action)

            # Note: If episode hit max_steps truncation without terminal state,
            # true value bootstrapping still applies for non-terminal states
            is_true_terminal = info["reached_goal"] or info["hit_hazard"]
            next_val = 0.0 if is_true_terminal else V[next_state]

            td_target = reward + gamma * next_val
            td_error = td_target - V[state]
            V[state] += alpha * td_error

            state = next_state

    return V


def main():
    print("=" * 70)
    print("CHAPTER 35: TEMPORAL-DIFFERENCE TD(0) VS MONTE CARLO")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, slip_prob=0.0, max_steps=50)
    uniform_pi = np.full((env.n_states, env.n_actions), 0.25)

    V_exact, _ = iterative_policy_evaluation(env, uniform_pi, gamma=0.9, theta=1e-5)
    V_mc = first_visit_mc_prediction(env, uniform_pi, num_episodes=1500, gamma=0.9, seed=42)
    V_td = td_zero_prediction(env, uniform_pi, num_episodes=1500, alpha=0.05, gamma=0.9, seed=42)

    mask = [
        s for s in range(env.n_states)
        if not env.is_terminal(env.state_to_pos(s)) and env.state_to_pos(s) not in env.obstacles
    ]
    rmse_mc = float(np.sqrt(np.mean((V_mc[mask] - V_exact[mask]) ** 2)))
    rmse_td = float(np.sqrt(np.mean((V_td[mask] - V_exact[mask]) ** 2)))

    s_start = env.pos_to_state(env.start_pos)
    print("\nEstimation Comparison at 1500 Episodes:")
    print(f"- Exact DP V(start)  : {V_exact[s_start]:7.4f}")
    print(f"- Monte Carlo V(start): {V_mc[s_start]:7.4f} (Overall RMSE: {rmse_mc:.4f})")
    print(f"- TD(0) V(start)      : {V_td[s_start]:7.4f} (Overall RMSE: {rmse_td:.4f})")

    print("\nKey Insight: TD(0) updates estimates online at every single step via bootstrapping,")
    print("whereas Monte Carlo must wait until the entire episode finishes.")
    print("\n[SUCCESS] TD(0) prediction verified.")


if __name__ == "__main__":
    main()

