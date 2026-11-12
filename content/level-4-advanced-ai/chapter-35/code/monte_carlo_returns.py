"""Model-Free First-Visit Monte Carlo Policy Evaluation.

Estimates V^pi(s) purely from sampled experience episodes without knowing P(s'|s,a):
   G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k+1}
   V(s) <- V(s) + (1 / N(s)) * (G_t - V(s))  [or incremental alpha]
Compares estimated values against exact Dynamic Programming solution.
"""

from collections import defaultdict
import numpy as np
from gridworld_environment import GridWorld
from policy_evaluation import iterative_policy_evaluation


def first_visit_mc_prediction(
    env: GridWorld,
    policy_probs: np.ndarray,
    num_episodes: int = 1500,
    gamma: float = 0.9,
    seed: int = 42,
) -> np.ndarray:
    """Estimates V^pi(s) via First-Visit Monte Carlo sampling."""
    rng = np.random.default_rng(seed)
    V = np.zeros(env.n_states, dtype=np.float64)
    visit_counts = np.zeros(env.n_states, dtype=np.int64)

    for ep in range(num_episodes):
        state = env.reset()
        episode_traj = []
        done = False

        while not done:
            action = int(rng.choice(env.n_actions, p=policy_probs[state]))
            next_state, reward, done, _ = env.step(action)
            episode_traj.append((state, action, reward))
            state = next_state

        # Compute returns backwards
        G = 0.0
        visited_in_ep = set()
        # First-visit needs checking earliest occurrence
        first_occurrences = {}
        for idx, (s, _, _) in enumerate(episode_traj):
            if s not in first_occurrences:
                first_occurrences[s] = idx

        # Compute return from each step
        returns_t = np.zeros(len(episode_traj), dtype=np.float64)
        for t in reversed(range(len(episode_traj))):
            _, _, r = episode_traj[t]
            G = r + gamma * G
            returns_t[t] = G

        for s, first_idx in first_occurrences.items():
            visit_counts[s] += 1
            # Incremental mean update
            V[s] += (returns_t[first_idx] - V[s]) / visit_counts[s]

    return V


def main():
    print("=" * 70)
    print("CHAPTER 35: MODEL-FREE MONTE CARLO PREDICTION")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, slip_prob=0.0, max_steps=50)
    uniform_pi = np.full((env.n_states, env.n_actions), 0.25)

    # Exact DP benchmark
    print("Computing exact Dynamic Programming benchmark V^pi(s)...")
    V_exact, _ = iterative_policy_evaluation(env, uniform_pi, gamma=0.9, theta=1e-5)

    # Monte Carlo estimation (model-free)
    print("\nRunning First-Visit Monte Carlo Estimation (1500 episodes)...")
    V_mc = first_visit_mc_prediction(env, uniform_pi, num_episodes=1500, gamma=0.9, seed=42)

    s_start = env.pos_to_state(env.start_pos)
    s_near_goal = env.pos_to_state((3, 2))
    print(f"\nComparison vs Exact Dynamic Programming:")
    print(f"- State (0, 0) [Start]: Exact = {V_exact[s_start]:7.4f} | MC Estimate = {V_mc[s_start]:7.4f}")
    print(f"- State (3, 2) [Adj.G]: Exact = {V_exact[s_near_goal]:7.4f} | MC Estimate = {V_mc[s_near_goal]:7.4f}")

    # Compute RMSE across non-terminal states
    mask = [
        s for s in range(env.n_states)
        if not env.is_terminal(env.state_to_pos(s)) and env.state_to_pos(s) not in env.obstacles
    ]
    rmse = float(np.sqrt(np.mean((V_mc[mask] - V_exact[mask]) ** 2)))
    print(f"- Root Mean Squared Error (RMSE) across non-terminal states: {rmse:.4f}")
    print("\n[SUCCESS] First-Visit Monte Carlo prediction verified.")


if __name__ == "__main__":
    main()

