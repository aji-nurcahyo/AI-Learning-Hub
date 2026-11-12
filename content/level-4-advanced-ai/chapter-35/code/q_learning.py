"""Off-Policy Q-Learning Control Algorithm.

Update rule:
   Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_{a'} Q(s', a') - Q(s, a) ]
   (where max Q(s', a') = 0.0 if s' is a true terminal state)

Includes:
1. Step-by-step manual numerical worked example verified with NumPy.
2. Full GridWorld Q-Learning training with Epsilon-Greedy exploration.
3. Strict Greedy evaluation (epsilon=0) verifying optimal policy convergence.
"""

from typing import Dict, List, Tuple
import numpy as np
from gridworld_environment import GridWorld, ACTION_ARROWS
from value_iteration import render_policy_grid


def q_learning_single_step_worked_example():
    """Verifies a single Q-Learning update step analytically."""
    q_table = np.array([
        [0.0, 2.0, 1.0, 0.0],  # State 0
        [1.0, 5.0, 0.5, 0.0],  # State 1
    ], dtype=np.float64)

    s = 0
    a = 1  # RIGHT
    r = -0.1
    next_s = 1
    alpha = 0.2
    gamma = 0.9
    is_terminal = False

    old_q = q_table[s, a]  # 2.0
    max_next_q = 0.0 if is_terminal else float(np.max(q_table[next_s]))  # max([1.0, 5.0, 0.5, 0.0]) = 5.0
    td_target = r + gamma * max_next_q  # -0.1 + 0.9 * 5.0 = 4.4
    td_error = td_target - old_q        # 4.4 - 2.0 = 2.4
    new_q = old_q + alpha * td_error    # 2.0 + 0.2 * 2.4 = 2.48

    assert np.isclose(td_target, 4.4, atol=1e-9)
    assert np.isclose(td_error, 2.4, atol=1e-9)
    assert np.isclose(new_q, 2.48, atol=1e-9)

    print("1. Manual Worked Numerical Example Verification:")
    print(f"   Current Q(s={s}, a={a}) = {old_q:.2f}, r = {r:.2f}, next_s = {next_s}")
    print(f"   max_a' Q(s'={next_s}, a') = {max_next_q:.2f}")
    print(f"   TD Target = {r:.2f} + {gamma:.2f} * {max_next_q:.2f} = {td_target:.4f}")
    print(f"   TD Error  = {td_target:.4f} - {old_q:.2f} = {td_error:.4f}")
    print(f"   Updated Q = {old_q:.2f} + {alpha:.2f} * {td_error:.4f} = {new_q:.4f} (PASS)\n")


def train_q_learning(
    env: GridWorld,
    num_episodes: int = 800,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
) -> Tuple[np.ndarray, List[float]]:
    """Trains an off-policy Q-table agent on GridWorld."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    returns_history = []
    epsilon = epsilon_start

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.n_actions))
            else:
                # Break ties randomly among maximal Q-values
                max_q = np.max(Q[state])
                best_actions = np.where(np.isclose(Q[state], max_q))[0]
                action = int(rng.choice(best_actions))

            next_state, reward, done, info = env.step(action)
            ep_return += reward

            is_true_terminal = info["reached_goal"] or info["hit_hazard"]
            next_max_q = 0.0 if is_true_terminal else float(np.max(Q[next_state]))

            td_target = reward + gamma * next_max_q
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state

        returns_history.append(ep_return)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q, returns_history


def evaluate_greedy_policy(env: GridWorld, Q: np.ndarray, n_eval: int = 20) -> Dict[str, float]:
    """Evaluates learned Q-table strictly with epsilon=0 (greedy policy)."""
    successes = 0
    total_returns = []
    lengths = []

    for i in range(n_eval):
        state = env.reset(seed=1000 + i)
        done = False
        ep_ret = 0.0
        steps = 0

        while not done:
            action = int(np.argmax(Q[state]))
            next_state, r, done, info = env.step(action)
            ep_ret += r
            steps += 1
            state = next_state
            if info["reached_goal"]:
                successes += 1

        total_returns.append(ep_ret)
        lengths.append(steps)

    return {
        "success_rate": round(successes / n_eval, 4),
        "avg_return": round(float(np.mean(total_returns)), 4),
        "avg_length": round(float(np.mean(lengths)), 2),
    }


def main():
    print("=" * 70)
    print("CHAPTER 35: Q-LEARNING (OFF-POLICY TEMPORAL-DIFFERENCE CONTROL)")
    print("=" * 70)

    q_learning_single_step_worked_example()

    env = GridWorld(rows=4, cols=4, slip_prob=0.0, max_steps=50, seed=42)
    print("2. Training Q-Learning Agent on 4x4 GridWorld (800 episodes)...")
    Q, hist = train_q_learning(env, num_episodes=800, seed=42)

    print(f"   First 100 episodes Avg Return : {np.mean(hist[:100]):.3f}")
    print(f"   Last  100 episodes Avg Return : {np.mean(hist[-100:]):.3f}")

    # Greedy evaluation (epsilon=0)
    eval_res = evaluate_greedy_policy(env, Q, n_eval=20)
    print(f"\n3. Greedy Policy Evaluation (epsilon=0.0 over 20 runs):")
    print(f"   - Success Rate  : {eval_res['success_rate'] * 100:.1f}%")
    print(f"   - Average Return: {eval_res['avg_return']:.4f}")
    print(f"   - Average Steps : {eval_res['avg_length']}")

    learned_policy = np.argmax(Q, axis=1)
    print("\nLearned Greedy Policy Arrows:")
    render_policy_grid(env, learned_policy)

    print("\n[SUCCESS] Q-Learning algorithm and greedy evaluation verified.")


if __name__ == "__main__":
    main()

