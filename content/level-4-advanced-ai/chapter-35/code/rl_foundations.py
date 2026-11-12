"""Reinforcement Learning Foundations: Agent-Environment Loop.

Demonstrates:
1. Interaction loop: state -> policy action -> reward & next_state.
2. Generating a full episode trajectory [(s_0, a_0, r_1), (s_1, a_1, r_2), ...].
3. Computing total episodic reward and length under a uniform random policy.
"""

import numpy as np
from gridworld_environment import GridWorld, ACTION_NAMES


def run_episode(env: GridWorld, seed: int = 123):
    rng = np.random.default_rng(seed)
    state = env.reset()
    trajectory = []
    done = False

    while not done:
        action = int(rng.integers(0, env.n_actions))
        next_state, reward, done, info = env.step(action)
        trajectory.append((state, action, reward, next_state, done))
        state = next_state

    return trajectory, info


def main():
    print("=" * 70)
    print("CHAPTER 35: REINFORCEMENT LEARNING FOUNDATIONS")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, max_steps=20, seed=42)
    trajectory, info = run_episode(env, seed=42)

    print(f"Environment: {env.rows}x{env.cols} GridWorld | States: {env.n_states} | Actions: {env.n_actions}")
    print(f"Executed Random Policy Episode (Length: {len(trajectory)} steps):\n")

    total_reward = 0.0
    for t, (s, a, r, ns, d) in enumerate(trajectory):
        total_reward += r
        pos_s = env.state_to_pos(s)
        pos_ns = env.state_to_pos(ns)
        print(f"  t={t:2d}: S={s:2d} {pos_s} --({ACTION_NAMES[a]:<5})--> S'={ns:2d} {pos_ns} | R={r:5.2f} | Done={d}")

    print(f"\nTotal Undiscounted Return: {total_reward:.2f}")
    print(f"Outcome: Reached Goal={info['reached_goal']}, Hit Hazard={info['hit_hazard']}, Truncated={info['truncated']}")
    print("\n[SUCCESS] Agent-Environment interaction loop verified.")


if __name__ == "__main__":
    main()

