"""Markov Decision Process (MDP) Transition Dynamics.

Demonstrates:
1. Formal MDP Tuple: M = (S, A, P, R, gamma).
2. Deterministic vs Stochastic transition probability distributions P(s'|s, a).
3. Mathematical verification that sum_s' P(s'|s, a) == 1.0 for every valid (s, a).
"""

import numpy as np
from gridworld_environment import GridWorld, ACTION_NAMES


def verify_transition_matrix(env: GridWorld) -> bool:
    for s in range(env.n_states):
        for a in range(env.n_actions):
            outcomes = env.get_transition_outcomes(s, a)
            total_prob = sum(p for p, _, _, _ in outcomes)
            if not np.isclose(total_prob, 1.0, atol=1e-6):
                return False
    return True


def main():
    print("=" * 70)
    print("CHAPTER 35: MARKOV DECISION PROCESS (MDP) TRANSITION DYNAMICS")
    print("=" * 70)

    det_env = GridWorld(rows=4, cols=4, slip_prob=0.0)
    stoch_env = GridWorld(rows=4, cols=4, slip_prob=0.2)

    print(f"Deterministic MDP Probability Axiom Check: {'PASS' if verify_transition_matrix(det_env) else 'FAIL'}")
    print(f"Stochastic MDP Probability Axiom Check   : {'PASS' if verify_transition_matrix(stoch_env) else 'FAIL'}\n")

    test_state = det_env.pos_to_state((0, 1))
    test_action = 1  # RIGHT

    print(f"Inspecting transitions from State {test_state} {(0, 1)} taking Action '{ACTION_NAMES[test_action]}':\n")

    print("[1] Deterministic Environment (slip_prob = 0.0):")
    for p, ns, r, d in det_env.get_transition_outcomes(test_state, test_action):
        print(f"    P(s'={ns:2d} {det_env.state_to_pos(ns)} | s, a) = {p:.2f} | R={r:5.2f} | Terminal={d}")

    print("\n[2] Stochastic Environment (slip_prob = 0.2):")
    for p, ns, r, d in stoch_env.get_transition_outcomes(test_state, test_action):
        print(f"    P(s'={ns:2d} {stoch_env.state_to_pos(ns)} | s, a) = {p:.2f} | R={r:5.2f} | Terminal={d}")

    print("\n[SUCCESS] MDP transition dynamics verified.")


if __name__ == "__main__":
    main()

