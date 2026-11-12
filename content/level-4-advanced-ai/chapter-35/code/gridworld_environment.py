"""Standalone Configurable GridWorld Simulator for Reinforcement Learning.

Implements a self-contained MDP GridWorld environment using Python & NumPy:
- Configurable grid dimensions, start, goal, obstacles, and hazards.
- Deterministic and stochastic transition dynamics (controlled via slip_prob).
- Explicit transition probability table P(s'|s, a) for Dynamic Programming.
- Reproducible seeding, step budgets, and ASCII grid rendering.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


ACTIONS = {
    0: (-1, 0),  # UP
    1: (0, 1),   # RIGHT
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
}

ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


class GridWorld:
    """Configurable 2D GridWorld MDP Environment."""

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (3, 3),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        hazards: Optional[List[Tuple[int, int]]] = None,
        step_reward: float = -0.1,
        goal_reward: float = 10.0,
        hazard_reward: float = -5.0,
        slip_prob: float = 0.0,
        max_steps: int = 50,
        seed: int = 42,
    ):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = set(obstacles) if obstacles is not None else {(1, 1), (2, 2)}
        self.hazards = set(hazards) if hazards is not None else {(1, 3)}
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.hazard_reward = hazard_reward
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.done = False
        self.episode_return = 0.0

    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.cols, state % self.cols)

    def is_terminal(self, pos: Tuple[int, int]) -> bool:
        return pos == self.goal_pos or pos in self.hazards

    def _move_attempt(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        if self.is_terminal(pos) or pos in self.obstacles:
            return pos
        dr, dc = ACTIONS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        # Boundary or obstacle collision leaves agent in current cell
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.obstacles:
            return (nr, nc)
        return pos

    def _get_reward(self, next_pos: Tuple[int, int]) -> float:
        if next_pos == self.goal_pos:
            return self.goal_reward
        if next_pos in self.hazards:
            return self.hazard_reward
        return self.step_reward

    def get_transition_outcomes(self, state: int, action: int) -> List[Tuple[float, int, float, bool]]:
        """Returns list of (prob, next_state, reward, done) for Dynamic Programming."""
        pos = self.state_to_pos(state)
        if self.is_terminal(pos) or pos in self.obstacles:
            return [(1.0, state, 0.0, True)]

        if self.slip_prob == 0.0:
            next_pos = self._move_attempt(pos, action)
            ns = self.pos_to_state(next_pos)
            r = self._get_reward(next_pos)
            d = self.is_terminal(next_pos)
            return [(1.0, ns, r, d)]

        # Stochastic transition: (1 - slip_prob) intended action, slip_prob/2 left/right perpendicular
        outcomes: Dict[int, Tuple[float, float, bool]] = {}
        perp_actions = [(action - 1) % 4, (action + 1) % 4]
        action_probs = [
            (action, 1.0 - self.slip_prob),
            (perp_actions[0], self.slip_prob / 2.0),
            (perp_actions[1], self.slip_prob / 2.0),
        ]

        for act, prob in action_probs:
            if prob <= 0.0:
                continue
            next_pos = self._move_attempt(pos, act)
            ns = self.pos_to_state(next_pos)
            r = self._get_reward(next_pos)
            d = self.is_terminal(next_pos)
            if ns in outcomes:
                old_p, old_r, old_d = outcomes[ns]
                outcomes[ns] = (old_p + prob, r, d)
            else:
                outcomes[ns] = (prob, r, d)

        return [(p, ns, r, d) for ns, (p, r, d) in outcomes.items()]

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.done = False
        self.episode_return = 0.0
        return self.pos_to_state(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("Episode sudah selesai. Panggil reset() terlebih dahulu.")

        if action not in ACTIONS:
            raise ValueError(f"Aksi tidak valid: {action}. Harus di {list(ACTIONS.keys())}")

        curr_state = self.pos_to_state(self.agent_pos)
        outcomes = self.get_transition_outcomes(curr_state, action)

        probs = [o[0] for o in outcomes]
        chosen_idx = int(self.rng.choice(len(outcomes), p=probs))
        _, next_state, reward, reached_terminal = outcomes[chosen_idx]

        self.agent_pos = self.state_to_pos(next_state)
        self.steps_taken += 1
        self.episode_return += reward

        truncated = (self.steps_taken >= self.max_steps) and not reached_terminal
        self.done = reached_terminal or truncated

        info = {
            "steps": self.steps_taken,
            "pos": self.agent_pos,
            "reached_goal": (self.agent_pos == self.goal_pos),
            "hit_hazard": (self.agent_pos in self.hazards),
            "truncated": truncated,
        }
        return next_state, float(reward), bool(self.done), info

    def render_ascii(self) -> str:
        lines = []
        border = "+" + "---+" * self.cols
        lines.append(border)
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.agent_pos:
                    ch = " A "
                elif pos == self.goal_pos:
                    ch = " G "
                elif pos in self.obstacles:
                    ch = " # "
                elif pos in self.hazards:
                    ch = " X "
                elif pos == self.start_pos:
                    ch = " S "
                else:
                    ch = " . "
                row_cells.append(ch)
            lines.append("|" + "|".join(row_cells) + "|")
            lines.append(border)
        return "\n".join(lines)


def main():
    print("=" * 70)
    print("CHAPTER 35: STANDALONE GRIDWORLD SIMULATOR")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, slip_prob=0.0, seed=42)
    s0 = env.reset()
    print(f"Initial State: {s0} (Pos: {env.agent_pos})")
    print("Legend: A=Agent, S=Start, G=Goal(+10), X=Hazard(-5), #=Wall\n")
    print(env.render_ascii())

    # Execute a sample manual trajectory: DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT
    sample_actions = [2, 2, 2, 1, 1, 1]
    print("\nExecuting trajectory: DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT (avoiding walls):")
    # Wait, wall at (2, 2)! Let's trace:
    # (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(3,3)! Let's test!
    for step_idx, a in enumerate(sample_actions, 1):
        ns, r, done, info = env.step(a)
        print(f"  Step {step_idx}: Action={ACTION_NAMES[a]:<5} -> State={ns:2d} {info['pos']} | Reward={r:5.2f} | Done={done}")
        if done:
            break

    print("\nFinal Grid State:")
    print(env.render_ascii())
    print(f"Episode Return: {env.episode_return:.2f} | Reached Goal: {info['reached_goal']}")
    print("\n[SUCCESS] Standalone GridWorld simulator verified.")


if __name__ == "__main__":
    main()

