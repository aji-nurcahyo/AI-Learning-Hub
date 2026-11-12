"""Track E (Robotics & Intelligent Systems) Applied Lab: A* Search vs Q-Learning Navigation."""

import heapq
import numpy as np


GRID_SIZE = 5
OBSTACLES = {(1, 1), (1, 2), (3, 1), (3, 3)}
START = (0, 0)
GOAL = (4, 4)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right


def astar_plan(start=START, goal=GOAL):
    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    pq = [(h(start), 0, start, [start])]
    best_g = {start: 0}
    while pq:
        _, g, cur, path = heapq.heappop(pq)
        if cur == goal:
            return path
        for dr, dc in ACTIONS:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and nxt not in OBSTACLES:
                ng = g + 1
                if ng < best_g.get(nxt, 10**9):
                    best_g[nxt] = ng
                    heapq.heappush(pq, (ng + h(nxt), ng, nxt, path + [nxt]))
    return []


def train_q_learning(episodes: int = 350, seed: int = 42):
    rng = np.random.default_rng(seed)
    q = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    alpha, gamma, eps = 0.25, 0.95, 0.30

    for _ in range(episodes):
        r, c = START
        for _ in range(50):
            if (r, c) == GOAL:
                break
            a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(q[r, c]))
            dr, dc = ACTIONS[a]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE) or (nr, nc) in OBSTACLES:
                nr, nc = r, c
                rew = -1.0
            elif (nr, nc) == GOAL:
                rew = 10.0
            else:
                rew = -0.1
            q[r, c, a] += alpha * (rew + gamma * np.max(q[nr, nc]) - q[r, c, a])
            r, c = nr, nc

    # Greedy rollout evaluation
    path = [START]
    r, c = START
    for _ in range(25):
        if (r, c) == GOAL:
            break
        a = int(np.argmax(q[r, c]))
        nr, nc = r + ACTIONS[a][0], c + ACTIONS[a][1]
        if (nr, nc) not in OBSTACLES and 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            r, c = nr, nc
        path.append((r, c))
    return path


def main():
    astar_path = astar_plan()
    ql_path = train_q_learning()
    print("=" * 75)
    print("TRACK E (ROBOTICS & INTELLIGENT SYSTEMS): A* vs Q-LEARNING NAVIGATION")
    print("=" * 75)
    print(f"A* Optimal Path Length     : {len(astar_path) - 1} steps -> Goal Reached={astar_path[-1] == GOAL}")
    print(f"Q-Learning Policy Rollout  : {len(ql_path) - 1} steps -> Goal Reached={ql_path[-1] == GOAL}")
    assert astar_path[-1] == GOAL and ql_path[-1] == GOAL
    print("[SUCCESS] Autonomous GridWorld planning & policy navigation verified offline.")


if __name__ == "__main__":
    main()

