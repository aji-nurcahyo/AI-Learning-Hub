"""REINFORCE (Monte Carlo Policy Gradient) Using PyTorch.

Demonstrates policy-based RL optimization directly over parameterized policies pi_theta(a|s):
   J(theta) = E_{tau ~ pi_theta}[R(tau)]
   grad_theta J(theta) = E[ sum_t grad_theta log pi_theta(a_t|s_t) * (G_t - baseline) ]
Uses a compact PyTorch Embedding logits policy with Softmax action sampling
and baseline variance reduction on CPU.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from gridworld_environment import GridWorld
from value_iteration import render_policy_grid


class SoftmaxPolicyNetwork(nn.Module):
    """Parameterized stochastic policy mapping discrete states to action logits."""

    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.logits = nn.Embedding(n_states, n_actions)
        nn.init.zeros_(self.logits.weight)

    def forward(self, state_idx: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(state_idx), dim=-1)


def compute_returns(rewards: List[float], gamma: float = 0.9) -> torch.Tensor:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return torch.tensor(returns, dtype=torch.float32)


def train_reinforce(
    env: GridWorld,
    num_episodes: int = 600,
    lr: float = 0.05,
    gamma: float = 0.9,
    seed: int = 42,
) -> Tuple[SoftmaxPolicyNetwork, List[float]]:
    """Trains a REINFORCE policy gradient agent with running baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy_net = SoftmaxPolicyNetwork(env.n_states, env.n_actions)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    returns_history = []
    running_baseline = 0.0

    for ep in range(num_episodes):
        state = env.reset(seed=seed + ep)
        log_probs = []
        rewards = []
        done = False

        while not done:
            s_tensor = torch.tensor(state, dtype=torch.long)
            probs = policy_net(s_tensor)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(int(action.item()))
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state

        ep_return = sum(rewards)
        returns_history.append(ep_return)

        G_tensor = compute_returns(rewards, gamma=gamma)
        # Update exponential moving average baseline for variance reduction
        running_baseline = 0.95 * running_baseline + 0.05 * float(G_tensor[0].item())
        advantages = G_tensor - running_baseline

        # Policy gradient loss: -sum(log_prob * advantage)
        policy_loss = []
        for lp, adv in zip(log_probs, advantages):
            policy_loss.append(-lp * adv)
        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy_net, returns_history


def evaluate_reinforce_policy(env: GridWorld, policy_net: SoftmaxPolicyNetwork, n_eval: int = 20) -> Dict[str, float]:
    """Evaluates learned REINFORCE policy deterministically (argmax action)."""
    successes = 0
    total_returns = []
    lengths = []

    with torch.no_grad():
        for i in range(n_eval):
            state = env.reset(seed=2000 + i)
            done = False
            ep_ret = 0.0
            steps = 0

            while not done:
                probs = policy_net(torch.tensor(state, dtype=torch.long))
                action = int(torch.argmax(probs).item())
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
    print("CHAPTER 35: POLICY GRADIENT (REINFORCE WITH PYTORCH)")
    print("=" * 70)

    env = GridWorld(rows=4, cols=4, slip_prob=0.0, max_steps=30, seed=42)
    print("Training REINFORCE Softmax Policy on 4x4 GridWorld (600 episodes)...")

    policy_net, hist = train_reinforce(env, num_episodes=600, lr=0.05, gamma=0.9, seed=42)

    print(f"  First 100 episodes Avg Return: {np.mean(hist[:100]):.3f}")
    print(f"  Last  100 episodes Avg Return: {np.mean(hist[-100:]):.3f}")

    eval_res = evaluate_reinforce_policy(env, policy_net, n_eval=20)
    print(f"\nDeterministic Policy Evaluation (argmax over 20 runs):")
    print(f"  - Success Rate  : {eval_res['success_rate'] * 100:.1f}%")
    print(f"  - Average Return: {eval_res['avg_return']:.4f}")
    print(f"  - Average Steps : {eval_res['avg_length']}")

    # Extract argmax arrows
    with torch.no_grad():
        greedy_actions = np.array([int(torch.argmax(policy_net(torch.tensor(s))).item()) for s in range(env.n_states)])
    print("\nLearned REINFORCE Greedy Action Arrows:")
    render_policy_grid(env, greedy_actions)

    print("\n[SUCCESS] PyTorch REINFORCE algorithm verified.")


if __name__ == "__main__":
    main()

