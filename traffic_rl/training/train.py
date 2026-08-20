"""
training/train.py

DQN agent combining:
  - epsilon-greedy action selection
  - experience replay
  - target network (hard update)
  - gradient clipping

Also used as the main agent class from training/train.py.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.dqn import QNetwork
from memory.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg    = cfg
        self.device = device

        agent_cfg  = cfg["agent"]
        env_cfg    = cfg["environment"]
        net_cfg    = cfg["network"]
        buf_cfg    = cfg["replay_buffer"]

        self.state_dim  = env_cfg["state_dim"]
        self.action_dim = 2   # 2 actions: NS-green=0, EW-green=1 (simplified)

        self.gamma          = agent_cfg["gamma"]
        self.batch_size     = agent_cfg["batch_size"]
        self.lr             = agent_cfg["learning_rate"]
        self.grad_clip      = agent_cfg["grad_clip_norm"]
        self.target_update_freq = agent_cfg["target_update_freq"]

        self.epsilon_start  = agent_cfg["epsilon_start"]
        self.epsilon_end    = agent_cfg["epsilon_end"]
        self.epsilon_decay_episodes = agent_cfg["epsilon_decay_episodes"]
        self.epsilon        = self.epsilon_start

        # Networks
        hidden = net_cfg["hidden_layers"]
        self.policy_net = QNetwork(self.state_dim, self.action_dim, hidden).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer & loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        loss_name = agent_cfg.get("loss", "huber")
        self.criterion = nn.SmoothL1Loss() if loss_name == "huber" else nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buf_cfg["capacity"])
        self.min_buffer    = buf_cfg["min_size"]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy: random action or greedy Q-value action."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def train_step(self) -> float:
        """
        Sample one mini-batch, compute DQN loss, backprop.
        Returns loss value (float) or 0.0 if buffer too small.
        """
        if len(self.replay_buffer) < self.min_buffer:
            return 0.0

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size, self.device)

        # Current Q-values: Q(s, a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double-DQN Target Q-values: r + γ * Q_target(s', argmax_a Q_policy(s', a)) * (1 - done)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def update_target_network(self):
        """Hard copy policy → target."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self, episode: int):
        """Linear decay from epsilon_start to epsilon_end over N episodes."""
        fraction = min(episode / self.epsilon_decay_episodes, 1.0)
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)
        self.epsilon = max(self.epsilon, self.epsilon_end)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
