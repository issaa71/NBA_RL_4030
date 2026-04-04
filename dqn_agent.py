"""
dqn_agent.py
============
Baseline DQN agent for NBA possession-level shot selection.

Implements a standard Deep Q-Network (Mnih et al., 2015) with:
  - Online Q-network + frozen target network (hard update)
  - Epsilon-greedy exploration with exponential decay
  - Experience replay via ReplayBuffer (see replay_buffer.py)
  - Huber (smooth L1) loss for stable TD-error minimization
  - Adam optimizer

AISE 4030 - Group 11
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional

from q_network import QNetwork, build_q_network
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent for shot-selection in NBA possessions.

    Maintains an online Q-network (updated every training step) and a
    target Q-network (synced periodically) to stabilize learning.

    Attributes:
        config (dict): Full configuration dictionary from config.yaml.
        device (torch.device): Compute device (CPU, CUDA, or MPS).
        online_net (QNetwork): The Q-network being trained.
        target_net (QNetwork): The frozen target network.
        optimizer (torch.optim.Adam): Optimizer for online_net.
        loss_fn (nn.SmoothL1Loss): Huber loss for TD-error.
        replay_buffer (ReplayBuffer): Transition storage.
        epsilon (float): Current exploration rate.
        episode_count (int): Total episodes completed.
        training_losses (list): History of per-step training losses.
    """

    def __init__(self, config: dict, device: torch.device):
        """
        Initializes the DQNAgent: builds both Q-networks, optimizer, loss
        function, replay buffer, and all hyperparameters from config.

        Args:
            config (dict): Full configuration dictionary loaded from config.yaml.
            device (torch.device): The compute device to run training on.
        """
        self.config = config
        self.device = device

        dqn_cfg = config["dqn"]
        net_cfg = config["network"]

        self.online_net: QNetwork = build_q_network(config, device)
        self.target_net: QNetwork = build_q_network(config, device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=dqn_cfg["learning_rate"]
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.gamma: float = dqn_cfg["gamma"]
        self.n_step: int = dqn_cfg.get("n_step", 1)

        self.replay_buffer = ReplayBuffer(
            capacity=dqn_cfg["replay_buffer_capacity"],
            state_dim=net_cfg["state_dim"],
            n_step=self.n_step,
            gamma=self.gamma,
        )

        self.epsilon: float = dqn_cfg["epsilon_start"]
        self.epsilon_start: float = dqn_cfg["epsilon_start"]
        self.epsilon_end: float = dqn_cfg["epsilon_end"]
        self.epsilon_decay: float = dqn_cfg["epsilon_decay"]
        self.epsilon_decay_type: str = dqn_cfg.get("epsilon_decay_type", "multiplicative")
        self.epsilon_decay_episodes: int = dqn_cfg.get("epsilon_decay_episodes", 10000)
        self.batch_size: int = dqn_cfg["batch_size"]
        self.target_update_freq: int = dqn_cfg["target_update_freq"]
        self.tau: float = dqn_cfg.get("tau", 0.005)
        self.min_replay_size: int = dqn_cfg["min_replay_size"]
        self.grad_clip: float = dqn_cfg["grad_clip"]

        self.cql_alpha: float = config.get("cql", {}).get("alpha", 1.0)

        self.episode_count: int = 0
        self.training_losses: list = []

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Chooses an action for a given state using epsilon-greedy policy.

        During training, a random action is taken with probability epsilon.
        During evaluation (training=False), always returns the greedy action.

        Args:
            state (np.ndarray): The current state vector of shape (78,).
            training (bool): If True, applies epsilon-greedy exploration.

        Returns:
            int: Selected action index in range [0, 4].
        """
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.config["environment"]["num_actions"]))

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        self.online_net.train()
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Stores a single (s, a, r, s', done) transition in the replay buffer.

        Args:
            state (np.ndarray): State at time t, shape (5,).
            action (int): Action taken (0-4).
            reward (float): Reward received.
            next_state (np.ndarray): State at time t+1, shape (5,).
            done (bool): True if the episode terminated at this step.

        Returns:
            None
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Samples a mini-batch from the replay buffer and performs one
        DQN gradient update on the online network.

        TD target:
            y = r + gamma * max_a' Q_target(s', a')  if not done
            y = r                                      if done

        Returns:
            float or None: The scalar training loss, or None if the buffer
                does not yet have enough transitions.
        """
        if not self.replay_buffer.is_ready(self.min_replay_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Current Q-values for all actions (needed for CQL)
        all_q_values = self.online_net(states)
        q_taken = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: online net selects action, target net evaluates
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            # Use gamma^n for n-step bootstrap
            gamma_n = self.gamma ** self.n_step
            targets = rewards + gamma_n * max_next_q * (1.0 - dones)

        td_loss = self.loss_fn(q_taken, targets)

        # CQL penalty: push down Q-values for unobserved actions
        cql_penalty = (
            torch.logsumexp(all_q_values, dim=1).mean()
            - q_taken.mean()
        )
        loss = td_loss + self.cql_alpha * cql_penalty

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def decay_epsilon(self) -> None:
        """
        Applies epsilon decay at the end of each training episode.
        Supports linear or multiplicative decay based on config.

        Returns:
            None
        """
        if self.epsilon_decay_type == "linear":
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.epsilon_start - self.epsilon_end)
                * (self.episode_count / self.epsilon_decay_episodes)
            )
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def sync_target_network(self) -> None:
        """
        Performs a hard update of the target network by copying all weights
        from the online network.

        Returns:
            None
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def on_episode_end(self) -> None:
        """
        End-of-episode bookkeeping: increment counter, decay epsilon.
        Target network is updated via soft Polyak averaging in update().

        Returns:
            None
        """
        self.episode_count += 1
        self.decay_epsilon()
        if self.episode_count % self.target_update_freq == 0:
            self.sync_target_network()

    def save(self, path: str) -> None:
        """
        Saves the online network weights and optimizer state to disk.

        Args:
            path (str): Full file path to save the checkpoint.

        Returns:
            None
        """
        torch.save({
            'online_net': self.online_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
        }, path)

    def load(self, path: str) -> None:
        """
        Loads a saved checkpoint and restores the online network, target
        network, optimizer state, epsilon, and episode count.

        Args:
            path (str): Full file path to the saved checkpoint.

        Returns:
            None
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['online_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.target_net.eval()
