"""
replay_buffer.py
================
Experience replay buffer with n-step returns for off-policy DQN training.

Supports n-step return computation: accumulates n transitions before
pushing the discounted multi-step return to the main buffer. This
propagates reward signal faster through short episodes (avg 5.7 steps).

AISE 4030 - Group 11
"""

import numpy as np
import torch
from typing import Tuple
from collections import deque
import random


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer with n-step return support.

    Attributes:
        capacity (int): Maximum number of transitions stored.
        buffer (deque): Internal storage of (s, a, r, s', done) tuples.
        state_dim (int): Dimensionality of the state vector.
        n_step (int): Number of steps for multi-step returns.
        gamma (float): Discount factor for n-step computation.
    """

    def __init__(self, capacity: int, state_dim: int,
                 n_step: int = 1, gamma: float = 0.95):
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer: deque = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer: deque = deque(maxlen=n_step)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Adds a transition. If n_step > 1, accumulates n transitions
        and computes the discounted multi-step return before storing."""
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )

        if self.n_step <= 1:
            # Standard 1-step: push directly
            self.buffer.append(transition)
            if done:
                self.buffer.append(transition)
                self.buffer.append(transition)
            return

        # N-step accumulation
        self.n_step_buffer.append(transition)

        # If episode ended, flush all remaining transitions
        if done:
            while len(self.n_step_buffer) > 0:
                n_step_transition = self._compute_n_step_return()
                self.buffer.append(n_step_transition)
                # Oversample terminal (last transition)
                if len(self.n_step_buffer) == 0:
                    self.buffer.append(n_step_transition)
                    self.buffer.append(n_step_transition)
            return

        # Only push when buffer is full (have n steps of lookahead)
        if len(self.n_step_buffer) == self.n_step:
            n_step_transition = self._compute_n_step_return()
            self.buffer.append(n_step_transition)

    def _compute_n_step_return(self):
        """Compute n-step discounted return from the n_step_buffer.
        Uses the first transition's (s, a) and the last transition's (s', done),
        with accumulated discounted reward."""
        # First transition: state, action
        s, a, _, _, _ = self.n_step_buffer[0]

        # Accumulate discounted reward over n steps
        n_step_reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:  # Episode ended within n steps
                # Next state is terminal, done=True
                _, _, _, ns, _ = self.n_step_buffer[i]
                self.n_step_buffer.clear()
                return (s, a, float(n_step_reward), ns, 1.0)

        # Episode didn't end: next_state from last transition
        _, _, _, ns, done = self.n_step_buffer[-1]
        self.n_step_buffer.popleft()
        return (s, a, float(n_step_reward), ns, float(done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a random mini-batch of transitions from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states),      dtype=torch.float32).to(device),
            torch.tensor(np.array(actions),     dtype=torch.int64).to(device),
            torch.tensor(np.array(rewards),     dtype=torch.float32).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(np.array(dones),       dtype=torch.float32).to(device),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self.buffer) >= min_size
