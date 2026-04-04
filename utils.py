"""
utils.py
========
Shared utility functions for the NBA Shot Selection DQN project.

Provides helpers for:
  - Config loading (YAML)
  - Device detection (CPU / CUDA / MPS)
  - Random seed management
  - Results directory setup
  - Training history persistence (JSON)
  - Learning curve plotting (matplotlib)
  - Episode logging

AISE 4030 - Group 11
"""

import os
import json
import random
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from typing import List, Optional


def load_config(path: str) -> dict:
    """
    Loads a YAML configuration file and returns it as a Python dict.

    Args:
        path (str): Relative or absolute path to the config.yaml file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist at the given path.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """
    Detects and returns the best available compute device.

    Priority order: CUDA GPU > Apple MPS > CPU.

    Returns:
        torch.device: The selected device (cuda, mps, or cpu).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def set_seed(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure
    reproducible results across training runs.

    Args:
        seed (int): The integer seed value (e.g., 42 from config.yaml).

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_results_dir(results_dir: str) -> None:
    """
    Creates the results output directory and its subdirectories (plots/)
    if they do not already exist.

    Args:
        results_dir (str): Path to the results directory (e.g., 'dqn_results').

    Returns:
        None
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)


def save_training_history(
    episode_rewards: List[float],
    episode_losses: List[float],
    path: str,
    episode_lengths: Optional[List[int]] = None,
    episode_epsilons: Optional[List[float]] = None,
) -> None:
    """
    Saves training history to a JSON file for later analysis.

    Args:
        episode_rewards (list of float): Total reward per training episode.
        episode_losses (list of float): Mean training loss per episode.
        path (str): File path for the JSON output.
        episode_lengths (list of int, optional): Steps per episode.
        episode_epsilons (list of float, optional): Epsilon per episode.

    Returns:
        None
    """
    history = {
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
    }
    if episode_lengths is not None:
        history["episode_lengths"] = episode_lengths
    if episode_epsilons is not None:
        history["episode_epsilons"] = episode_epsilons
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def load_training_history(path: str) -> dict:
    """
    Loads a saved training history JSON file.

    Args:
        path (str): File path to the saved training_history.json.

    Returns:
        dict: Dictionary with keys 'episode_rewards' and 'episode_losses'.
    """
    with open(path, 'r') as f:
        return json.load(f)


def plot_learning_curve(
    episode_rewards: List[float],
    window: int,
    plots_dir: str,
    filename: str = "learning_curve.png"
) -> None:
    """
    Plots and saves the rolling-average learning curve (points per possession
    vs. episodes). Uses a rolling window of 'window' episodes, matching the
    primary evaluation metric from Phase 1 Section 3.4.

    Also plots a horizontal dashed reference line at 1.02 (the 2014-15 NBA
    average of 1.02 expected points per possession).

    Args:
        episode_rewards (list of float): Per-episode total rewards from training.
        window (int): Rolling average window size (e.g., 500 from config.yaml).
        plots_dir (str): Directory to save the plot image.
        filename (str): Output filename (default: 'learning_curve.png').

    Returns:
        None
    """
    smoothed = rolling_average(episode_rewards, window)
    episodes = list(range(1, len(episode_rewards) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards, alpha=0.2, color='steelblue', label='Raw reward')
    plt.plot(episodes, smoothed, color='steelblue', linewidth=2,
             label=f'Rolling avg (w={window})')
    plt.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5,
                label='League-avg baseline (EPSA = 0)')
    plt.xlabel("Episode")
    plt.ylabel("EPSA")
    plt.title("DQN Learning Curve — NBA Shot Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close()


def plot_decision_map(
    q_values_by_zone: np.ndarray,
    action_names: List[str],
    plots_dir: str,
    filename: str = "decision_map.png"
) -> None:
    """
    Plots a 10x5 half-court heatmap showing the agent's recommended action
    (argmax Q-value) for each grid zone.

    Args:
        q_values_by_zone (np.ndarray): Array of shape (50, 5).
        action_names (list of str): Display names for each action.
        plots_dir (str): Directory to save the plot image.
        filename (str): Output filename.

    Returns:
        None
    """
    best_actions = np.argmax(q_values_by_zone, axis=1).reshape(5, 10)
    plt.figure(figsize=(12, 6))
    plt.imshow(best_actions, cmap='tab10', aspect='auto',
               vmin=0, vmax=len(action_names) - 1)
    plt.colorbar(ticks=range(len(action_names)), label='Action')
    plt.title("DQN Decision Map — Recommended Action by Court Zone")
    plt.xlabel("Court X Zone (0–9)")
    plt.ylabel("Court Y Zone (0–4)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close()


def log_episode(
    episode: int,
    reward: float,
    steps: int,
    epsilon: float,
    loss: float,
    log_freq: int = 10
) -> None:
    """
    Logs episode metrics to stdout at a fixed frequency.

    Args:
        episode (int): Current episode number.
        reward (float): Total reward for the episode.
        steps (int): Number of steps in the episode.
        epsilon (float): Current exploration rate.
        loss (float): Mean training loss for the episode.
        log_freq (int): Print every log_freq episodes (default: 10).

    Returns:
        None
    """
    if episode % log_freq == 0:
        print(
            f"Episode {episode:5d} | "
            f"Reward: {reward:6.3f} | "
            f"Steps: {steps:3d} | "
            f"Epsilon: {epsilon:.4f} | "
            f"Loss: {loss:.6f}"
        )


def rolling_average(values: List[float], window: int) -> List[float]:
    """
    Computes the rolling average of a list of values with a given window size.

    Args:
        values (list of float): Raw per-episode metric values.
        window (int): Window size for the rolling average.

    Returns:
        list of float: Rolling averages of the same length as input.
    """
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start:i + 1])))
    return result
