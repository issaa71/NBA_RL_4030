"""
evaluate_and_compare.py
=======================
Evaluates trained DQN and Dueling DQN agents and generates all comparison
plots required for the Phase 3 submission rubric.

Generates:
  1. Learning speed — overlaid smoothed reward curves
  2. Loss convergence — overlaid smoothed loss curves
  3. Final performance — bar chart with error bars + baselines
  4. Stability/variance — shaded variance regions on reward curves
  5. Decision maps — 10x5 heatmap of recommended action per zone
  6. Epsilon decay curve

Usage:
    python evaluate_and_compare.py --config config.yaml

AISE 4030 - Group 11
"""

import argparse
import json
import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from environment import NBAShootOrPassEnv, load_and_preprocess_dataset
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from utils import load_config, get_device, rolling_average


# ============================================================
# Colors & Style
# ============================================================

DQN_COLOR = '#2196F3'
DUELING_COLOR = '#FF5722'
BASELINE_COLORS = {'random': '#9E9E9E', 'always_shoot': '#4CAF50', 'behavior': '#FFC107'}
DPI = 300
SMOOTH_WINDOW = 200


# ============================================================
# Baselines
# ============================================================

def eval_random_policy(env, num_episodes=500):
    """Evaluate a uniform random policy."""
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return np.array(rewards)


def eval_always_shoot(env, num_episodes=500):
    """Evaluate an always-shoot policy (action 0 every step)."""
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            state, reward, terminated, truncated, _ = env.step(0)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return np.array(rewards)


def eval_behavior_policy(env, num_episodes=500):
    """Evaluate the behavior policy (replay historical actions from data)."""
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        poss = env.current_possession
        ep_reward = 0.0
        done = False
        step_idx = 0
        while not done and step_idx < len(poss):
            historical_action = poss[step_idx]['action']
            action = 0 if historical_action == 0 else 1
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step_idx += 1
        rewards.append(ep_reward)
    return np.array(rewards)


def eval_agent(agent, env, num_episodes=500):
    """Evaluate a trained agent (greedy, no exploration)."""
    rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return np.array(rewards)


# ============================================================
# Plot Functions
# ============================================================

def plot_learning_speed(dqn_hist, dueling_hist, plots_dir):
    """Plot 1: Overlaid smoothed reward curves."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for hist, color, label in [
        (dqn_hist, DQN_COLOR, 'DQN'),
        (dueling_hist, DUELING_COLOR, 'Dueling DQN'),
    ]:
        rewards = hist['episode_rewards']
        episodes = range(1, len(rewards) + 1)
        smoothed = rolling_average(rewards, SMOOTH_WINDOW)
        ax.plot(episodes, rewards, alpha=0.08, color=color)
        ax.plot(episodes, smoothed, color=color, linewidth=2, label=label)

    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.6,
               label='League-avg baseline (EPSA = 0)')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Expected Points per Shot Attempt (EPSA)', fontsize=12)
    ax.set_title('Learning Speed — DQN vs Dueling DQN', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'learning_speed.png'), dpi=DPI)
    plt.close()
    print("  Saved learning_speed.png")


def plot_loss_convergence(dqn_hist, dueling_hist, plots_dir):
    """Plot 2: Overlaid smoothed loss curves."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for hist, color, label in [
        (dqn_hist, DQN_COLOR, 'DQN'),
        (dueling_hist, DUELING_COLOR, 'Dueling DQN'),
    ]:
        losses = hist['episode_losses']
        episodes = range(1, len(losses) + 1)
        smoothed = rolling_average(losses, SMOOTH_WINDOW)
        ax.plot(episodes, losses, alpha=0.08, color=color)
        ax.plot(episodes, smoothed, color=color, linewidth=2, label=label)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Loss per Episode', fontsize=12)
    ax.set_title('Loss Convergence — DQN vs Dueling DQN', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_convergence.png'), dpi=DPI)
    plt.close()
    print("  Saved loss_convergence.png")


def plot_final_performance(results_dict, plots_dir):
    """Plot 3: Bar chart with error bars for all policies."""
    names = list(results_dict.keys())
    means = [np.mean(v) for v in results_dict.values()]
    stds = [np.std(v) for v in results_dict.values()]

    colors = []
    for name in names:
        if name == 'DQN':
            colors.append(DQN_COLOR)
        elif name == 'Dueling DQN':
            colors.append(DUELING_COLOR)
        else:
            colors.append(BASELINE_COLORS.get(name.lower().replace(' ', '_'),
                                               '#9E9E9E'))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.6,
               label='League-avg baseline (EPSA = 0)')
    ax.set_ylabel('Avg EPSA', fontsize=12)
    ax.set_title('Final Performance — All Policies (500 episodes)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'final_performance.png'), dpi=DPI)
    plt.close()
    print("  Saved final_performance.png")


def plot_stability(dqn_hist, dueling_hist, plots_dir, window=200):
    """Plot 4: Shaded variance regions on reward curves."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for hist, color, label in [
        (dqn_hist, DQN_COLOR, 'DQN'),
        (dueling_hist, DUELING_COLOR, 'Dueling DQN'),
    ]:
        rewards = np.array(hist['episode_rewards'])
        n = len(rewards)
        means = []
        stds = []
        episodes = []
        for i in range(n):
            start = max(0, i - window + 1)
            chunk = rewards[start:i + 1]
            means.append(np.mean(chunk))
            stds.append(np.std(chunk))
            episodes.append(i + 1)

        means = np.array(means)
        stds = np.array(stds)
        episodes = np.array(episodes)

        ax.plot(episodes, means, color=color, linewidth=2, label=label)
        ax.fill_between(episodes, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('EPSA', fontsize=12)
    ax.set_title('Stability — Rolling Mean ± Std Dev', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'stability_variance.png'), dpi=DPI)
    plt.close()
    print("  Saved stability_variance.png")


def plot_epsilon_decay(dqn_hist, dueling_hist, plots_dir):
    """Plot epsilon over training."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for hist, color, label in [
        (dqn_hist, DQN_COLOR, 'DQN'),
        (dueling_hist, DUELING_COLOR, 'Dueling DQN'),
    ]:
        epsilons = hist.get('episode_epsilons', [])
        if epsilons:
            ax.plot(range(1, len(epsilons) + 1), epsilons,
                    color=color, linewidth=2, label=label)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon', fontsize=12)
    ax.set_title('Exploration Rate (Epsilon) Decay', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'epsilon_decay.png'), dpi=DPI)
    plt.close()
    print("  Saved epsilon_decay.png")


def plot_episode_lengths(dqn_hist, dueling_hist, plots_dir):
    """Plot episode length over training."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for hist, color, label in [
        (dqn_hist, DQN_COLOR, 'DQN'),
        (dueling_hist, DUELING_COLOR, 'Dueling DQN'),
    ]:
        lengths = hist.get('episode_lengths', [])
        if lengths:
            smoothed = rolling_average(lengths, SMOOTH_WINDOW)
            ax.plot(range(1, len(lengths) + 1), smoothed,
                    color=color, linewidth=2, label=label)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps per Episode', fontsize=12)
    ax.set_title('Episode Length Over Training', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'episode_lengths.png'), dpi=DPI)
    plt.close()
    print("  Saved episode_lengths.png")


def plot_decision_maps(dqn_agent, dueling_agent, env, config, plots_dir):
    """Plot 5: Shoot-probability heatmaps from real test data using full 78D state."""
    device = next(dqn_agent.online_net.parameters()).device

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, agent, title in [
        (axes[0], dqn_agent, 'DQN'),
        (axes[1], dueling_agent, 'Dueling DQN'),
    ]:
        # Aggregate shoot/pass decisions on real test possessions
        shoot_counts = np.zeros((5, 10), dtype=float)
        total_counts = np.zeros((5, 10), dtype=float)

        for poss in env.possessions:
            for dp in poss:
                zone = int(dp['grid_zone'])
                row = zone // 10
                col = zone % 10
                if row >= 5 or col >= 10:
                    continue

                # Build full 78D state using the environment's _build_state
                state = env._build_state(dp)
                state_tensor = torch.tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    q_values = agent.online_net(state_tensor)
                action = q_values.argmax().item()

                total_counts[row, col] += 1
                if action == 0:
                    shoot_counts[row, col] += 1

        # Compute shoot probability per zone
        shoot_prob = np.divide(
            shoot_counts, total_counts,
            out=np.full_like(shoot_counts, 0.5),
            where=total_counts > 0
        )

        im = ax.imshow(shoot_prob, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=1, origin='lower')
        ax.set_title(f'{title} — Shoot Probability by Zone', fontsize=12)
        ax.set_xlabel('Court X Zone (0-9)', fontsize=10)
        ax.set_ylabel('Court Y Zone (0-4)', fontsize=10)

        for r in range(5):
            for c in range(10):
                if total_counts[r, c] > 0:
                    ax.text(c, r, f'{shoot_prob[r, c]:.0%}',
                            ha='center', va='center', fontsize=6,
                            fontweight='bold',
                            color='white' if shoot_prob[r, c] > 0.6
                            or shoot_prob[r, c] < 0.3 else 'black')

    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label('P(Shoot)')
    plt.suptitle('Decision Maps — Shoot Probability from Real Test Data',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'decision_maps.png'),
                dpi=DPI, bbox_inches='tight')
    plt.close()
    print("  Saved decision_maps.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Create comparison plots directory
    plots_dir = 'comparison_plots'
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 3 Evaluation & Comparison")
    print("=" * 60)

    # --- Load training histories ---
    print("\n1. Loading training histories...")
    with open('results_v9/pbrs_dqn_nodist/training_history.json') as f:
        dqn_hist = json.load(f)
    with open('results_v9/pbrs_lr1e4/training_history.json') as f:
        dueling_hist = json.load(f)
    print(f"   DQN: {len(dqn_hist['episode_rewards'])} episodes")
    print(f"   Dueling: {len(dueling_hist['episode_rewards'])} episodes")

    # --- Generate comparison plots ---
    print("\n2. Generating comparison plots...")
    plot_learning_speed(dqn_hist, dueling_hist, plots_dir)
    plot_loss_convergence(dqn_hist, dueling_hist, plots_dir)
    plot_stability(dqn_hist, dueling_hist, plots_dir)
    plot_epsilon_decay(dqn_hist, dueling_hist, plots_dir)
    plot_episode_lengths(dqn_hist, dueling_hist, plots_dir)

    # --- Load agents and environment for evaluation ---
    print("\n3. Loading agents for evaluation...")
    shot_model = None
    shot_model_path = config['environment'].get(
        'shot_model_path', 'shot_probability_model.pkl'
    )
    if os.path.exists(shot_model_path):
        with open(shot_model_path, 'rb') as f:
            shot_model = pickle.load(f)

    _, test_possessions = load_and_preprocess_dataset(config)
    eval_env = NBAShootOrPassEnv(
        test_possessions, config, shot_model=shot_model
    )

    dqn_agent = DQNAgent(config, device)
    dqn_agent.load('results_v9/pbrs_dqn_nodist/dqn_weights.pth')

    dueling_agent = DuelingDQNAgent(config, device)
    dueling_agent.load('results_v9/pbrs_lr1e4/dueling_dqn_weights.pth')

    # --- Evaluate all policies ---
    print("\n4. Evaluating policies (500 episodes each)...")
    num_eval = 500

    print("   Random policy...")
    random_rewards = eval_random_policy(eval_env, num_eval)
    print(f"     Mean: {random_rewards.mean():.3f} ± {random_rewards.std():.3f}")

    print("   Always-shoot policy...")
    shoot_rewards = eval_always_shoot(eval_env, num_eval)
    print(f"     Mean: {shoot_rewards.mean():.3f} ± {shoot_rewards.std():.3f}")

    print("   Behavior policy (historical)...")
    behavior_rewards = eval_behavior_policy(eval_env, num_eval)
    print(f"     Mean: {behavior_rewards.mean():.3f} ± {behavior_rewards.std():.3f}")

    print("   DQN agent...")
    dqn_rewards = eval_agent(dqn_agent, eval_env, num_eval)
    print(f"     Mean: {dqn_rewards.mean():.3f} ± {dqn_rewards.std():.3f}")

    print("   Dueling DQN agent...")
    dueling_rewards = eval_agent(dueling_agent, eval_env, num_eval)
    print(f"     Mean: {dueling_rewards.mean():.3f} ± {dueling_rewards.std():.3f}")

    # Final performance bar chart
    results = {
        'Random': random_rewards,
        'Always Shoot': shoot_rewards,
        'Behavior': behavior_rewards,
        'DQN': dqn_rewards,
        'Dueling DQN': dueling_rewards,
    }
    plot_final_performance(results, plots_dir)

    # --- Decision maps ---
    print("\n5. Generating decision maps...")
    plot_decision_maps(dqn_agent, dueling_agent, eval_env, config, plots_dir)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Policy':<20s} {'Mean EPSA':>10s} {'Std':>8s}")
    print("-" * 40)
    for name, r in results.items():
        print(f"{name:<20s} {np.mean(r):>10.3f} {np.std(r):>8.3f}")
    print("=" * 60)

    # Save numerical results
    summary = {name: {'mean': float(np.mean(r)), 'std': float(np.std(r)),
                       'rewards': r.tolist()}
               for name, r in results.items()}
    with open(os.path.join(plots_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll plots saved to {plots_dir}/")
    print(f"Numerical results saved to {plots_dir}/evaluation_results.json")

    eval_env.close()


if __name__ == '__main__':
    main()
