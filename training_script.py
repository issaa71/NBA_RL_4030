"""
training_script.py
==================
Main entry point for training, evaluation, and deployment of the DQN
and Dueling DQN shot-selection agents on NBA possession data.

Running this script with mode='train' will:
  1. Print environment diagnostics.
  2. Load processed SportVU possession data and shot probability model.
  3. Initialize the selected agent (DQN or Dueling DQN).
  4. Run the full training loop across all episodes.
  5. Periodically evaluate the agent and save checkpoints.
  6. Save the final model, training history, and plots.

Usage:
    python training_script.py --config config.yaml --mode train
    python training_script.py --config config.yaml --mode train --agent dueling
    python training_script.py --config config.yaml --mode eval
    python training_script.py --config config.yaml --mode deploy

AISE 4030 - Group 11
"""

import argparse
import pickle
import torch
import numpy as np

from environment import make_env, load_and_preprocess_dataset, NBAShootOrPassEnv
from dqn_agent import DQNAgent
from utils import (
    load_config,
    set_seed,
    get_device,
    setup_results_dir,
    save_training_history,
    plot_learning_curve,
    log_episode,
)


# =============================================================================
# Environment Diagnostics
# =============================================================================

def print_environment_diagnostics(
    env: NBAShootOrPassEnv, device: torch.device
) -> None:
    """
    Prints environment diagnostics to confirm the observation space, action
    space, and compute device.

    Args:
        env (NBAShootOrPassEnv): An initialized environment instance.
        device (torch.device): The selected compute device.

    Returns:
        None
    """
    obs_space = env.observation_space
    act_space = env.action_space

    print("=" * 60)
    print("NBA Shot Selection Environment - API Confirmation")
    print("=" * 60)
    print("Observation (State) Space:")
    print(f"  Shape      : {obs_space.shape}")
    print(f"  Dtype      : {obs_space.dtype}")
    print(f"  Low bounds : {obs_space.low}")
    print(f"  High bounds: {obs_space.high}")
    print(f"  Features   : 78D (73 continuous + 5 player IDs)")
    print(f"                See ARCHITECTURE.md for full state layout")
    print("Action Space:")
    print(f"  Type       : Discrete")
    print(f"  Num actions: {act_space.n}")
    print(f"  Actions    : {{0: shoot, 1-4: pass to teammate 1-4}}")
    print(f"Device       : {device}")
    print(f"Shot model   : {'Loaded (stochastic)' if env.shot_model else 'None (historical)'}")
    print("=" * 60)


def run_env_step_test(env: NBAShootOrPassEnv) -> None:
    """
    Executes a single environment reset and step to confirm the environment
    API is functional.

    Args:
        env (NBAShootOrPassEnv): An initialized environment instance.

    Returns:
        None

    Raises:
        AssertionError: If the step output does not match expected format.
    """
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, \
        f"Shape mismatch: got {obs.shape}, expected {env.observation_space.shape}"
    assert obs.dtype == np.float32, \
        f"Dtype mismatch: got {obs.dtype}, expected float32"

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    assert obs2.shape == env.observation_space.shape, "Next obs shape mismatch"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(terminated, bool), "terminated should be a bool"

    print(f"Env step test : PASSED")
    print(f"  Sample obs  : {obs}")
    print(f"  Action taken: {action}  "
          f"({'shoot' if action == 0 else f'pass_p{action}'})")
    print(f"  Reward      : {reward}")
    print(f"  Terminated  : {terminated}")
    print(f"  Next obs    : {obs2}")
    print("=" * 60)


def load_shot_model(config: dict):
    """
    Loads the trained shot probability model from disk.

    Args:
        config (dict): Config dict with 'environment.shot_model_path'.

    Returns:
        sklearn model or None: The trained model, or None if file not found.
    """
    model_path = config["environment"].get(
        "shot_model_path", "shot_probability_model.pkl"
    )
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Shot probability model loaded from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Warning: Shot model not found at {model_path}. "
              f"Using historical outcomes.")
        return None


def create_agent(agent_type: str, config: dict, device: torch.device):
    """
    Creates the appropriate agent based on the agent type string.

    Args:
        agent_type (str): Either 'dqn' or 'dueling'.
        config (dict): Full configuration dictionary.
        device (torch.device): Compute device.

    Returns:
        DQNAgent or DuelingDQNAgent: The initialized agent.
    """
    if agent_type == "dueling":
        from dueling_dqn_agent import DuelingDQNAgent
        return DuelingDQNAgent(config, device)
    elif agent_type == "bcq":
        from bcq_agent import BCQAgent
        return BCQAgent(config, device)
    else:
        return DQNAgent(config, device)


def get_results_config(agent_type: str, config: dict) -> dict:
    """
    Returns the appropriate results directory and paths for the agent type.

    Args:
        agent_type (str): Either 'dqn' or 'dueling'.
        config (dict): Full configuration dictionary.

    Returns:
        dict: Keys 'results_dir', 'model_save', 'training_history',
            'plots_dir'.
    """
    if agent_type == "dueling":
        dueling_cfg = config.get("dueling", {})
        return {
            'results_dir': dueling_cfg.get(
                "results_dir", "dueling_dqn_results"
            ),
            'model_save': dueling_cfg.get(
                "model_save", "dueling_dqn_results/dueling_dqn_weights.pth"
            ),
            'training_history': dueling_cfg.get(
                "training_history",
                "dueling_dqn_results/training_history.json"
            ),
            'plots_dir': dueling_cfg.get(
                "plots_dir", "dueling_dqn_results/plots"
            ),
        }
    elif agent_type == "bcq":
        bcq_cfg = config.get("bcq", {})
        return {
            'results_dir': bcq_cfg.get("results_dir", "bcq_results"),
            'model_save': bcq_cfg.get("model_save", "bcq_results/bcq_weights.pth"),
            'training_history': bcq_cfg.get("training_history", "bcq_results/training_history.json"),
            'plots_dir': bcq_cfg.get("plots_dir", "bcq_results/plots"),
        }
    else:
        return {
            'results_dir': config["training"]["results_dir"],
            'model_save': config["paths"]["model_save"],
            'training_history': config["paths"]["training_history"],
            'plots_dir': config["paths"]["plots_dir"],
        }


# =============================================================================
# Training
# =============================================================================

def train(config: dict, agent_type: str = "dqn") -> None:
    """
    Full training loop for the DQN or Dueling DQN shot-selection agent.

    Args:
        config (dict): Full configuration dictionary from config.yaml.
        agent_type (str): Agent type ('dqn' or 'dueling').

    Returns:
        None
    """
    device = get_device()
    set_seed(config["environment"]["random_seed"])

    res_cfg = get_results_config(agent_type, config)
    setup_results_dir(res_cfg['results_dir'])

    # Load shot probability model for stochastic rewards
    shot_model = load_shot_model(config)

    train_possessions, test_possessions = load_and_preprocess_dataset(config)
    env = NBAShootOrPassEnv(
        train_possessions, config, shot_model=shot_model
    )

    # Environment diagnostics
    print_environment_diagnostics(env, device)
    run_env_step_test(env)

    agent = create_agent(agent_type, config, device)
    print(f"\nAgent type: {agent_type.upper()}")

    episode_rewards = []
    episode_losses = []
    episode_lengths = []
    episode_epsilons = []

    num_episodes = config["training"]["num_episodes"]
    max_steps = config["training"]["max_steps_per_episode"]
    eval_freq = config["training"]["eval_frequency"]
    checkpoint_freq = config["training"]["checkpoint_frequency"]

    best_eval_score = -float('inf')

    print(f"Starting training for {num_episodes} episodes...")
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss_sum = 0.0
        step_count = 0
        loss_count = 0

        for step in range(max_steps):
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store terminated (not truncated) so Q-bootstrap isn't
            # zeroed on truncation — only on true episode ends
            agent.store_transition(state, action, reward, next_state, float(terminated))
            loss = agent.update()

            if loss is not None:
                episode_loss_sum += loss
                loss_count += 1

            episode_reward += reward
            state = next_state
            step_count += 1

            if done:
                break

        agent.on_episode_end()

        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss_sum / max(loss_count, 1))
        episode_lengths.append(step_count)
        episode_epsilons.append(agent.epsilon)

        log_episode(episode, episode_reward, step_count,
                    agent.epsilon, episode_losses[-1])

        if episode % eval_freq == 0:
            eval_env = NBAShootOrPassEnv(
                test_possessions, config, shot_model=shot_model
            )
            eval_score = evaluate(agent, config, eval_env)
            eval_env.close()
            marker = ""
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                agent.save(res_cfg['model_save'])
                marker = " *** BEST ***"
            print(f"  [Eval] Episode {episode} | "
                  f"EPSA: {eval_score:.4f}{marker}")

        if episode % checkpoint_freq == 0:
            agent.save(res_cfg['model_save'])
            print(f"  [Checkpoint saved at episode {episode}]")

    # Keep best checkpoint (already saved during training), don't overwrite with final
    print(f"  Best eval EPSA: {best_eval_score:.4f}")
    save_training_history(
        episode_rewards, episode_losses, res_cfg['training_history'],
        episode_lengths=episode_lengths, episode_epsilons=episode_epsilons
    )
    plot_learning_curve(
        episode_rewards,
        config["training"]["log_window"],
        res_cfg['plots_dir']
    )
    env.close()
    print("\nTraining complete.")
    print(f"Model saved to: {res_cfg['model_save']}")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(agent, config: dict, eval_env: NBAShootOrPassEnv) -> float:
    """
    Evaluates the agent on the held-out test set in pure exploitation mode
    (epsilon = 0, no exploration).

    Args:
        agent: The trained or partially trained DQN/Dueling DQN agent.
        config (dict): Full configuration dictionary.
        eval_env (NBAShootOrPassEnv): Environment with test possessions.

    Returns:
        float: Mean points per possession over all evaluated episodes.
    """
    num_eval = config["training"]["eval_episodes"]
    total_reward = 0.0

    for _ in range(num_eval):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward

    return total_reward / num_eval


def deploy(config: dict, agent_type: str = "dqn") -> None:
    """
    Loads a saved model and runs it on test possessions in pure
    exploitation mode, printing a summary of decisions made.

    Args:
        config (dict): Full configuration dictionary from config.yaml.
        agent_type (str): Agent type ('dqn' or 'dueling').

    Returns:
        None
    """
    device = get_device()
    res_cfg = get_results_config(agent_type, config)

    shot_model = load_shot_model(config)
    _, test_possessions = load_and_preprocess_dataset(config)

    agent = create_agent(agent_type, config, device)
    agent.load(res_cfg['model_save'])

    eval_env = NBAShootOrPassEnv(
        test_possessions, config, shot_model=shot_model
    )
    score = evaluate(agent, config, eval_env)
    eval_env.close()
    print(f"Deployment evaluation: {score:.4f} avg points/possession")


# =============================================================================
# Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - config (str): Path to the config YAML file.
            - mode (str): Execution mode ('train', 'eval', or 'deploy').
            - agent (str): Agent type ('dqn' or 'dueling').
    """
    parser = argparse.ArgumentParser(
        description="NBA Shot Selection DQN - AISE 4030 Group 11"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the config YAML file."
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "eval", "deploy"],
        default="train",
        help="Execution mode: 'train', 'eval', or 'deploy'."
    )
    parser.add_argument(
        "--agent", type=str, choices=["dqn", "dueling", "bcq"], default="dqn",
        help="Agent type: 'dqn' or 'dueling'."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        train(cfg, agent_type=args.agent)
    elif args.mode == "eval":
        device = get_device()
        res_cfg = get_results_config(args.agent, cfg)
        shot_model = load_shot_model(cfg)
        _, test_possessions = load_and_preprocess_dataset(cfg)
        agent = create_agent(args.agent, cfg, device)
        agent.load(res_cfg['model_save'])
        eval_env = NBAShootOrPassEnv(
            test_possessions, cfg, shot_model=shot_model
        )
        score = evaluate(agent, cfg, eval_env)
        eval_env.close()
        print(f"Evaluation result: {score:.4f} avg points/possession")
    elif args.mode == "deploy":
        deploy(cfg, agent_type=args.agent)
