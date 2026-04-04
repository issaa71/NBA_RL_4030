"""
demo.py
=======
Live demo script for NBA Shot Selection RL project.
Loads trained agents and runs them on test possessions, showing
Q-values and decisions at each step.

Usage:
    python demo.py --agent dqn
    python demo.py --agent dueling
    python demo.py --agent both

AISE 4030 - Group 11
"""

import argparse
import pickle
import numpy as np
import torch

from environment import NBAShootOrPassEnv, load_and_preprocess_dataset
from dqn_agent import DQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from utils import load_config, get_device


ACTION_NAMES = ['SHOOT', 'PASS 1', 'PASS 2', 'PASS 3', 'PASS 4']

FEATURE_NAMES = [
    'grid_zone', 'dist_basket', 'def_dist', 'defs_6ft',
    'tm_openness', 'open_tms', 'shot_clk', 'is_3pt',
    'bh_zone_fg', 'tm_dist_bask', 'tm_zone_fg'
]


def run_demo(agent, env, agent_name, num_possessions=5):
    """Run agent on possessions and display decisions."""
    print(f"\n{'='*60}")
    print(f"  {agent_name} — Live Demo")
    print(f"{'='*60}")

    device = next(agent.online_net.parameters()).device
    total_reward = 0.0

    for ep in range(num_possessions):
        state, _ = env.reset()
        poss = env.current_possession
        print(f"\n--- Possession {ep+1} ({len(poss)} steps in data) ---")

        done = False
        step = 0
        while not done:
            # Get Q-values
            state_tensor = torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.online_net(state_tensor).cpu().numpy()[0]

            action = int(np.argmax(q_values))
            q_str = ' | '.join(
                f'{ACTION_NAMES[i]}={q_values[i]:+.3f}'
                for i in range(len(ACTION_NAMES))
            )

            # Raw state info (denormalized approximately)
            raw_point = poss[min(step, len(poss) - 1)]
            dist = raw_point.get('distance_to_basket', 0)
            def_d = raw_point.get('closest_defender_dist', 0)
            fg = raw_point.get('ball_handler_zone_fg_pct', 0)
            clk = raw_point.get('shot_clock', 0)

            print(f"  Step {step}: dist={dist:.0f}ft | "
                  f"def={def_d:.1f}ft | fg={fg:.0%} | "
                  f"clk={clk:.0f}s")
            print(f"    Q: {q_str}")
            print(f"    => {ACTION_NAMES[action]}")

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                if action == 0:
                    # Simulate make/miss for presentation using zone FG%
                    fg = raw_point.get('ball_handler_zone_fg_pct', 0.40)
                    made = np.random.random() < fg
                    pts = 3 if raw_point.get('is_three', False) else 2
                    outcome = f"{'MADE' if made else 'MISSED'} {pts}PT"
                elif reward <= -1.0:
                    outcome = "TURNOVER"
                else:
                    outcome = "END"
                print(f"    Result: {outcome} (EPSA: {reward:+.2f})")

            state = next_state
            step += 1

    avg = total_reward / num_possessions
    print(f"\n  Average: {avg:.3f} EPSA over {num_possessions} possessions")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--agent', default='both',
                        choices=['dqn', 'dueling', 'both'])
    parser.add_argument('--num', type=int, default=5,
                        help='Number of demo possessions')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    # Load shot model
    shot_model = None
    try:
        with open(config['environment']['shot_model_path'], 'rb') as f:
            shot_model = pickle.load(f)
    except FileNotFoundError:
        pass

    _, test_possessions = load_and_preprocess_dataset(config)
    env = NBAShootOrPassEnv(
        test_possessions, config, shot_model=shot_model
    )

    print("=" * 60)
    print("NBA Shot Selection — Live Demo")
    print(f"Test possessions: {len(test_possessions)}")
    print(f"Shot model: {'Stochastic' if shot_model else 'Historical'}")
    print(f"Device: {device}")
    print("=" * 60)

    if args.agent in ('dqn', 'both'):
        dqn_agent = DQNAgent(config, device)
        dqn_agent.load('dqn_results/dqn_weights.pth')
        run_demo(dqn_agent, env, 'DQN', args.num)

    if args.agent in ('dueling', 'both'):
        dueling_agent = DuelingDQNAgent(config, device)
        dueling_agent.load('dueling_dqn_results/dueling_dqn_weights.pth')
        run_demo(dueling_agent, env, 'Dueling DQN', args.num)

    env.close()


if __name__ == '__main__':
    main()
