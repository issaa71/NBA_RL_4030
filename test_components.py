"""
test_components.py
==================
Tests for the core components of the NBA Shot Selection RL project.
Covers environment, agents, pipeline helpers, and integration.

Run: python -m pytest test_components.py -v
"""

import numpy as np
import pytest
import pickle
import os
import yaml

# ============================================================
# Pipeline helper tests
# ============================================================

class TestDistToZoneIndex:
    """Tests for the dist_to_zone_index helper in 02_segment_possessions.py."""

    def test_at_rim(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        assert mod.dist_to_zone_index(0.0) == 0
        assert mod.dist_to_zone_index(4.9) == 0

    def test_short_midrange(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        assert mod.dist_to_zone_index(5.0) == 1
        assert mod.dist_to_zone_index(9.9) == 1

    def test_three_point_range(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        assert mod.dist_to_zone_index(20.0) == 4
        assert mod.dist_to_zone_index(24.9) == 4

    def test_deep_three(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        assert mod.dist_to_zone_index(25.0) == 5
        assert mod.dist_to_zone_index(29.9) == 5

    def test_heave(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        assert mod.dist_to_zone_index(30.0) == 6
        assert mod.dist_to_zone_index(50.0) == 6


class TestGridZone:
    """Tests for xy_to_grid_zone."""

    def test_origin(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        assert mod.xy_to_grid_zone(0.0, 0.0) == 0

    def test_max_corner(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        zone = mod.xy_to_grid_zone(47.0, 50.0)
        assert zone == 49  # last zone

    def test_clipping(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        # Should clip negative values
        zone = mod.xy_to_grid_zone(-5.0, -5.0)
        assert zone == 0


class TestDefenderFeatures:
    """Tests for compute_defender_features."""

    def test_no_defenders(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        result = mod.compute_defender_features(
            np.array([10.0, 25.0]),
            [np.array([20.0, 25.0])],
            []
        )
        assert result['closest_defender_dist'] == 30.0
        assert result['num_defenders_within_6ft'] == 0

    def test_close_defender(self):
        from importlib import import_module
        mod = import_module("02_segment_possessions")
        result = mod.compute_defender_features(
            np.array([10.0, 25.0]),
            [np.array([20.0, 25.0])],
            [np.array([12.0, 25.0])]
        )
        assert result['closest_defender_dist'] == pytest.approx(2.0)
        assert result['num_defenders_within_6ft'] == 1


# ============================================================
# Environment tests
# ============================================================

def _make_test_config():
    """Create a minimal config for testing."""
    return {
        "environment": {
            "processed_data_path": "processed_possessions.pkl",
            "shot_model_path": "shot_probability_model.pkl",
            "grid_rows": 5,
            "grid_cols": 10,
            "num_actions": 5,
            "shot_clock_max": 24.0,
            "test_split": 0.2,
            "random_seed": 42,
        },
        "rewards": {
            "made_2pt": 2.0,
            "made_3pt": 3.0,
            "missed_shot": 0.0,
            "pass": -0.05,
            "turnover": -1.0,
        },
        "network": {
            "state_dim": 66,
            "continuous_dim": 61,
            "action_dim": 5,
            "hidden_layers": [512, 256, 128],
            "activation": "relu",
            "num_players": 500,
            "embed_dim": 8,
        },
        "dqn": {
            "learning_rate": 0.0001,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "epsilon_decay_type": "linear",
            "epsilon_decay_episodes": 10000,
            "batch_size": 64,
            "replay_buffer_capacity": 50000,
            "target_update_freq": 10,
            "tau": 0.005,
            "n_step": 3,
            "min_replay_size": 500,
            "grad_clip": 1.0,
        },
        "training": {
            "num_episodes": 100,
            "max_steps_per_episode": 50,
            "eval_frequency": 50,
            "eval_episodes": 10,
            "checkpoint_frequency": 50,
            "log_window": 20,
            "results_dir": "test_results",
        },
    }


def _make_test_possession():
    """Create a single test possession with v4 features."""
    v4_fields = {
        'help_defender_dist': 8.0,
        'teammate_openness': [6.0, 8.0, 4.0, 10.0],
        'teammate_zone_fg': [0.42, 0.45, 0.38, 0.40],
        'teammate_dist_to_basket': [12.0, 10.0, 20.0, 15.0],
        'defender_positions': [(10.0, 20.0), (15.0, 25.0), (20.0, 30.0), (30.0, 25.0), (40.0, 25.0)],
        'teammate_positions': [(12.0, 18.0), (8.0, 30.0), (25.0, 10.0), (20.0, 40.0)],
        'defender_velocities': [(1.0, -0.5), (0.5, 0.3), (-1.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        'teammate_velocities': [(2.0, 1.0), (-1.0, 0.5), (0.0, -2.0), (1.5, 0.0)],
        'ball_handler_vx': 2.5,
        'ball_handler_vy': -1.0,
        'ball_handler_player_id': 201939,
        'teammate_player_ids': [201566, 201567, 201568, 201569],
    }
    return [
        {
            'grid_zone': 24, 'distance_to_basket': 15.0,
            'closest_defender_dist': 5.0, 'num_defenders_within_6ft': 1,
            'best_teammate_openness': 8.0, 'num_open_teammates': 2,
            'shot_clock': 18.0, 'is_three_point_zone': 0,
            'ball_handler_zone_fg_pct': 0.42, 'best_open_teammate_dist_to_basket': 10.0,
            'best_open_teammate_zone_fg_pct': 0.45,
            'action': 1, 'shot_made': False, 'is_three': False, 'turnover': False,
            **v4_fields,
        },
        {
            'grid_zone': 10, 'distance_to_basket': 8.0,
            'closest_defender_dist': 3.0, 'num_defenders_within_6ft': 2,
            'best_teammate_openness': 12.0, 'num_open_teammates': 3,
            'shot_clock': 12.0, 'is_three_point_zone': 0,
            'ball_handler_zone_fg_pct': 0.55, 'best_open_teammate_dist_to_basket': 22.0,
            'best_open_teammate_zone_fg_pct': 0.38,
            'action': 0, 'shot_made': True, 'is_three': False, 'turnover': False,
            **v4_fields,
        },
    ]


class TestEnvironment:
    """Tests for the NBAShootOrPassEnv with 48D state."""

    def test_observation_shape(self):
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        possessions = [_make_test_possession()] * 10
        env = NBAShootOrPassEnv(possessions, config)
        obs, info = env.reset(seed=42)
        assert obs.shape == (66,), f"Expected (66,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_state_normalization(self):
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        possessions = [_make_test_possession()] * 10
        env = NBAShootOrPassEnv(possessions, config)
        obs, _ = env.reset(seed=42)
        # Continuous features (0:43) should be in [0, 1]
        assert np.all(obs[:61] >= 0.0), f"Continuous state has negative values: {obs[:61]}"
        assert np.all(obs[:61] <= 1.0), f"Continuous state exceeds 1.0: {obs[:61]}"
        # Player IDs (61:66) are raw ints, not normalized

    def test_shoot_action_terminates(self):
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        possessions = [_make_test_possession()] * 10
        env = NBAShootOrPassEnv(possessions, config)
        env.reset(seed=42)
        _, reward, terminated, truncated, _ = env.step(0)  # shoot
        assert terminated is True

    def test_pass_action_continues(self):
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        possessions = [_make_test_possession()] * 10
        env = NBAShootOrPassEnv(possessions, config)
        env.reset(seed=42)
        _, reward, terminated, truncated, _ = env.step(1)  # pass to teammate 1
        # Pass reward is teammate-quality-dependent, not flat -0.05
        assert isinstance(reward, float)
        # Should be in reasonable range (centered EPV * 0.3 + pass penalty)
        assert -0.5 < reward < 0.5

    def test_turnover_terminates(self):
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        v4 = {
            'help_defender_dist': 4.0,
            'teammate_openness': [5.0, 3.0, 2.0, 7.0],
            'teammate_zone_fg': [0.38, 0.35, 0.40, 0.42],
            'teammate_dist_to_basket': [25.0, 20.0, 15.0, 18.0],
            'defender_positions': [(5.0, 25.0)] * 5,
            'teammate_positions': [(12.0, 20.0)] * 4,
            'ball_handler_vx': 0.0, 'ball_handler_vy': 0.0,
            'ball_handler_player_id': 201939,
            'teammate_player_ids': [0, 0, 0, 0],
        }
        to_poss = [{
            'grid_zone': 15, 'distance_to_basket': 20.0,
            'closest_defender_dist': 2.0, 'num_defenders_within_6ft': 3,
            'best_teammate_openness': 5.0, 'num_open_teammates': 1,
            'shot_clock': 20.0, 'is_three_point_zone': 0,
            'ball_handler_zone_fg_pct': 0.38, 'best_open_teammate_dist_to_basket': 25.0,
            'best_open_teammate_zone_fg_pct': 0.35,
            'action': 1, 'shot_made': False, 'is_three': False, 'turnover': True,
            **v4,
        }]
        env = NBAShootOrPassEnv([to_poss] * 10, config)
        env.reset(seed=42)
        _, reward, terminated, _, _ = env.step(1)
        assert terminated is True
        assert reward == config["rewards"]["turnover"]

    def test_action_space(self):
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        possessions = [_make_test_possession()] * 10
        env = NBAShootOrPassEnv(possessions, config)
        assert env.action_space.n == 5

    def test_epv_reward(self):
        """Contest-adjusted, baseline-centered EPV reward."""
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        v4 = {
            'help_defender_dist': 10.0,
            'teammate_openness': [8.0, 6.0, 4.0, 10.0],
            'teammate_zone_fg': [0.40, 0.42, 0.38, 0.45],
            'teammate_dist_to_basket': [15.0, 12.0, 20.0, 10.0],
            'defender_positions': [(15.0, 25.0)] * 5,
            'teammate_positions': [(12.0, 20.0)] * 4,
            'ball_handler_vx': 0.0, 'ball_handler_vy': 0.0,
            'ball_handler_player_id': 201939,
            'teammate_player_ids': [0, 0, 0, 0],
        }
        poss = [{
            'grid_zone': 10, 'distance_to_basket': 5.0,
            'closest_defender_dist': 10.0, 'num_defenders_within_6ft': 0,
            'best_teammate_openness': 8.0, 'num_open_teammates': 2,
            'shot_clock': 15.0, 'is_three_point_zone': 0,
            'ball_handler_zone_fg_pct': 0.60, 'best_open_teammate_dist_to_basket': 15.0,
            'best_open_teammate_zone_fg_pct': 0.40,
            'action': 0, 'shot_made': True, 'is_three': False, 'turnover': False,
            **v4,
        }]
        env = NBAShootOrPassEnv([poss] * 10, config)
        env.reset(seed=42)
        _, reward, terminated, _, _ = env.step(0)
        # Pure EPV: (0.60 * 2.0 * min(10/6, 1.0)) - 0.375 = 1.2 - 0.375 = 0.825
        assert reward == pytest.approx(0.825, abs=0.01)
        assert terminated is True

    def test_contested_shot_reward(self):
        """Contested shot should have reduced (negative) reward."""
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        v4 = {
            'help_defender_dist': 5.0,
            'teammate_openness': [8.0, 6.0, 4.0, 10.0],
            'teammate_zone_fg': [0.40, 0.42, 0.38, 0.45],
            'teammate_dist_to_basket': [15.0, 12.0, 20.0, 10.0],
            'defender_positions': [(5.0, 25.0)] * 5,
            'teammate_positions': [(12.0, 20.0)] * 4,
            'ball_handler_vx': 0.0, 'ball_handler_vy': 0.0,
            'ball_handler_player_id': 201939,
            'teammate_player_ids': [0, 0, 0, 0],
        }
        poss = [{
            'grid_zone': 10, 'distance_to_basket': 5.0,
            'closest_defender_dist': 3.0, 'num_defenders_within_6ft': 1,
            'best_teammate_openness': 8.0, 'num_open_teammates': 2,
            'shot_clock': 15.0, 'is_three_point_zone': 0,
            'ball_handler_zone_fg_pct': 0.60, 'best_open_teammate_dist_to_basket': 15.0,
            'best_open_teammate_zone_fg_pct': 0.40,
            'action': 0, 'shot_made': True, 'is_three': False, 'turnover': False,
            **v4,
        }]
        env = NBAShootOrPassEnv([poss] * 10, config)
        env.reset(seed=42)
        _, reward, terminated, _, _ = env.step(0)
        # Pure EPV: (0.60 * 2.0 * min(3/6, 1.0)) - 0.375 = 0.6 - 0.375 = 0.225
        assert reward == pytest.approx(0.225, abs=0.01)
        assert terminated is True

    def test_observation_space_bounds(self):
        from environment import NBAShootOrPassEnv
        config = _make_test_config()
        possessions = [_make_test_possession()] * 10
        env = NBAShootOrPassEnv(possessions, config)
        assert env.observation_space.shape == (66,)
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)


# ============================================================
# Agent tests
# ============================================================

class TestDQNAgent:
    """Tests for DQN agent with linear epsilon decay."""

    def test_linear_epsilon_decay(self):
        import torch
        from dqn_agent import DQNAgent
        config = _make_test_config()
        device = torch.device('cpu')
        agent = DQNAgent(config, device)
        assert agent.epsilon == 1.0
        assert agent.epsilon_decay_type == "linear"

        # Simulate 5000 episodes
        for _ in range(5000):
            agent.on_episode_end()

        # At episode 5000 out of 10000 decay episodes, should be ~0.505
        assert 0.4 < agent.epsilon < 0.6, f"Epsilon at ep 5000: {agent.epsilon}"

    def test_epsilon_reaches_minimum(self):
        import torch
        from dqn_agent import DQNAgent
        config = _make_test_config()
        device = torch.device('cpu')
        agent = DQNAgent(config, device)

        for _ in range(15000):
            agent.on_episode_end()

        assert agent.epsilon == pytest.approx(0.01), f"Epsilon: {agent.epsilon}"

    def test_choose_action_shape(self):
        import torch
        from dqn_agent import DQNAgent
        config = _make_test_config()
        device = torch.device('cpu')
        agent = DQNAgent(config, device)
        state = np.random.rand(66).astype(np.float32)
        action = agent.choose_action(state, training=False)
        assert 0 <= action <= 4

    def test_store_and_update(self):
        import torch
        from dqn_agent import DQNAgent
        config = _make_test_config()
        device = torch.device('cpu')
        agent = DQNAgent(config, device)

        # Store transitions (player IDs in slots 43-47 must be valid ints)
        for _ in range(600):
            s = np.random.rand(66).astype(np.float32)
            s[61:66] = np.random.randint(0, 300, size=5).astype(np.float32)
            a = np.random.randint(0, 5)
            r = np.random.choice([-1.0, 0.0, 2.0, 3.0])
            ns = np.random.rand(66).astype(np.float32)
            ns[61:66] = np.random.randint(0, 300, size=5).astype(np.float32)
            d = np.random.random() < 0.3
            agent.store_transition(s, a, r, ns, d)

        # Should be able to update now (min_replay_size=500)
        loss = agent.update()
        assert loss is not None
        assert loss >= 0.0


class TestDuelingDQNAgent:
    """Tests for Dueling DQN agent."""

    def test_linear_epsilon_decay(self):
        import torch
        from dueling_dqn_agent import DuelingDQNAgent
        config = _make_test_config()
        device = torch.device('cpu')
        agent = DuelingDQNAgent(config, device)
        assert agent.epsilon_decay_type == "linear"

        for _ in range(10000):
            agent.on_episode_end()

        assert agent.epsilon == pytest.approx(0.01)

    def test_choose_action(self):
        import torch
        from dueling_dqn_agent import DuelingDQNAgent
        config = _make_test_config()
        device = torch.device('cpu')
        agent = DuelingDQNAgent(config, device)
        state = np.random.rand(66).astype(np.float32)
        action = agent.choose_action(state, training=False)
        assert 0 <= action <= 4


# ============================================================
# Utils tests
# ============================================================

class TestUtils:
    """Tests for utility functions."""

    def test_rolling_average(self):
        from utils import rolling_average
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = rolling_average(values, 3)
        assert len(result) == 5
        assert result[0] == pytest.approx(1.0)
        assert result[2] == pytest.approx(2.0)
        assert result[4] == pytest.approx(4.0)

    def test_save_load_training_history(self, tmp_path):
        from utils import save_training_history, load_training_history
        path = str(tmp_path / "test_history.json")
        rewards = [1.0, 2.0, 3.0]
        losses = [0.5, 0.3, 0.1]
        lengths = [5, 3, 4]
        epsilons = [1.0, 0.9, 0.8]
        save_training_history(
            rewards, losses, path,
            episode_lengths=lengths, episode_epsilons=epsilons
        )
        loaded = load_training_history(path)
        assert loaded["episode_rewards"] == rewards
        assert loaded["episode_losses"] == losses
        assert loaded["episode_lengths"] == lengths
        assert loaded["episode_epsilons"] == epsilons


# ============================================================
# Integration test
# ============================================================

class TestIntegration:
    """End-to-end test: create env, run agent for a few episodes."""

    def test_training_loop_smoke(self):
        import torch
        from environment import NBAShootOrPassEnv
        from dqn_agent import DQNAgent

        config = _make_test_config()
        config["dqn"]["min_replay_size"] = 10  # lower for test
        config["dqn"]["batch_size"] = 8  # small for test
        device = torch.device('cpu')

        possessions = [_make_test_possession()] * 100
        env = NBAShootOrPassEnv(possessions, config)
        agent = DQNAgent(config, device)

        rewards = []
        for ep in range(50):
            state, _ = env.reset()
            ep_reward = 0.0
            for step in range(50):
                action = agent.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.store_transition(
                    state, action, reward, next_state,
                    terminated or truncated
                )
                agent.update()
                ep_reward += reward
                state = next_state
                if terminated or truncated:
                    break
            agent.on_episode_end()
            rewards.append(ep_reward)

        # Should have completed 50 episodes without error
        assert len(rewards) == 50
        # Agent should have some training losses recorded
        assert len(agent.training_losses) > 0
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
