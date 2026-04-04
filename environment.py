"""
environment.py
==============
Custom Gymnasium environment for NBA possession-level shot selection.

Wraps processed SportVU 2015-16 tracking data as a sequential offline replay
environment with stochastic shot outcomes. Each episode is one historical
possession. On reset(), a random possession is sampled. The episode steps
through passes until a shoot action is taken or the possession ends.

Phase 3: Uses real SportVU data with 11-dimensional state and a shot
probability model for stochastic rewards.

State vector (11 features):
    [grid_zone, distance_to_basket, closest_defender_dist,
     num_defenders_within_6ft, best_teammate_openness, num_open_teammates,
     shot_clock, is_three_point_zone, ball_handler_zone_fg_pct,
     best_open_teammate_dist_to_basket, best_open_teammate_zone_fg_pct]

Action space (Discrete, 5):
    0: shoot
    1-4: pass to teammate 1-4 (sorted by distance to ball-handler)
         Reward differentiated by target teammate's expected value

AISE 4030 - Group 11
"""

import numpy as np
import pickle
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List


class NBAShootOrPassEnv(gym.Env):
    """
    A custom Gymnasium environment that models NBA offensive possessions
    as sequential MDPs for shot-selection reinforcement learning.

    Phase 3 version: uses real SportVU tracking data with 11-dimensional
    state and a trained shot probability model for stochastic outcomes.

    Attributes:
        possessions (list): Preprocessed possession sequences from SportVU data.
        config (dict): Environment configuration dictionary.
        shot_model: Trained sklearn model for P(made | features).
        current_possession (list): The possession being replayed.
        current_step (int): Index of current decision point.
        state (np.ndarray): Current 11D state vector.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, possessions: list, config: dict, shot_model=None):
        """
        Initializes the NBAShootOrPassEnv.

        Args:
            possessions (list): List of preprocessed possession sequences.
                Each possession is a list of dicts with 12 state features
                plus metadata (action, shot_made, is_three, turnover).
            config (dict): Configuration dictionary loaded from config.yaml.
            shot_model: Trained sklearn model with predict_proba() method.
                If None, falls back to historical outcomes (Phase 2 behavior).
        """
        super().__init__()

        self.possessions = possessions
        self.config = config
        self.shot_model = shot_model
        env_cfg = config["environment"]
        reward_cfg = config["rewards"]

        self.num_actions = env_cfg["num_actions"]  # 5
        self.max_steps = config["training"]["max_steps_per_episode"]

        self.reward_made_2pt = reward_cfg["made_2pt"]
        self.reward_made_3pt = reward_cfg["made_3pt"]
        self.reward_missed = reward_cfg["missed_shot"]
        self.reward_pass = reward_cfg["pass"]
        self.reward_turnover = reward_cfg["turnover"]
        self.pass_reward_scale = reward_cfg.get("pass_reward_scale", 0.5)
        self.pass_distance_penalty = reward_cfg.get("pass_distance_penalty", 0.0)
        self.shoot_baseline = reward_cfg.get("shoot_baseline", 0.75)
        self.pass_baseline = reward_cfg.get("pass_baseline", 0.75)
        self.gamma = config["dqn"]["gamma"]

        # --- Gymnasium Action Space ---
        self.action_space = spaces.Discrete(self.num_actions)

        # --- Gymnasium Observation Space (v7: 78D = 73 continuous + 5 player IDs) ---
        state_dim = config["network"]["state_dim"]
        low = np.array([
            # Original features (0-10)
            0.0,    # 0: grid_zone
            0.0,    # 1: distance_to_basket
            0.0,    # 2: closest_defender_dist
            0.0,    # 3: help_defender_dist
            0.0,    # 4: num_defenders_within_6ft
            0.0,    # 5: best_teammate_openness
            0.0,    # 6: num_open_teammates
            0.0,    # 7: shot_clock
            0.0,    # 8: shot_clock_urgency
            0.0,    # 9: is_three_point_zone
            0.0,    # 10: ball_handler_zone_fg_pct
            # Per-teammate features (11-22)
            0.0, 0.0, 0.0, 0.0,    # 11-14: teammate 1-4 openness
            0.0, 0.0, 0.0, 0.0,    # 15-18: teammate 1-4 zone FG%
            0.0, 0.0, 0.0, 0.0,    # 19-22: teammate 1-4 dist to basket
            # Defender positions (23-32)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Teammate positions (33-40)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Ball-handler velocity (41-42)
            -20.0, -20.0,
            # Defender velocities (43-52)
            -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0,
            # Teammate velocities (53-60)
            -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0,
            # v7: Pass lane features (61-72)
            0.0, 0.0, 0.0, 0.0,    # 61-64: min_def_dist_lane × 4
            0.0, 0.0, 0.0, 0.0,    # 65-68: corridor_defs × 4
            0.0, 0.0, 0.0, 0.0,    # 69-72: pass_distance × 4
            # Player IDs (73-77) — raw ints, split out in network
            0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        high = np.array([
            # Original features (0-10)
            49.0, 50.0, 50.0, 50.0, 5.0, 50.0, 4.0, 24.0, 1.0, 1.0, 1.0,
            # Per-teammate features (11-22)
            50.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 50.0,
            # Defender positions (23-32)
            47.0, 50.0, 47.0, 50.0, 47.0, 50.0, 47.0, 50.0, 47.0, 50.0,
            # Teammate positions (33-40)
            47.0, 50.0, 47.0, 50.0, 47.0, 50.0, 47.0, 50.0,
            # Ball-handler velocity (41-42)
            20.0, 20.0,
            # Defender velocities (43-52)
            20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
            # Teammate velocities (53-60)
            20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
            # v7: Pass lane features (61-72)
            47.0, 47.0, 47.0, 47.0,    # min_def_dist_lane
            5.0, 5.0, 5.0, 5.0,        # corridor_defs
            47.0, 47.0, 47.0, 47.0,    # pass_distance
            # Player IDs (73-77)
            500.0, 500.0, 500.0, 500.0, 500.0,
        ], dtype=np.float32)
        # Save raw bounds for state normalization
        self._obs_low = low
        self._obs_range = high - low
        # Avoid division by zero for any constant features
        self._obs_range[self._obs_range == 0] = 1.0

        # Observation space reflects normalized [0, 1] output
        self.observation_space = spaces.Box(
            low=np.zeros(state_dim, dtype=np.float32),
            high=np.ones(state_dim, dtype=np.float32),
            dtype=np.float32
        )

        # Internal state
        self.current_possession: list = []
        self.current_step: int = 0
        self.state: Optional[np.ndarray] = None
        self.np_random = np.random.default_rng()

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment by sampling a new possession from the dataset.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Unused.

        Returns:
            tuple: (state vector of shape (11,), empty info dict)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        idx = self.np_random.integers(0, len(self.possessions))
        self.current_possession = self.possessions[idx]
        self.current_step = 0
        self.state = self._build_state(self.current_possession[0])
        return self.state.copy(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one decision step within the current possession.

        If action == 0 (shoot), the episode terminates and reward is
        sampled from the shot probability model (stochastic).
        If action == 1-4 (pass), the environment transitions to the next
        decision point with reward 0.

        Args:
            action (int): 0 = shoot, 1-4 = pass_to_teammate_{1-4}.

        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
        """
        current_point = self.current_possession[self.current_step]

        # Check for turnover
        if self._is_turnover(current_point):
            epv_current = self._compute_epv(current_point)
            reward = self.reward_turnover - epv_current  # PBRS: 0 - Φ(s)
            return self.state.copy(), reward, True, False, {}

        if action == 0:  # shoot
            epv_current = self._compute_epv(current_point)
            raw_reward = self._compute_raw_shoot_reward(current_point)
            # PBRS: r + γ·Φ(s') - Φ(s). Terminal: Φ(s')=0
            reward = raw_reward - epv_current
            terminated = True
            self.current_step += 1
        else:  # pass to teammate (action 1-4)
            epv_current = self._compute_epv(current_point)
            terminated = False
            self.current_step += 1
            if self.current_step >= len(self.current_possession):
                terminated = True

            if not terminated:
                next_point = self.current_possession[self.current_step]
                epv_next = self._compute_epv(next_point)
                # PBRS: r + γ·Φ(s') - Φ(s). r=pass_step_penalty
                reward = self.reward_pass + self.gamma * epv_next - epv_current

                # Pass distance penalty
                pass_lane = current_point.get('pass_lane_features', [(47, 0, 25)] * 4)
                idx = min(action - 1, len(pass_lane) - 1)
                pass_dist = pass_lane[idx][2] if idx < len(pass_lane) else 25.0
                reward -= self.pass_distance_penalty * max(0, pass_dist - 10)

                # Shot clock urgency: penalize passing when clock is low
                shot_clock = current_point.get('shot_clock', 24.0)
                if shot_clock < 7:
                    reward -= 0.7 * (1 - shot_clock / 7.0) ** 1.5
            else:
                # Possession ended without a shot
                reward = self.reward_pass - epv_current

        truncated = self.current_step >= self.max_steps

        if not terminated and not truncated:
            next_point = self.current_possession[self.current_step]
            self.state = self._build_state(next_point)

        return self.state.copy(), reward, terminated, truncated, {}

    def _build_state(self, decision_point: dict) -> np.ndarray:
        """
        Constructs a normalized 48D state vector from a decision point dict.
        43 continuous features + 5 player IDs (raw ints, split in network).

        Args:
            decision_point (dict): Dict with all feature keys.

        Returns:
            np.ndarray: State vector of shape (48,). Continuous features
                normalized to [0, 1], player IDs as raw ints.
        """
        dp = decision_point
        shot_clock = dp['shot_clock']

        # Per-teammate features (lists of 4, sorted by dist to BH)
        tm_open = dp.get('teammate_openness', [0.0] * 4)
        tm_fg = dp.get('teammate_zone_fg', [0.40] * 4)
        tm_dist = dp.get('teammate_dist_to_basket', [25.0] * 4)

        # Defender positions (list of 5 (x,y) tuples, sorted closest-first)
        def_pos = dp.get('defender_positions', [(47.0, 25.0)] * 5)

        # Teammate positions (list of 4 (x,y) tuples)
        tm_pos = dp.get('teammate_positions', [(25.0, 25.0)] * 4)

        # Pass lane features (list of 4 tuples: (min_def, corridor, dist))
        pass_lane = dp.get('pass_lane_features', [(47.0, 0, 25.0)] * 4)
        while len(pass_lane) < 4:
            pass_lane.append((47.0, 0, 25.0))

        # Player IDs
        bh_pid = dp.get('ball_handler_player_id', 0)
        tm_pids = dp.get('teammate_player_ids', [0] * 4)

        raw = np.array([
            # Original core features (0-10)
            dp['grid_zone'],
            dp['distance_to_basket'],
            dp['closest_defender_dist'],
            dp.get('help_defender_dist', 50.0),
            dp['num_defenders_within_6ft'],
            dp['best_teammate_openness'],
            dp['num_open_teammates'],
            shot_clock,
            max(0.0, (7.0 - shot_clock) / 7.0),  # urgency
            dp['is_three_point_zone'],
            dp['ball_handler_zone_fg_pct'],
            # Per-teammate features (11-22)
            tm_open[0], tm_open[1], tm_open[2], tm_open[3],
            tm_fg[0], tm_fg[1], tm_fg[2], tm_fg[3],
            tm_dist[0], tm_dist[1], tm_dist[2], tm_dist[3],
            # Defender positions flattened (23-32)
            def_pos[0][0], def_pos[0][1],
            def_pos[1][0], def_pos[1][1],
            def_pos[2][0], def_pos[2][1],
            def_pos[3][0], def_pos[3][1],
            def_pos[4][0], def_pos[4][1],
            # Teammate positions flattened (33-40)
            tm_pos[0][0], tm_pos[0][1],
            tm_pos[1][0], tm_pos[1][1],
            tm_pos[2][0], tm_pos[2][1],
            tm_pos[3][0], tm_pos[3][1],
            # Ball-handler velocity (41-42)
            dp.get('ball_handler_vx', 0.0),
            dp.get('ball_handler_vy', 0.0),
            # Defender velocities flattened (43-52)
            *[v for dv in dp.get('defender_velocities', [(0,0)]*5) for v in dv],
            # Teammate velocities flattened (53-60)
            *[v for tv in dp.get('teammate_velocities', [(0,0)]*4) for v in tv],
            # v7: Pass lane features (61-72)
            *[f[0] for f in pass_lane],  # 61-64: min_def_dist_lane × 4
            *[f[1] for f in pass_lane],  # 65-68: corridor_defs × 4
            *[f[2] for f in pass_lane],  # 69-72: pass_distance × 4
            # Player IDs (73-77) — raw ints, NOT normalized
            float(bh_pid),
            float(tm_pids[0]), float(tm_pids[1]),
            float(tm_pids[2]), float(tm_pids[3]),
        ], dtype=np.float32)

        # Normalize continuous features (0-72) to [0, 1]
        # Player IDs (73-77) are left as-is for embedding lookup
        normalized = (raw - self._obs_low) / self._obs_range
        # Don't normalize player IDs — keep raw values
        normalized[73:78] = raw[73:78]
        return normalized

    def _compute_epv(self, decision_point: dict) -> float:
        """
        Computes EPV proxy for a state: the max contest-adjusted expected
        value across ball handler and all teammates.

        This serves as the potential function Φ(s) for PBRS.

        Args:
            decision_point (dict): The decision point dict.

        Returns:
            float: EPV proxy value (always positive).
        """
        # EPV = max of TEAMMATE EPVs only (excludes ball handler).
        # This represents the continuation value — what you'd get by passing.
        # Shoot reward = bh_epv - Φ(s) → positive when BH has the best look.
        tm_fg = decision_point.get('teammate_zone_fg', [0.40] * 4)
        tm_open = decision_point.get('teammate_openness', [0.0] * 4)
        tm_dist = decision_point.get('teammate_dist_to_basket', [25.0] * 4)

        epvs = []
        for i in range(min(4, len(tm_fg))):
            pts = 3.0 if tm_dist[i] >= 22.0 else 2.0
            contest = min(tm_open[i] / 6.0, 1.0)
            epvs.append(tm_fg[i] * pts * contest)

        return max(epvs) if epvs else 0.5

    def _compute_raw_shoot_reward(self, decision_point: dict) -> float:
        """
        Computes raw shoot reward (EPV of the shot, no baseline subtraction).
        Used with PBRS where the baseline comes from Φ(s).

        Args:
            decision_point (dict): The decision point dict.

        Returns:
            float: Raw expected points value of taking this shot.
        """
        zone_fg_pct = decision_point.get('ball_handler_zone_fg_pct', 0.40)
        is_three = decision_point.get('is_three_point_zone', 0)
        point_value = 3.0 if is_three else 2.0
        closest_def = decision_point.get('closest_defender_dist', 6.0)
        contest_factor = min(closest_def / 6.0, 1.0)
        return zone_fg_pct * point_value * contest_factor

    def _compute_reward(self, decision_point: dict) -> float:
        """Legacy reward function (unused with PBRS, kept for reference)."""
        return self._compute_raw_shoot_reward(decision_point) - self.shoot_baseline

    def _compute_pass_reward(self, decision_point: dict, teammate_idx: int) -> float:
        """
        Computes pass reward based on the target teammate's expected value.

        A pass to a wide-open teammate with high FG% gets a better reward
        than a pass to a contested teammate. This differentiates the 4 pass
        actions so the agent learns WHO to pass to, not just whether to pass.

        Args:
            decision_point (dict): The decision point dict.
            teammate_idx (int): Index 0-3 of the target teammate (sorted by
                distance to ball-handler).

        Returns:
            float: Pass reward based on target teammate's quality.
        """
        tm_fg = decision_point.get('teammate_zone_fg', [0.40] * 4)
        tm_open = decision_point.get('teammate_openness', [0.0] * 4)
        tm_dist = decision_point.get('teammate_dist_to_basket', [25.0] * 4)

        idx = min(teammate_idx, len(tm_fg) - 1)

        # Teammate's expected shot value if they were to shoot
        is_three = 1 if tm_dist[idx] >= 22.0 else 0
        point_value = 3.0 if is_three else 2.0
        contest = min(tm_open[idx] / 6.0, 1.0)
        teammate_epv = tm_fg[idx] * point_value * contest

        # Reward = teammate's EPV (centered), scaled by configurable factor
        base_reward = (teammate_epv - self.pass_baseline) * self.pass_reward_scale

        # v8: penalize long passes (higher interception risk)
        pass_lane = decision_point.get('pass_lane_features', [(47, 0, 25)] * 4)
        if idx < len(pass_lane):
            pass_dist = pass_lane[idx][2]  # pass distance in feet
        else:
            pass_dist = 25.0
        distance_penalty = self.pass_distance_penalty * max(0, pass_dist - 10)

        return base_reward - distance_penalty

    def _is_turnover(self, decision_point: dict) -> bool:
        """
        Checks if the current step is a turnover.

        Args:
            decision_point (dict): Current decision point dict.

        Returns:
            bool: True if turnover.
        """
        return decision_point.get('turnover', False)

    def render(self, mode: str = "human") -> None:
        """
        Renders current state to stdout.

        Args:
            mode (str): Render mode. Only 'human' is supported.

        Returns:
            None
        """
        if self.state is not None:
            print(
                f"  Zone: {self.state[0]:.2f} | "
                f"Dist: {self.state[1]:.2f} | "
                f"Def: {self.state[2]:.2f} | "
                f"Open TM: {self.state[5]:.2f} | "
                f"Shot clk: {self.state[6]:.2f} | "
                f"FG%: {self.state[8]:.2f} | "
                f"TM FG%: {self.state[10]:.2f}"
            )

    def close(self) -> None:
        """
        Cleans up any resources held by the environment.

        Returns:
            None
        """
        pass


def make_env(
    config: dict, split: str = "train", shot_model=None
) -> NBAShootOrPassEnv:
    """
    Factory function to create a configured NBAShootOrPassEnv.

    Args:
        config (dict): Full config dict.
        split (str): 'train' or 'test'.
        shot_model: Trained sklearn shot probability model (or None).

    Returns:
        NBAShootOrPassEnv: Ready-to-use environment.
    """
    train_possessions, test_possessions = load_and_preprocess_dataset(config)
    possessions = train_possessions if split == "train" else test_possessions
    return NBAShootOrPassEnv(possessions, config, shot_model=shot_model)


def load_and_preprocess_dataset(config: dict):
    """
    Loads processed possession data and splits by game_id to prevent
    within-game data leakage between train and test sets.

    Falls back to index-based split if game_id is not present in data
    (backward compatibility with v2 pickle).

    Args:
        config (dict): Config dict with 'environment.processed_data_path'
            pointing to the processed_possessions.pkl file.

    Returns:
        tuple: (train_possessions, test_possessions)
    """
    from collections import defaultdict

    dataset_path = config["environment"].get(
        "processed_data_path", "processed_possessions.pkl"
    )

    with open(dataset_path, 'rb') as f:
        all_possessions = pickle.load(f)

    # Build compact player ID mapping (raw NBA IDs -> 0..N)
    all_player_ids = set()
    for poss in all_possessions:
        for dp in poss:
            pid = dp.get('ball_handler_player_id', 0)
            if pid:
                all_player_ids.add(pid)
            for tid in dp.get('teammate_player_ids', []):
                if tid:
                    all_player_ids.add(tid)
    pid_map = {pid: idx + 1 for idx, pid in enumerate(sorted(all_player_ids))}
    pid_map[0] = 0  # unknown/padding -> 0

    # Remap all player IDs to compact indices
    for poss in all_possessions:
        for dp in poss:
            dp['ball_handler_player_id'] = pid_map.get(
                dp.get('ball_handler_player_id', 0), 0)
            dp['teammate_player_ids'] = [
                pid_map.get(tid, 0)
                for tid in dp.get('teammate_player_ids', [0, 0, 0, 0])
            ]

    print(f"Remapped {len(all_player_ids)} player IDs to compact indices 0-{len(pid_map)-1}")

    # Group possessions by game_id for game-level split
    games = defaultdict(list)
    for poss in all_possessions:
        gid = poss[0].get('game_id', 'unknown')
        games[gid].append(poss)

    if len(games) <= 1:
        print("WARNING: No game_id found in possessions. "
              "Re-run 02_segment_possessions.py to enable game-level split.")
        split_idx = int(
            len(all_possessions) * (1 - config["environment"]["test_split"])
        )
        return all_possessions[:split_idx], all_possessions[split_idx:]

    # Sort game_ids for reproducibility, split games 80/20
    sorted_game_ids = sorted(games.keys())
    split_idx = int(
        len(sorted_game_ids) * (1 - config["environment"]["test_split"])
    )
    train_game_ids = set(sorted_game_ids[:split_idx])

    train = []
    test = []
    for gid in sorted_game_ids:
        if gid in train_game_ids:
            train.extend(games[gid])
        else:
            test.extend(games[gid])

    return train, test
