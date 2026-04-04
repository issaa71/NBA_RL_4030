"""
02_segment_possessions.py
=========================
Segments games into possessions and extracts 78 features (73 continuous + 5
player IDs) at each decision point from SportVU 2015-16 tracking data.

Processes .7z game files from local clone of linouk23/NBA-Player-Movements.
Within each possession, detects ball-handler changes (passes = decision
points) and extracts the full state vector including all player positions,
velocities, pass lane features, and player IDs.

Run after 00_pull_player_stats.py and 01_explore_data.py.
Estimated time: ~20-30 minutes for medium config (~120 games).

AISE 4030 - Group 11
"""

import numpy as np
import pickle
from datasets import load_dataset
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# ============================================================
# Constants
# ============================================================

# Basket positions (feet)
LEFT_BASKET = np.array([5.25, 25.0])
RIGHT_BASKET = np.array([88.75, 25.0])

# Court dimensions
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
HALF_COURT_LENGTH = 47.0

# Grid discretization (10 cols x 5 rows = 50 zones)
GRID_COLS = 10
GRID_ROWS = 5

# Three-point line distance (22 feet in corners, ~23.75 elsewhere)
THREE_POINT_CORNER_DIST = 22.0

# Ball possession detection
BALL_HANDLER_RADIUS = 5.0  # feet - player must be within this of ball
OPEN_DEFENDER_THRESHOLD = 6.0  # feet - NBA definition of "open"

# Minimum possession length to keep
MIN_POSSESSION_STEPS = 1

# Event message types
MADE_SHOT = 1
MISSED_SHOT = 2
TURNOVER = 5


# ============================================================
# Helper Functions
# ============================================================

def get_ball_handler(
    ball_x: float, ball_y: float,
    player_coords: list,
    offensive_team_id: int
) -> Optional[int]:
    """
    Identifies the ball-handler from a single moment's tracking data.

    Finds the closest offensive player to the ball within BALL_HANDLER_RADIUS.

    Args:
        ball_x (float): Ball x-coordinate.
        ball_y (float): Ball y-coordinate.
        player_coords (list): List of player coordinate dicts with keys
            'teamid', 'playerid', 'x', 'y', 'z'.
        offensive_team_id (int): Team ID of the offensive team.

    Returns:
        int or None: playerid of the ball-handler, or None if no one is
            close enough.
    """
    ball_pos = np.array([ball_x, ball_y])
    closest_dist = float('inf')
    closest_player_id = None

    for p in player_coords:
        if p['teamid'] != offensive_team_id:
            continue
        player_pos = np.array([p['x'], p['y']])
        dist = np.linalg.norm(player_pos - ball_pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_player_id = p['playerid']

    if closest_dist <= BALL_HANDLER_RADIUS:
        return closest_player_id
    return None


def normalize_to_half_court(
    x: float, y: float, attacking_right: bool
) -> Tuple[float, float]:
    """
    Normalizes full-court (x, y) coordinates to half-court coordinates
    where the basket is always at (5.25, 25).

    Args:
        x (float): Full-court x coordinate (0-94 feet).
        y (float): Full-court y coordinate (0-50 feet).
        attacking_right (bool): True if the offensive team attacks the right
            basket.

    Returns:
        tuple: (hc_x, hc_y) half-court coordinates where x is 0-47, y is 0-50.
    """
    if attacking_right:
        hc_x = COURT_LENGTH - x
        hc_y = COURT_WIDTH - y
    else:
        hc_x = x
        hc_y = y

    hc_x = np.clip(hc_x, 0, HALF_COURT_LENGTH)
    hc_y = np.clip(hc_y, 0, COURT_WIDTH)
    return hc_x, hc_y


def xy_to_grid_zone(hc_x: float, hc_y: float) -> int:
    """
    Maps half-court (x, y) coordinates to a discrete grid zone index.

    Args:
        hc_x (float): Half-court x coordinate (0-47 feet).
        hc_y (float): Half-court y coordinate (0-50 feet).

    Returns:
        int: Zone index in [0, 49].
    """
    col = int(np.clip(hc_x / HALF_COURT_LENGTH * GRID_COLS, 0, GRID_COLS - 1))
    row = int(np.clip(hc_y / COURT_WIDTH * GRID_ROWS, 0, GRID_ROWS - 1))
    return row * GRID_COLS + col


def distance_to_basket(hc_x: float, hc_y: float) -> float:
    """
    Computes Euclidean distance from a half-court position to the basket.

    Args:
        hc_x (float): Half-court x coordinate.
        hc_y (float): Half-court y coordinate.

    Returns:
        float: Distance in feet.
    """
    return np.linalg.norm(np.array([hc_x, hc_y]) - LEFT_BASKET)


def is_three_point_zone(dist_to_basket: float) -> bool:
    """
    Determines if a position is beyond the three-point line.

    Uses 22 feet as a conservative threshold (corner three distance).

    Args:
        dist_to_basket (float): Distance from position to basket in feet.

    Returns:
        bool: True if beyond the three-point line.
    """
    return dist_to_basket >= THREE_POINT_CORNER_DIST


def dist_to_zone_index(distance_ft: float) -> int:
    """Maps shot distance in feet to zone index 0-6 for FG% lookup."""
    if distance_ft < 5:
        return 0
    elif distance_ft < 10:
        return 1
    elif distance_ft < 15:
        return 2
    elif distance_ft < 20:
        return 3
    elif distance_ft < 25:
        return 4
    elif distance_ft < 30:
        return 5
    else:
        return 6


def compute_defender_features(
    ball_handler_pos: np.ndarray,
    teammate_positions: List[np.ndarray],
    defender_positions: List[np.ndarray]
) -> Dict[str, float]:
    """
    Computes all defender-aware features for a single decision point.

    Args:
        ball_handler_pos (np.ndarray): (x, y) of ball-handler on half-court.
        teammate_positions (list of np.ndarray): (x, y) of each offensive
            teammate.
        defender_positions (list of np.ndarray): (x, y) of each defender.

    Returns:
        dict: Keys are feature names, values are floats.
            - closest_defender_dist
            - num_defenders_within_6ft
            - best_teammate_openness
            - num_open_teammates
    """
    if defender_positions:
        defender_dists = [
            np.linalg.norm(ball_handler_pos - d) for d in defender_positions
        ]
        closest_defender_dist = min(defender_dists)
        num_defenders_within_6ft = sum(
            1 for d in defender_dists if d <= OPEN_DEFENDER_THRESHOLD
        )
    else:
        closest_defender_dist = 30.0
        num_defenders_within_6ft = 0

    teammate_openness = []
    for t_pos in teammate_positions:
        if defender_positions:
            min_def_dist = min(
                np.linalg.norm(t_pos - d) for d in defender_positions
            )
        else:
            min_def_dist = 30.0
        teammate_openness.append(min_def_dist)

    best_teammate_openness = max(teammate_openness) if teammate_openness else 0.0
    num_open_teammates = sum(
        1 for o in teammate_openness if o > OPEN_DEFENDER_THRESHOLD
    )

    return {
        'closest_defender_dist': closest_defender_dist,
        'num_defenders_within_6ft': num_defenders_within_6ft,
        'best_teammate_openness': best_teammate_openness,
        'num_open_teammates': num_open_teammates,
    }


def determine_offensive_direction(
    game_events: list, home_team_id: int, visitor_team_id: int
) -> dict:
    """
    Determines which direction each team attacks in each quarter.

    NBA teams switch sides each quarter. We infer direction from made shot
    locations in the tracking data.

    Args:
        game_events (list): List of event dicts from one game.
        home_team_id (int): Home team's ID.
        visitor_team_id (int): Visitor team's ID.

    Returns:
        dict: Mapping of (team_id, quarter) -> bool (True = attacking right).
    """
    directions = {}

    for ev in game_events:
        if ev['event_info']['type'] != MADE_SHOT:
            continue

        moments = ev['moments']
        if not moments:
            continue

        shooter_team_id = ev['primary_info']['team_id']
        if shooter_team_id is None or np.isnan(shooter_team_id):
            continue
        shooter_team_id = int(shooter_team_id)

        last_moment = moments[-1]
        quarter = last_moment['quarter']
        ball_x = last_moment['ball_coordinates']['x']

        attacking_right = ball_x > HALF_COURT_LENGTH
        directions[(shooter_team_id, quarter)] = attacking_right

    return directions


# ============================================================
# Main Possession Extraction
# ============================================================

def extract_possession_from_event(
    event: dict,
    offensive_team_id: int,
    attacking_right: bool,
    player_zone_fg: dict,
    league_avg_fg: dict,
) -> Optional[List[Dict]]:
    """
    Extracts a single possession sequence from one event's tracking data.

    Scans through the moments, detects ball-handler changes (passes),
    and builds a list of decision-point feature dicts.

    Args:
        event (dict): One event row from the HuggingFace dataset.
        offensive_team_id (int): Team ID of the team on offense.
        attacking_right (bool): Whether offense attacks the right basket.
        player_zone_fg (dict): {player_id: {zone_idx: fg_pct}} from nba_api.
        league_avg_fg (dict): {zone_idx: fg_pct} league averages as fallback.

    Returns:
        list of dict or None: Possession sequence with 11 state features
            plus metadata (action, shot_made, is_three, turnover).
    """
    moments = event['moments']
    if not moments or len(moments) < 2:
        return None

    event_type = event['event_info']['type']

    decision_points = []
    prev_ball_handler = None
    prev_player_positions = {}  # {player_id: (hc_x, hc_y)} from previous moment

    # Sample moments at ~2Hz (every 12-13 frames from 25Hz)
    sample_rate = 12
    time_delta = sample_rate / 25.0  # ~0.48 seconds between samples
    sampled_moments = moments[::sample_rate]

    for moment in sampled_moments:
        game_clock = moment['game_clock']
        shot_clock = moment['shot_clock']
        if shot_clock is None or np.isnan(shot_clock):
            shot_clock = 24.0

        ball_coords = moment['ball_coordinates']
        ball_x = ball_coords['x']
        ball_y = ball_coords['y']

        player_coords = moment['player_coordinates']
        if len(player_coords) < 10:
            continue

        # Identify ball-handler
        ball_handler_id = get_ball_handler(
            ball_x, ball_y, player_coords, offensive_team_id
        )
        if ball_handler_id is None:
            continue

        # Track ALL player positions for velocity computation
        # (update every sampled moment, not just decision points)
        current_positions = {}
        for p in player_coords:
            px, py = p['x'], p['y']
            hc = normalize_to_half_court(px, py, attacking_right)
            current_positions[p['playerid']] = np.array(hc)

        # Detect ball-handler change (= pass occurred) or first detection
        if ball_handler_id == prev_ball_handler:
            prev_ball_handler = ball_handler_id
            prev_player_positions = current_positions
            continue

        prev_ball_handler = ball_handler_id

        # --- Extract positions and IDs ---
        ball_handler_raw = None
        teammates_raw = []  # (px, py, player_id)
        defenders_raw = []

        for p in player_coords:
            px, py = p['x'], p['y']
            if p['playerid'] == ball_handler_id:
                ball_handler_raw = (px, py)
            elif p['teamid'] == offensive_team_id:
                teammates_raw.append((px, py, p['playerid']))
            else:
                defenders_raw.append((px, py))

        if ball_handler_raw is None:
            continue

        # Normalize to half-court
        bh_hc = np.array(normalize_to_half_court(
            ball_handler_raw[0], ball_handler_raw[1], attacking_right
        ))
        teammates_hc = [
            np.array(normalize_to_half_court(t[0], t[1], attacking_right))
            for t in teammates_raw
        ]
        teammate_ids = [t[2] for t in teammates_raw]
        defenders_hc = [
            np.array(normalize_to_half_court(d[0], d[1], attacking_right))
            for d in defenders_raw
        ]

        # --- Compute features ---
        grid_zone = xy_to_grid_zone(bh_hc[0], bh_hc[1])
        dist_basket = distance_to_basket(bh_hc[0], bh_hc[1])
        is_three = is_three_point_zone(dist_basket)
        def_features = compute_defender_features(
            bh_hc, teammates_hc, defenders_hc
        )

        # Ball-handler zone FG% from nba_api data
        bh_zone = dist_to_zone_index(dist_basket)
        bh_fg = player_zone_fg.get(ball_handler_id, {})
        ball_handler_zone_fg_pct = bh_fg.get(
            bh_zone, league_avg_fg.get(bh_zone, 0.40)
        )

        # --- Defender positions sorted by distance to ball-handler ---
        if defenders_hc:
            def_dists = [(np.linalg.norm(bh_hc - d), d) for d in defenders_hc]
            def_dists.sort(key=lambda x: x[0])
            help_defender_dist = def_dists[1][0] if len(def_dists) > 1 else 50.0
            # Pad to 5 defenders with far-away default
            def_pad = np.array([47.0, 25.0])
            sorted_def_pos = [d[1] for d in def_dists]
            while len(sorted_def_pos) < 5:
                sorted_def_pos.append(def_pad)
            sorted_def_pos = sorted_def_pos[:5]
        else:
            help_defender_dist = 50.0
            sorted_def_pos = [np.array([47.0, 25.0])] * 5

        # --- All 4 teammates sorted by distance to ball-handler ---
        tm_data = []  # (dist_to_bh, pos, openness, zone_fg, dist_to_basket, player_id)
        for ti, t_pos in enumerate(teammates_hc):
            dist_to_bh = np.linalg.norm(bh_hc - t_pos)
            if defenders_hc:
                openness = min(np.linalg.norm(t_pos - d) for d in defenders_hc)
            else:
                openness = 30.0
            tm_dist = distance_to_basket(t_pos[0], t_pos[1])
            tm_zone = dist_to_zone_index(tm_dist)
            tm_fg_dict = player_zone_fg.get(teammate_ids[ti], {})
            tm_fg = tm_fg_dict.get(tm_zone, league_avg_fg.get(tm_zone, 0.40))
            pid = teammate_ids[ti] if ti < len(teammate_ids) else 0
            tm_data.append((dist_to_bh, t_pos, openness, tm_fg, tm_dist, pid))

        tm_data.sort(key=lambda x: x[0])  # Sort by distance to ball-handler
        # Pad to 4 teammates
        while len(tm_data) < 4:
            tm_data.append((25.0, np.array([25.0, 25.0]), 0.0, 0.40, 25.0, 0))
        tm_data = tm_data[:4]

        # Best teammate (for backward compat with v3 features)
        best_tm = max(tm_data, key=lambda x: x[2])  # Most open

        # --- All-player velocities from previous moment ---
        def _get_velocity(player_id, current_pos):
            if player_id in prev_player_positions:
                prev = prev_player_positions[player_id]
                return ((current_pos[0] - prev[0]) / time_delta,
                        (current_pos[1] - prev[1]) / time_delta)
            return (0.0, 0.0)

        bh_vx, bh_vy = _get_velocity(ball_handler_id, bh_hc)

        # Defender velocities (sorted same order as positions)
        def_velocities = []
        for dd in def_dists if defenders_hc else []:
            # dd is (dist, position) — need player_id to look up velocity
            # Since we don't have defender IDs easily, approximate from position
            def_velocities.append((0.0, 0.0))
        # Use position-based velocity for defenders (match by nearest prev position)
        if prev_player_positions and defenders_hc:
            def_velocities = []
            for d_pos in sorted_def_pos[:5]:
                best_vel = (0.0, 0.0)
                best_dist = 999.0
                for pid, prev_pos in prev_player_positions.items():
                    d = np.linalg.norm(d_pos - prev_pos)
                    if d < best_dist and d < 10.0:  # within 10ft = same player
                        best_dist = d
                        best_vel = ((d_pos[0] - prev_pos[0]) / time_delta,
                                    (d_pos[1] - prev_pos[1]) / time_delta)
                def_velocities.append(best_vel)
        while len(def_velocities) < 5:
            def_velocities.append((0.0, 0.0))
        def_velocities = def_velocities[:5]

        # Teammate velocities (we have their IDs)
        tm_velocities = []
        for t in tm_data:
            tid = t[5]  # player_id
            if tid and tid in prev_player_positions:
                prev = prev_player_positions[tid]
                tm_velocities.append(((t[1][0] - prev[0]) / time_delta,
                                      (t[1][1] - prev[1]) / time_delta))
            else:
                tm_velocities.append((0.0, 0.0))
        while len(tm_velocities) < 4:
            tm_velocities.append((0.0, 0.0))
        tm_velocities = tm_velocities[:4]

        prev_player_positions = current_positions

        # --- v7: Pass lane features per teammate ---
        pass_lane_features = []
        for t in tm_data:
            t_pos = t[1]  # teammate half-court position
            # Compute min defender distance to BH→teammate line segment
            lane_dists = []
            for d_pos in sorted_def_pos[:5]:
                # Point-to-segment distance
                AP = d_pos - bh_hc
                AB = t_pos - bh_hc
                ab_sq = np.dot(AB, AB)
                if ab_sq < 0.01:
                    lane_dists.append(float(np.linalg.norm(AP)))
                else:
                    proj_t = np.clip(np.dot(AP, AB) / ab_sq, 0.0, 1.0)
                    closest = bh_hc + proj_t * AB
                    lane_dists.append(float(np.linalg.norm(d_pos - closest)))
            min_def_lane = min(lane_dists) if lane_dists else 47.0
            corridor_defs = sum(1 for d in lane_dists if d < 6.0)
            pass_dist = float(np.linalg.norm(t_pos - bh_hc))
            pass_lane_features.append((
                round(min_def_lane, 2),
                corridor_defs,
                round(pass_dist, 2),
            ))
        while len(pass_lane_features) < 4:
            pass_lane_features.append((47.0, 0, 25.0))

        decision_point = {
            # --- Original 11 features (backward compat) ---
            'grid_zone': grid_zone,
            'distance_to_basket': round(dist_basket, 2),
            'closest_defender_dist': round(
                def_features['closest_defender_dist'], 2
            ),
            'num_defenders_within_6ft': def_features['num_defenders_within_6ft'],
            'best_teammate_openness': round(
                def_features['best_teammate_openness'], 2
            ),
            'num_open_teammates': def_features['num_open_teammates'],
            'shot_clock': round(shot_clock, 1),
            'is_three_point_zone': 1 if is_three else 0,
            'ball_handler_zone_fg_pct': round(ball_handler_zone_fg_pct, 3),
            'best_open_teammate_dist_to_basket': round(best_tm[4], 2),
            'best_open_teammate_zone_fg_pct': round(best_tm[3], 3),

            # --- v4: Help defender ---
            'help_defender_dist': round(help_defender_dist, 2),

            # --- v4: Per-teammate features (sorted by dist to BH) ---
            'teammate_openness': [round(t[2], 2) for t in tm_data],
            'teammate_zone_fg': [round(t[3], 3) for t in tm_data],
            'teammate_dist_to_basket': [round(t[4], 2) for t in tm_data],

            # --- v4: All player positions (half-court, sorted) ---
            'defender_positions': [(round(d[0], 2), round(d[1], 2))
                                   for d in sorted_def_pos],
            'teammate_positions': [(round(t[1][0], 2), round(t[1][1], 2))
                                   for t in tm_data],

            # --- v4/v5: All-player velocities ---
            'ball_handler_vx': round(bh_vx, 2),
            'ball_handler_vy': round(bh_vy, 2),
            'defender_velocities': [(round(v[0], 2), round(v[1], 2))
                                    for v in def_velocities],
            'teammate_velocities': [(round(v[0], 2), round(v[1], 2))
                                    for v in tm_velocities],

            # --- v7: Pass lane risk features (per teammate) ---
            'pass_lane_features': pass_lane_features,  # [(min_def, corridor, dist) × 4]

            # --- v4: Player IDs ---
            'ball_handler_player_id': ball_handler_id,
            'teammate_player_ids': [t[5] for t in tm_data],

            # --- Metadata ---
            'action': 1,
            'shot_made': False,
            'is_three': is_three,
            'turnover': False,
        }
        decision_points.append(decision_point)

    if not decision_points:
        return None

    # --- Set terminal action based on event type ---
    if event_type == MADE_SHOT:
        decision_points[-1]['action'] = 0  # shoot
        decision_points[-1]['shot_made'] = True
    elif event_type == MISSED_SHOT:
        decision_points[-1]['action'] = 0  # shoot
        decision_points[-1]['shot_made'] = False
    elif event_type == TURNOVER:
        decision_points[-1]['turnover'] = True

    if len(decision_points) < MIN_POSSESSION_STEPS:
        return None

    return decision_points


def process_game(
    game_events: list, player_zone_fg: dict, league_avg_fg: dict
) -> List[List[Dict]]:
    """
    Processes all events from one game into a list of possession sequences.

    Args:
        game_events (list): All event rows from one game (same gameid).
        player_zone_fg (dict): {player_id: {zone_idx: fg_pct}}.
        league_avg_fg (dict): {zone_idx: fg_pct} league averages.

    Returns:
        list of list of dict: Each inner list is one possession sequence.
    """
    if not game_events:
        return []

    home_id = game_events[0]['home']['teamid']
    visitor_id = game_events[0]['visitor']['teamid']

    directions = determine_offensive_direction(
        game_events, home_id, visitor_id
    )

    possessions = []

    for ev in game_events:
        event_type = ev['event_info']['type']

        if event_type not in [MADE_SHOT, MISSED_SHOT, TURNOVER]:
            continue

        primary_team_id = ev['primary_info']['team_id']
        if primary_team_id is None or (
            isinstance(primary_team_id, float) and np.isnan(primary_team_id)
        ):
            continue
        offensive_team_id = int(primary_team_id)

        moments = ev['moments']
        if not moments:
            continue

        quarter = moments[0]['quarter']
        attacking_right = directions.get(
            (offensive_team_id, quarter), True
        )

        possession = extract_possession_from_event(
            ev, offensive_team_id, attacking_right,
            player_zone_fg, league_avg_fg,
        )
        if possession:
            possessions.append(possession)

    return possessions


# ============================================================
# Entry Point
# ============================================================

def load_game_from_7z(filepath):
    """Extract and load a single .7z game file. Returns parsed JSON dict."""
    import py7zr
    import tempfile
    import shutil
    tmpdir = tempfile.mkdtemp()
    try:
        with py7zr.SevenZipFile(filepath, mode='r') as z:
            z.extractall(path=tmpdir)
        extracted = os.listdir(tmpdir)
        if not extracted:
            return None
        with open(os.path.join(tmpdir, extracted[0]), encoding='utf-8') as fp:
            return json.load(fp)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def convert_raw_event_to_hf_format(event, game_id, game_date, pbp_df):
    """Convert a raw JSON event to the HuggingFace-compatible dict format
    that process_game() expects. Returns None if event can't be matched."""
    event_id = event.get("eventId")
    if event_id is None:
        return None

    # Match with play-by-play data
    event_row = pbp_df.loc[
        (pbp_df.GAME_ID == int(game_id)) & (pbp_df.EVENTNUM == int(event_id))
    ]
    if len(event_row) != 1:
        return None

    event_type = int(event_row["EVENTMSGTYPE"].item())
    primary_player_id = event_row["PLAYER1_ID"].item()
    primary_team_id = event_row["PLAYER1_TEAM_ID"].item()

    moments = []
    for moment in event.get("moments", []):
        if len(moment) < 6 or len(moment[5]) < 1:
            continue
        player_coords = []
        for i in moment[5][1:]:
            if len(i) >= 5:
                player_coords.append({
                    "teamid": i[0], "playerid": i[1],
                    "x": i[2], "y": i[3], "z": i[4]
                })
        moments.append({
            "quarter": moment[0],
            "game_clock": moment[2],
            "shot_clock": moment[3],
            "ball_coordinates": {
                "x": moment[5][0][2], "y": moment[5][0][3],
                "z": moment[5][0][4] if len(moment[5][0]) > 4 else 0
            },
            "player_coordinates": player_coords,
        })

    return {
        "gameid": game_id,
        "gamedate": game_date,
        "event_info": {"id": event_id, "type": event_type},
        "primary_info": {
            "player_id": primary_player_id,
            "team_id": primary_team_id,
        },
        "home": event["home"],
        "visitor": event["visitor"],
        "moments": moments,
    }


if __name__ == "__main__":
    import os
    import glob
    import json
    import pandas as pd

    print("=" * 60)
    print("Step 2: Segmenting possessions from SportVU tracking data")
    print("         (Local .7z files + PBP CSV merge)")
    print("=" * 60)

    # Load player zone FG% from nba_api (Step 0)
    zone_fg_path = "player_zone_fg.pkl"
    if os.path.exists(zone_fg_path):
        with open(zone_fg_path, "rb") as f:
            player_zone_fg = pickle.load(f)
        league_avg_fg = player_zone_fg.pop("__league_avg__", {})
        print(f"Loaded zone FG% for {len(player_zone_fg)} players")
    else:
        print("WARNING: player_zone_fg.pkl not found, using empty fallback")
        player_zone_fg = {}
        league_avg_fg = {0: 0.60, 1: 0.40, 2: 0.40, 3: 0.38,
                         4: 0.35, 5: 0.35, 6: 0.25}

    # --- Download PBP CSV (same source as HuggingFace loader) ---
    PBP_URL = "https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv"
    pbp_cache = "pbp_2015_16.csv"
    if not os.path.exists(pbp_cache):
        print(f"\nDownloading play-by-play CSV...")
        import urllib.request
        urllib.request.urlretrieve(PBP_URL, pbp_cache)
    pbp_df = pd.read_csv(pbp_cache)
    print(f"Loaded PBP: {len(pbp_df)} play-by-play events")

    # --- Find local .7z game files ---
    data_dir = os.path.expanduser(
        "~/Downloads/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs"
    )
    game_files = sorted(glob.glob(os.path.join(data_dir, "*.7z")))
    print(f"Found {len(game_files)} game files in {data_dir}")

    if not game_files:
        print("ERROR: No .7z files found. Clone the repo first:")
        print("  git clone https://github.com/linouk23/NBA-Player-Movements.git ~/Downloads/NBA-Player-Movements")
        exit(1)

    # --- Process games one at a time (memory-safe) ---
    all_possessions = []
    num_games = 0
    num_failed = 0

    for filepath in game_files:
        num_games += 1
        game_name = os.path.basename(filepath)

        try:
            game_json = load_game_from_7z(filepath)
            if game_json is None:
                num_failed += 1
                continue

            game_id = game_json.get("gameid", "unknown")
            game_date = game_json.get("gamedate", "unknown")

            # Convert raw events to HuggingFace-compatible format
            hf_events = []
            for event in game_json.get("events", []):
                hf_event = convert_raw_event_to_hf_format(
                    event, game_id, game_date, pbp_df
                )
                if hf_event is not None:
                    hf_events.append(hf_event)

            # Free raw JSON immediately
            del game_json

            if hf_events:
                possessions = process_game(
                    hf_events, player_zone_fg, league_avg_fg
                )
                for poss in possessions:
                    for dp in poss:
                        dp['game_id'] = game_id
                all_possessions.extend(possessions)

            del hf_events

        except Exception as e:
            num_failed += 1
            if num_games <= 5 or num_failed <= 3:
                print(f"  WARNING: Failed {game_name}: {e}")

        if num_games % 25 == 0 or num_games == 1:
            print(f"  Game {num_games}/{len(game_files)}: "
                  f"{len(all_possessions)} possessions "
                  f"({num_failed} failed)")

    print(f"\nDone: {num_games} games, {num_failed} failed")

    # --- Summary stats ---
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Games processed: {num_games}")
    print(f"Total possessions: {len(all_possessions)}")

    if all_possessions:
        lengths = [len(p) for p in all_possessions]
        print(f"Possession lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}")

        # Count outcomes
        shots_made = sum(1 for p in all_possessions if p[-1]['shot_made'])
        shots_missed = sum(
            1 for p in all_possessions
            if p[-1]['action'] == 0 and not p[-1]['shot_made']
        )
        turnovers = sum(1 for p in all_possessions if p[-1]['turnover'])
        print(f"Made shots: {shots_made}, Missed shots: {shots_missed}, "
              f"Turnovers: {turnovers}")

        # Show sample possession
        print(f"\n--- Sample Possession (first one) ---")
        for i, dp in enumerate(all_possessions[0]):
            action_name = "SHOOT" if dp['action'] == 0 else "PASS"
            print(
                f"  Step {i}: zone={dp['grid_zone']:2d}, "
                f"dist_basket={dp['distance_to_basket']:5.1f}ft, "
                f"def_dist={dp['closest_defender_dist']:4.1f}ft, "
                f"open_tm={dp['num_open_teammates']}, "
                f"shot_clk={dp['shot_clock']:4.1f}s, "
                f"action={action_name}"
                f"{' (MADE)' if dp['shot_made'] else ''}"
                f"{' (TURNOVER)' if dp['turnover'] else ''}"
            )

        # Verify 11 state features and zone FG% variance
        all_dps = [dp for p in all_possessions for dp in p]
        state_keys = [
            'grid_zone', 'distance_to_basket', 'closest_defender_dist',
            'num_defenders_within_6ft', 'best_teammate_openness',
            'num_open_teammates', 'shot_clock', 'is_three_point_zone',
            'ball_handler_zone_fg_pct', 'best_open_teammate_dist_to_basket',
            'best_open_teammate_zone_fg_pct',
        ]
        missing = [k for k in state_keys if k not in all_dps[0]]
        if missing:
            print(f"WARNING: Missing state keys: {missing}")
        else:
            print(f"All 11 state features present")

        fg_vals = [dp['ball_handler_zone_fg_pct'] for dp in all_dps]
        print(f"ball_handler_zone_fg_pct: min={min(fg_vals):.3f}, "
              f"max={max(fg_vals):.3f}, "
              f"unique={len(set(fg_vals))}")

    # --- Save ---
    output_path = "processed_possessions.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_possessions, f)
    print(f"\nSaved {len(all_possessions)} possessions to {output_path}")
    print(f"Next step: Run 03_build_shot_model.py")
