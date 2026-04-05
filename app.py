"""
app.py — FastAPI backend for NBA Court Visualization with Agent Overlay.

Serves a web-based visualization of real SportVU tracking data with
real-time RL agent shot/pass recommendations.

Usage:
    uvicorn app:app --reload
    # Open http://localhost:8000 in browser

AISE 4030 - Group 11
"""

import os
import sys
import json
import re
import pickle
import tempfile
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware

# ---- Constants ----

COURT_LENGTH = 94.0
HALF_COURT = 47.0
COURT_WIDTH = 50.0
BASKET = np.array([5.25, 25.0])
TIME_DELTA = 12 / 25.0  # 0.48s — matches training pipeline sample rate
FRAME_BUFFER_SIZE = 13  # Need 12 frames back for velocity

DATA_DIR = os.path.expanduser(
    "~/Downloads/NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs"
)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TEAM_INFO = {
    1610612737: ("#E13A3E", "ATL", "Hawks"),
    1610612738: ("#008348", "BOS", "Celtics"),
    1610612751: ("#061922", "BKN", "Nets"),
    1610612766: ("#1D1160", "CHA", "Hornets"),
    1610612741: ("#CE1141", "CHI", "Bulls"),
    1610612739: ("#860038", "CLE", "Cavaliers"),
    1610612742: ("#007DC5", "DAL", "Mavericks"),
    1610612743: ("#4D90CD", "DEN", "Nuggets"),
    1610612765: ("#006BB6", "DET", "Pistons"),
    1610612744: ("#FDB927", "GSW", "Warriors"),
    1610612745: ("#CE1141", "HOU", "Rockets"),
    1610612754: ("#00275D", "IND", "Pacers"),
    1610612746: ("#ED174C", "LAC", "Clippers"),
    1610612747: ("#552582", "LAL", "Lakers"),
    1610612763: ("#0F586C", "MEM", "Grizzlies"),
    1610612748: ("#98002E", "MIA", "Heat"),
    1610612749: ("#00471B", "MIL", "Bucks"),
    1610612750: ("#005083", "MIN", "Timberwolves"),
    1610612740: ("#002B5C", "NOP", "Pelicans"),
    1610612752: ("#006BB6", "NYK", "Knicks"),
    1610612760: ("#007DC3", "OKC", "Thunder"),
    1610612753: ("#007DC5", "ORL", "Magic"),
    1610612755: ("#006BB6", "PHI", "76ers"),
    1610612756: ("#1D1160", "PHX", "Suns"),
    1610612757: ("#E03A3E", "POR", "Trail Blazers"),
    1610612758: ("#724C9F", "SAC", "Kings"),
    1610612759: ("#BAC3C9", "SAS", "Spurs"),
    1610612761: ("#CE1141", "TOR", "Raptors"),
    1610612762: ("#00471B", "UTA", "Jazz"),
    1610612764: ("#002B5C", "WAS", "Wizards"),
}

# Normalization bounds — MUST match environment.py exactly
OBS_LOW = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0-10: game context
    + [0] * 12                            # 11-22: per-teammate features
    + [0] * 10                            # 23-32: defender positions
    + [0] * 8                             # 33-40: teammate positions
    + [-20, -20]                          # 41-42: BH velocity
    + [-20] * 10                          # 43-52: defender velocities
    + [-20] * 8                           # 53-60: teammate velocities
    + [0] * 4 + [0] * 4 + [0] * 4        # 61-72: pass lane features
    + [0] * 5,                            # 73-77: player IDs
    dtype=np.float32,
)
OBS_HIGH = np.array(
    [49, 50, 50, 50, 5, 50, 4, 24, 1, 1, 1]
    + [50] * 4 + [1] * 4 + [50] * 4
    + [47, 50] * 5
    + [47, 50] * 4
    + [20, 20]
    + [20] * 10
    + [20] * 8
    + [47] * 4 + [5] * 4 + [47] * 4      # pass lane features
    + [500] * 5,
    dtype=np.float32,
)
OBS_RANGE = OBS_HIGH - OBS_LOW
OBS_RANGE[OBS_RANGE == 0] = 1.0

ACTION_NAMES = ["SHOOT", "PASS 1", "PASS 2", "PASS 3", "PASS 4"]


# ---- Data Loading ----

def load_player_fg():
    path = os.path.join(PROJECT_DIR, "player_zone_fg.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        league_avg = data.pop("__league_avg__", {})
        return data, league_avg
    return {}, {0: 0.60, 1: 0.40, 2: 0.40, 3: 0.38, 4: 0.35, 5: 0.35, 6: 0.25}


PLAYER_FG, LEAGUE_AVG = load_player_fg()


def get_fg(player_id, dist_ft):
    zone = (
        0 if dist_ft < 5 else
        1 if dist_ft < 10 else
        2 if dist_ft < 15 else
        3 if dist_ft < 20 else
        4 if dist_ft < 25 else
        5 if dist_ft < 30 else 6
    )
    player_fg = PLAYER_FG.get(player_id, {}).get(zone)
    league = LEAGUE_AVG.get(zone, 0.40)
    return player_fg if player_fg is not None else league


def load_pid_map():
    """Load or build compact player ID mapping matching training."""
    cache_path = os.path.join(PROJECT_DIR, "pid_map.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    pkl_path = os.path.join(PROJECT_DIR, "processed_possessions.pkl")
    if not os.path.exists(pkl_path):
        print("WARNING: No pid_map.json or processed_possessions.pkl found.")
        return {}

    print("Building pid_map from processed_possessions.pkl (first run only)...")
    with open(pkl_path, "rb") as f:
        possessions = pickle.load(f)

    all_ids = set()
    for poss in possessions:
        for dp in poss:
            pid = dp.get("ball_handler_player_id", 0)
            if pid:
                all_ids.add(pid)
            for tid in dp.get("teammate_player_ids", []):
                if tid:
                    all_ids.add(tid)

    pid_map = {pid: idx + 1 for idx, pid in enumerate(sorted(all_ids))}
    pid_map[0] = 0
    del possessions

    with open(cache_path, "w") as f:
        json.dump(pid_map, f)
    print(f"Saved pid_map.json ({len(pid_map)} players)")
    return pid_map


def load_game_7z(filepath):
    import py7zr
    tmpdir = tempfile.mkdtemp()
    try:
        with py7zr.SevenZipFile(filepath, mode="r") as z:
            z.extractall(path=tmpdir)
        extracted = os.listdir(tmpdir)
        with open(os.path.join(tmpdir, extracted[0]), encoding="utf-8") as fp:
            return json.load(fp)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---- Agent Loading ----

def load_agent(agent_type="dueling"):
    sys.path.insert(0, PROJECT_DIR)
    from utils import load_config, get_device

    config = load_config(os.path.join(PROJECT_DIR, "config.yaml"))
    device = get_device()

    # Find best weights — try PBRS models first, then fallback
    weights = None
    if agent_type == "dqn":
        candidates = [
            "results_v9/pbrs_dqn_nodist/dqn_weights.pth",
            "results_v9/dqn_ps10/dqn_weights.pth",
        ]
    else:
        candidates = [
            "results_v9/pbrs_lr1e4/dueling_dqn_weights.pth",
            "results_v9/pbrs_smallnet/dueling_dqn_weights.pth",
            "results_v9/dueling_ps10/dueling_dqn_weights.pth",
        ]

    for w in candidates:
        full = os.path.join(PROJECT_DIR, w)
        if os.path.exists(full):
            weights = full
            break

    if weights is None:
        raise FileNotFoundError("No agent weights found")

    if agent_type == "dueling":
        from dueling_dqn_agent import DuelingDQNAgent
        agent = DuelingDQNAgent(config, device)
    else:
        from dqn_agent import DQNAgent
        agent = DQNAgent(config, device)

    agent.load(weights)
    print(f"Loaded {agent_type} agent from {weights}")
    return agent, device


# ---- Direction Detection ----

def determine_direction_from_settled_possessions(all_moments, home_id, visitor_id):
    """Scan Q1 moments for settled possessions where teams are clearly separated.

    In settled half-court offense, the offensive team clusters on one side (avg x > 60)
    and the defensive team on the other (avg x < 34). We count votes to determine
    which team attacks which side, then lock it for the whole half.
    """
    home_right_votes = 0
    home_left_votes = 0

    for m in all_moments:
        if len(m) < 6 or len(m[5]) < 11:
            continue
        if m[0] != 1:  # Q1 only
            continue

        players = m[5][1:]
        home_xs = [p[2] for p in players if p[0] == home_id]
        away_xs = [p[2] for p in players if p[0] == visitor_id]

        if not home_xs or not away_xs:
            continue

        home_avg = sum(home_xs) / len(home_xs)
        away_avg = sum(away_xs) / len(away_xs)

        # Only count frames where teams are clearly separated (settled offense)
        if home_avg > 60 and away_avg < 34:
            home_right_votes += 1
        elif home_avg < 34 and away_avg > 60:
            home_left_votes += 1

    # When teams are separated, the team clustered on one side is DEFENDING that
    # basket (packed in paint), not attacking. So the logic is inverted:
    # home on right side → home is defending right → home attacks LEFT
    if home_right_votes > home_left_votes:
        return False  # home clustered right = defending right = attacks LEFT
    elif home_left_votes > home_right_votes:
        return True   # home clustered left = defending left = attacks RIGHT
    else:
        return True   # fallback


# ---- Score Tracking ----

def load_score_data(game_id):
    """Load score progression from PBP CSV for a specific game.

    Returns list of (period, game_clock_seconds, home_score, away_score) sorted by time.
    """
    pbp_path = os.path.join(PROJECT_DIR, "pbp_2015_16.csv")
    if not os.path.exists(pbp_path):
        return []

    df = pd.read_csv(pbp_path)
    gid = int(game_id) if isinstance(game_id, str) else game_id
    game_df = df[df.GAME_ID == gid].copy()

    if game_df.empty:
        return []

    scores = []
    for _, row in game_df.iterrows():
        if pd.notna(row.get("SCORE")):
            score_str = str(row["SCORE"]).strip()
            parts = score_str.split(" - ")
            if len(parts) == 2:
                try:
                    away_score = int(parts[0].strip())
                    home_score = int(parts[1].strip())
                except ValueError:
                    continue
                # Convert PCTIMESTRING "MM:SS" to seconds
                time_str = str(row.get("PCTIMESTRING", "12:00"))
                time_parts = time_str.split(":")
                gc_seconds = int(time_parts[0]) * 60 + int(time_parts[1]) if len(time_parts) == 2 else 720
                period = int(row.get("PERIOD", 1))
                scores.append((period, gc_seconds, home_score, away_score))

    scores.sort(key=lambda x: (x[0], -x[1]))  # Sort by period asc, clock desc
    return scores


def build_score_lookup(scores):
    """Build a lookup: (quarter, game_clock_rounded) → (home_score, away_score).

    Forward-fills scores so every frame gets the most recent score.
    """
    if not scores:
        return {}
    lookup = {}
    for period, gc, hs, aws in scores:
        lookup[(period, gc)] = (hs, aws)
    return lookup


def get_score_at(score_lookup, scores_list, quarter, game_clock):
    """Get the most recent score at a given quarter + game clock."""
    gc_int = int(game_clock)
    # Exact match first
    if (quarter, gc_int) in score_lookup:
        return score_lookup[(quarter, gc_int)]
    # Find most recent score in this quarter before this clock time
    best = (0, 0)
    for period, gc, hs, aws in scores_list:
        if period < quarter:
            best = (hs, aws)
        elif period == quarter and gc >= gc_int:
            best = (hs, aws)
        elif period == quarter and gc < gc_int:
            break
        elif period > quarter:
            break
    return best


# ---- Half-Court Normalization ----

def normalize_to_half_court(x, y, attacking_right):
    if attacking_right:
        hc_x = COURT_LENGTH - x
        hc_y = COURT_WIDTH - y
    else:
        hc_x = x
        hc_y = y
    return float(np.clip(hc_x, 0, HALF_COURT)), float(np.clip(hc_y, 0, COURT_WIDTH))


# ---- State Building ----

def build_state(moment, prev_positions, attacking_right, pid_map, device):
    """Build 66D normalized state from a raw moment with velocity."""
    if len(moment) < 6 or len(moment[5]) < 11:
        return None, None

    ball = moment[5][0]
    players = moment[5][1:]
    ball_pos = np.array([ball[2], ball[3]])
    shot_clock = moment[3] if moment[3] is not None else 24.0

    # Determine offensive team from ball proximity
    team_dists = {}
    for p in players:
        tid = p[0]
        d = np.linalg.norm(np.array([p[2], p[3]]) - ball_pos)
        if tid not in team_dists or d < team_dists[tid]:
            team_dists[tid] = d

    if not team_dists:
        return None, None

    offensive_team_id = min(team_dists, key=team_dists.get)

    # Find ball handler (closest offensive player to ball)
    bh_id, bh_dist = None, 999
    for p in players:
        if p[0] == offensive_team_id:
            d = np.linalg.norm(np.array([p[2], p[3]]) - ball_pos)
            if d < bh_dist:
                bh_dist = d
                bh_id = p[1]

    if bh_id is None or bh_dist > 5.0:
        return None, None

    # Separate players into BH, teammates, defenders with half-court coords
    bh_raw = None
    teammates = []  # [(hc_pos, raw_pos, player_id)]
    defenders = []  # [(hc_pos, raw_pos)]

    for p in players:
        raw_pos = np.array([p[2], p[3]])
        hc = np.array(normalize_to_half_court(p[2], p[3], attacking_right))
        if p[1] == bh_id:
            bh_raw = raw_pos
            bh_hc = hc
        elif p[0] == offensive_team_id:
            teammates.append((hc, raw_pos, p[1]))
        else:
            defenders.append((hc, raw_pos, p[1]))

    if bh_raw is None:
        return None, None

    # Compute REAL distance to target basket (for display + backcourt check)
    target_basket = np.array([88.75, 25.0]) if attacking_right else np.array([5.25, 25.0])
    real_dist = float(np.linalg.norm(bh_raw - target_basket))

    # Suppress agent when ball handler is in backcourt (own half)
    if attacking_right and bh_raw[0] < HALF_COURT:
        return None, None
    if not attacking_right and bh_raw[0] > HALF_COURT:
        return None, None

    # Compute features in half-court space
    dist_basket = float(np.linalg.norm(bh_hc - BASKET))
    is_three = 1 if dist_basket >= 22 else 0

    # Defender features
    def_dists = sorted([float(np.linalg.norm(bh_hc - d[0])) for d in defenders])
    if not def_dists:
        def_dists = [50.0]
    closest_def = def_dists[0]
    help_def = def_dists[1] if len(def_dists) > 1 else 50.0
    n_def_6ft = sum(1 for d in def_dists if d <= 6)

    # Sort defenders by distance to BH
    def_sorted = sorted(defenders, key=lambda d: np.linalg.norm(bh_hc - d[0]))
    while len(def_sorted) < 5:
        def_sorted.append((np.array([47.0, 25.0]), np.array([47.0, 25.0]), 0))
    def_sorted = def_sorted[:5]

    # Sort teammates by distance to BH
    tm_sorted = sorted(teammates, key=lambda t: np.linalg.norm(bh_hc - t[0]))
    while len(tm_sorted) < 4:
        tm_sorted.append((np.array([25.0, 25.0]), np.array([25.0, 25.0]), 0))
    tm_sorted = tm_sorted[:4]

    # Teammate features
    tm_openness = []
    for t_hc, _, _ in tm_sorted:
        if defenders:
            tm_openness.append(min(float(np.linalg.norm(t_hc - d[0])) for d in defenders))
        else:
            tm_openness.append(30.0)

    grid_col = int(np.clip(bh_hc[0] / 47.0 * 10, 0, 9))
    grid_row = int(np.clip(bh_hc[1] / 50.0 * 5, 0, 4))

    # Build raw 78D state
    raw = np.zeros(78, dtype=np.float32)
    raw[0] = grid_row * 10 + grid_col
    raw[1] = dist_basket
    raw[2] = closest_def
    raw[3] = help_def
    raw[4] = n_def_6ft
    raw[5] = max(tm_openness) if tm_openness else 0
    raw[6] = sum(1 for o in tm_openness if o > 6)
    raw[7] = shot_clock
    raw[8] = max(0, (7 - shot_clock) / 7)
    raw[9] = is_three
    raw[10] = get_fg(bh_id, dist_basket)

    for i in range(4):
        raw[11 + i] = tm_openness[i]
        t_hc = tm_sorted[i][0]
        t_dist = float(np.linalg.norm(t_hc - BASKET))
        raw[15 + i] = get_fg(tm_sorted[i][2], t_dist)
        raw[19 + i] = t_dist

    for i in range(5):
        raw[23 + i * 2] = def_sorted[i][0][0]
        raw[24 + i * 2] = def_sorted[i][0][1]
    for i in range(4):
        raw[33 + i * 2] = tm_sorted[i][0][0]
        raw[34 + i * 2] = tm_sorted[i][0][1]

    # Velocities from position buffer
    if prev_positions is not None:
        # Ball-handler velocity
        if bh_id in prev_positions:
            prev = prev_positions[bh_id]
            raw[41] = np.clip((bh_hc[0] - prev[0]) / TIME_DELTA, -20, 20)
            raw[42] = np.clip((bh_hc[1] - prev[1]) / TIME_DELTA, -20, 20)

        # Defender velocities (position-matched)
        for i, (d_hc, _, d_id) in enumerate(def_sorted[:5]):
            if d_id and d_id in prev_positions:
                prev = prev_positions[d_id]
                raw[43 + i * 2] = np.clip((d_hc[0] - prev[0]) / TIME_DELTA, -20, 20)
                raw[44 + i * 2] = np.clip((d_hc[1] - prev[1]) / TIME_DELTA, -20, 20)
            elif prev_positions:
                # Position-based matching for defenders without IDs
                best_vel = (0.0, 0.0)
                best_d = 999.0
                for pid, prev_pos in prev_positions.items():
                    dd = np.linalg.norm(d_hc - prev_pos)
                    if dd < best_d and dd < 10.0:
                        best_d = dd
                        best_vel = (
                            np.clip((d_hc[0] - prev_pos[0]) / TIME_DELTA, -20, 20),
                            np.clip((d_hc[1] - prev_pos[1]) / TIME_DELTA, -20, 20),
                        )
                raw[43 + i * 2] = best_vel[0]
                raw[44 + i * 2] = best_vel[1]

        # Teammate velocities (ID-matched)
        for i, (t_hc, _, t_id) in enumerate(tm_sorted[:4]):
            if t_id and t_id in prev_positions:
                prev = prev_positions[t_id]
                raw[53 + i * 2] = np.clip((t_hc[0] - prev[0]) / TIME_DELTA, -20, 20)
                raw[54 + i * 2] = np.clip((t_hc[1] - prev[1]) / TIME_DELTA, -20, 20)

    # (Player IDs now set at indices 73-77 after pass lane features)

    # Normalize continuous features (0-60)
    # Pass lane features (indices 61-72)
    for i in range(4):
        t_hc = tm_sorted[i][0]
        # Point-to-segment distances from each defender to BH→teammate line
        lane_dists = []
        for d_hc, _, _ in def_sorted[:5]:
            AP = d_hc - bh_hc
            AB = t_hc - bh_hc
            ab_sq = float(np.dot(AB, AB))
            if ab_sq < 0.01:
                lane_dists.append(float(np.linalg.norm(AP)))
            else:
                proj_t = np.clip(float(np.dot(AP, AB)) / ab_sq, 0.0, 1.0)
                closest = bh_hc + proj_t * (t_hc - bh_hc)
                lane_dists.append(float(np.linalg.norm(d_hc - closest)))
        raw[61 + i] = min(lane_dists) if lane_dists else 47.0          # min_def_dist_lane
        raw[65 + i] = sum(1 for d in lane_dists if d < 6.0)            # corridor_defs
        raw[69 + i] = float(np.linalg.norm(t_hc - bh_hc))             # pass_distance

    # Player IDs (compact mapping) at indices 73-77
    raw[73] = float(pid_map.get(bh_id, 0))
    for i in range(4):
        raw[74 + i] = float(pid_map.get(tm_sorted[i][2], 0))

    normalized = (raw - OBS_LOW) / OBS_RANGE
    normalized[73:78] = raw[73:78]  # Keep IDs raw

    # Build info dict for frontend
    contest = "OPEN" if closest_def > 6 else ("CONTESTED" if closest_def > 2 else "TIGHT")
    info = {
        "bh": bh_id,
        "fg": round(float(raw[10]), 3),
        "dd": round(closest_def, 1),
        "dist": round(real_dist, 1),
        "contest": contest,
        "off_team": offensive_team_id,
        "teammates": [
            {"id": tm_sorted[i][2], "raw": tm_sorted[i][1].tolist()}
            for i in range(min(4, len(tm_sorted)))
        ],
    }

    state_t = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
    return state_t, info


# ---- Game Processing ----

AGENT_SAMPLE_RATE = 12  # Run agent every 12th frame (matches training 2Hz)


def process_game(game_json, agent, device, pid_map):
    """Process all moments in a game, returning frames with agent inference.

    Optimizations:
    - Deduplicates moments by timestamp (events overlap → 2.5x duplication)
    - Runs agent inference every 12th frame (matches training sample rate)
    - Carries forward last recommendation for intermediate frames
    - Uses per-quarter direction detection (not per-frame)
    - Includes score data from play-by-play CSV
    """
    events = game_json["events"]
    if not events:
        return {"game_info": {}, "frames": []}

    home_id = events[0]["home"]["teamid"]
    visitor_id = events[0]["visitor"]["teamid"]

    # Load PBP data for this game
    pbp_path = os.path.join(PROJECT_DIR, "pbp_2015_16.csv")
    pbp_df = pd.read_csv(pbp_path) if os.path.exists(pbp_path) else pd.DataFrame()

    # Load score data from PBP
    game_id = game_json.get("gameid", "0")
    scores_list = load_score_data(game_id)
    score_lookup = build_score_lookup(scores_list)

    # Player name lookup
    pnames = {}
    for ev in events[:10]:
        for p in ev.get("home", {}).get("players", []):
            pnames[p["playerid"]] = {
                "name": p.get("lastname", "?"),
                "first": p.get("firstname", ""),
                "num": p.get("jersey", ""),
                "team": home_id,
            }
        for p in ev.get("visitor", {}).get("players", []):
            pnames[p["playerid"]] = {
                "name": p.get("lastname", "?"),
                "first": p.get("firstname", ""),
                "num": p.get("jersey", ""),
                "team": visitor_id,
            }

    # Collect and deduplicate moments by timestamp
    seen_ts = set()
    all_moments = []
    for ev in events:
        for m in ev.get("moments", []):
            ts = m[1]
            if ts not in seen_ts:
                seen_ts.add(ts)
                all_moments.append(m)
    all_moments.sort(key=lambda m: m[1])

    total = len(all_moments)
    if total == 0:
        return {"game_info": {}, "frames": []}

    # Determine direction once from settled Q1 possessions — locked for the half
    home_attacks_right = determine_direction_from_settled_possessions(
        all_moments, home_id, visitor_id
    )

    # Build game info
    h = TEAM_INFO.get(home_id, ("#0066CC", "HOME", "Home"))
    v = TEAM_INFO.get(visitor_id, ("#CC6600", "AWAY", "Away"))
    game_info = {
        "home": {"id": home_id, "abbr": h[1], "name": h[2], "color": h[0]},
        "away": {"id": visitor_id, "abbr": v[1], "name": v[2], "color": v[0]},
        "players": {str(k): v for k, v in pnames.items()},
        "total_frames": total,
    }

    # Process frames
    position_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
    frames = []
    last_agent_data = None

    for idx, moment in enumerate(all_moments):
        if len(moment) < 6 or len(moment[5]) < 11:
            frames.append(None)
            continue

        quarter = moment[0]
        game_clock = moment[2]
        shot_clock = moment[3] if moment[3] is not None else 0

        ball = moment[5][0]
        players_raw = moment[5][1:]

        # Find which team's player is closest to ball (= ball handler)
        ball_pos = np.array([ball[2], ball[3]])
        best_tid, best_pid, best_dist, best_x = None, None, 999, 47.0
        for p in players_raw:
            d = np.linalg.norm(np.array([p[2], p[3]]) - ball_pos)
            if d < best_dist:
                best_dist = d
                best_tid = p[0]
                best_pid = p[1]
                best_x = p[2]

        offensive_team_id = best_tid if best_tid else home_id

        # Direction: locked per-half from settled possession analysis
        if offensive_team_id == home_id:
            attacking_right = home_attacks_right if quarter <= 2 else not home_attacks_right
        else:
            attacking_right = (not home_attacks_right) if quarter <= 2 else home_attacks_right

        # Build current positions (half-court) for velocity buffer
        ar_for_buffer = attacking_right
        current_hc_positions = {}
        for p in players_raw:
            hc = normalize_to_half_court(p[2], p[3], ar_for_buffer)
            current_hc_positions[p[1]] = np.array(hc)
        position_buffer.append(current_hc_positions)

        # Get score at this moment
        score = get_score_at(score_lookup, scores_list, quarter, game_clock)

        # Build frame data
        player_data = []
        for p in players_raw:
            player_data.append([p[1], p[0], round(p[2], 1), round(p[3], 1)])

        frame = {
            "q": quarter,
            "gc": round(game_clock, 1),
            "sc": round(shot_clock, 1),
            "ball": [round(ball[2], 1), round(ball[3], 1)],
            "players": player_data,
            "hs": score[0],  # home score
            "as": score[1],  # away score
        }

        # Run agent inference every AGENT_SAMPLE_RATE frames
        if idx % AGENT_SAMPLE_RATE == 0:
            prev_positions = position_buffer[0] if len(position_buffer) >= FRAME_BUFFER_SIZE else None

            state_t, info = build_state(
                moment, prev_positions, attacking_right, pid_map, device
            )

            if state_t is not None and info is not None:
                with torch.no_grad():
                    q_vals = agent.online_net(state_t).cpu().numpy()[0]
                action = int(np.argmax(q_vals))

                tm_ids = [t["id"] for t in info["teammates"]]

                agent_data = {
                    "bh": info["bh"],
                    "action": action,
                    "action_name": ACTION_NAMES[action],
                    "qv": [round(float(q), 3) for q in q_vals],
                    "fg": info["fg"],
                    "dd": info["dd"],
                    "dist": info["dist"],
                    "contest": info["contest"],
                    "off_team": info["off_team"],
                    "tm": tm_ids,
                }

                if action > 0 and action - 1 < len(info["teammates"]):
                    target = info["teammates"][action - 1]
                    agent_data["target"] = {
                        "id": target["id"],
                        "x": round(target["raw"][0], 1),
                        "y": round(target["raw"][1], 1),
                    }
                else:
                    agent_data["target"] = None

                last_agent_data = agent_data
            else:
                last_agent_data = None

        frame["agent"] = last_agent_data
        frames.append(frame)

    return {"game_info": game_info, "frames": frames}


# ---- FastAPI App ----

app = FastAPI(title="NBA Shot Selection Viz")
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Available models for the viz
MODELS = {
    # --- Top 3 PBRS models (best results) ---
    "pbrs_lr1e4": {
        "label": "★ BEST: PBRS LR=1e-4 (EPSA +0.177, 67% pass, 33% shoot)",
        "type": "dueling",
        "weights": "results_v9/pbrs_lr1e4/dueling_dqn_weights.pth",
    },
    "pbrs_smallnet": {
        "label": "PBRS Small Net [256,128] (Dueling, 2nd best)",
        "type": "dueling",
        "weights": "results_v9/pbrs_smallnet/dueling_dqn_weights.pth",
    },
    "pbrs_gamma099": {
        "label": "PBRS gamma=0.99 (Dueling, 3rd best)",
        "type": "dueling",
        "weights": "results_v9/pbrs_gamma099/dueling_dqn_weights.pth",
    },
    # --- DQN comparison ---
    "pbrs_dqn_nodist": {
        "label": "PBRS DQN (EPSA +0.123, 65% pass, 35% shoot)",
        "type": "dqn",
        "weights": "results_v9/pbrs_dqn_nodist/dqn_weights.pth",
    },
    # --- Older pre-PBRS models (for comparison) ---
    "pbrs_cql00": {
        "label": "PBRS Dueling CQL=0 (early, 50% pass)",
        "type": "dueling",
        "weights": "results_v9/pbrs_dueling_cql00/dueling_dqn_weights.pth",
    },
    "dueling_ps10": {
        "label": "Pre-PBRS: Dueling ps=1.0 (pass-heavy, 83% pass)",
        "type": "dueling",
        "weights": "results_v9/dueling_ps10/dueling_dqn_weights.pth",
    },
    "dueling_ps06": {
        "label": "Pre-PBRS: Dueling ps=0.6 (shoot-biased)",
        "type": "dueling",
        "weights": "results_v9/dueling_ps06/dueling_dqn_weights.pth",
    },
    # v8 mean-pool excluded — incompatible architecture (66D vs 78D per-entity)
}

# Load agent and pid_map at startup
AGENT = None
DEVICE = None
PID_MAP = {}
CACHED_GAME = {}
CURRENT_MODEL = "pbrs_lr1e4"


@app.on_event("startup")
async def startup():
    global AGENT, DEVICE, PID_MAP
    PID_MAP = load_pid_map()

    model = MODELS["pbrs_lr1e4"]
    AGENT, DEVICE = load_agent(model["type"])


@app.get("/api/models")
async def list_models():
    """List available models for the frontend selector."""
    return {
        "models": [{"id": k, "label": v["label"]} for k, v in MODELS.items()],
        "current": CURRENT_MODEL,
    }


@app.get("/api/switch_model/{model_id}")
async def switch_model(model_id: str):
    """Switch the active model. Requires reloading the game."""
    global AGENT, DEVICE, CURRENT_MODEL
    if model_id not in MODELS:
        return JSONResponse({"error": f"Unknown model: {model_id}"}, 404)

    model = MODELS[model_id]
    weights_path = os.path.join(PROJECT_DIR, model["weights"])
    if not os.path.exists(weights_path):
        return JSONResponse({"error": f"Weights not found: {weights_path}"}, 404)

    # Load new agent
    AGENT, DEVICE = load_agent(model["type"])
    # Load specific weights
    AGENT.load(weights_path)
    CURRENT_MODEL = model_id
    print(f"Switched to model: {model_id} ({weights_path})")

    return {"status": "ok", "model": model_id, "label": model["label"]}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(PROJECT_DIR, "templates", "index.html")
    with open(html_path) as f:
        return f.read()


@app.get("/api/games")
async def list_games():
    if not os.path.isdir(DATA_DIR):
        return JSONResponse({"error": f"Data dir not found: {DATA_DIR}"}, 404)

    games = []
    pattern = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})\.(\w+)\.at\.(\w+)\.7z$")
    months = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
        "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
        "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
    }

    for f in sorted(os.listdir(DATA_DIR)):
        m = pattern.match(f)
        if m:
            month, day, year, away, home = m.groups()
            games.append({
                "file": f,
                "label": f"{months.get(month, month)} {day} {year} | {away} @ {home}",
                "away": away,
                "home": home,
                "date": f"{year}-{month}-{day}",
            })

    return {"games": games}


@app.get("/api/load/{filename}")
async def load_game_sse(filename: str):
    """SSE endpoint for loading progress. Stores processed data in CACHED_GAME."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse({"error": "Game file not found"}, 404)

    async def generate():
        global CACHED_GAME

        yield f"data: {json.dumps({'type': 'progress', 'pct': 5, 'msg': 'Extracting game file...'})}\n\n"

        game_json = load_game_7z(filepath)
        total_events = len(game_json.get("events", []))

        yield f"data: {json.dumps({'type': 'progress', 'pct': 15, 'msg': f'Processing {total_events} events...'})}\n\n"

        result = process_game(game_json, AGENT, DEVICE, PID_MAP)

        yield f"data: {json.dumps({'type': 'progress', 'pct': 90, 'msg': 'Preparing visualization...'})}\n\n"

        # Store result server-side, send game_info only via SSE
        CACHED_GAME = result

        yield f"data: {json.dumps({'type': 'complete', 'game_info': result['game_info']})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/frames")
async def get_frames():
    """Returns cached game frames as gzipped JSON. Called after /api/load completes."""
    if not CACHED_GAME or "frames" not in CACHED_GAME:
        return JSONResponse({"error": "No game loaded"}, 404)
    return JSONResponse(content=CACHED_GAME["frames"])
