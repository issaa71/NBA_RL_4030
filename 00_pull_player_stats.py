"""
00_pull_player_stats.py
=======================
Pulls per-player, per-distance-zone FG% from nba_api for the 2015-16 season.
Applies Beta-Binomial Empirical Bayes shrinkage with position-group priors
to handle low-volume samples (e.g., Bogut 1/1 from three → 100% raw).

v7: Position-group priors (bigs vs perimeter) + 20-attempt minimum for
three-point zones (4-6). Below 20 attempts → use group average directly.

Zones:
    Zone 0: 0-5 ft   (at rim)
    Zone 1: 5-9 ft   (short midrange)
    Zone 2: 10-14 ft (midrange)
    Zone 3: 15-19 ft (long midrange)
    Zone 4: 20-24 ft (3PT line area)
    Zone 5: 25-29 ft (deep three)
    Zone 6: 30+ ft   (heave / very deep)

Output: player_zone_fg.pkl
    {player_id: {zone_idx: fg_pct}, "__league_avg__": {zone_idx: fg_pct}}

AISE 4030 - Group 11
"""

import pickle
import time
import sys
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
from scipy.special import gammaln
from nba_api.stats.endpoints import ShotChartDetail
from nba_api.stats.static import players


SEASON = "2015-16"
DELAY = 0.6
THREE_PT_MIN_ATTEMPTS = 20  # Minimum attempts for zones 4-6 to use player data


def dist_to_zone_index(distance_ft: float) -> int:
    if distance_ft < 5:   return 0
    elif distance_ft < 10: return 1
    elif distance_ft < 15: return 2
    elif distance_ft < 20: return 3
    elif distance_ft < 25: return 4
    elif distance_ft < 30: return 5
    else:                  return 6


def get_position_group(position_str: str) -> str:
    """Map NBA position string to 'bigs' or 'perimeter'."""
    if not position_str:
        return "perimeter"
    pos = position_str.upper().strip()
    big_patterns = ["C", "PF", "C-F", "F-C"]
    for p in big_patterns:
        if pos == p or pos.startswith(p + "-") or pos.endswith("-" + p):
            return "bigs"
    return "perimeter"


def beta_binomial_log_likelihood(params, makes_list, attempts_list):
    """Negative log-likelihood of Beta-Binomial model for MLE fitting."""
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return 1e10
    ll = 0
    for k, n in zip(makes_list, attempts_list):
        ll += (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
               + gammaln(k + alpha) + gammaln(n - k + beta)
               - gammaln(n + alpha + beta)
               + gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta))
    return -ll


def fit_beta_binomial_prior(makes_list, attempts_list):
    """Fit Beta(alpha, beta) prior via MLE on observed (makes, attempts) pairs."""
    if len(makes_list) < 3:
        # Not enough data to fit — use method of moments
        fg_vals = [m / a for m, a in zip(makes_list, attempts_list) if a > 0]
        if not fg_vals:
            return 5.0, 5.0
        mean_fg = np.mean(fg_vals)
        return max(1.0, mean_fg * 50), max(1.0, (1 - mean_fg) * 50)

    result = minimize(
        beta_binomial_log_likelihood,
        x0=[5.0, 5.0],
        args=(makes_list, attempts_list),
        method="Nelder-Mead",
        options={"maxiter": 1000},
    )
    alpha, beta = max(0.5, result.x[0]), max(0.5, result.x[1])
    return alpha, beta


def pull_player_shots(player_id: int) -> list:
    try:
        response = ShotChartDetail(
            player_id=player_id,
            team_id=0,
            season_nullable=SEASON,
            context_measure_simple="FGA",
        )
        shots = response.get_data_frames()[0]
        return shots.to_dict("records")
    except Exception as e:
        print(f"  Warning: failed for player {player_id}: {e}")
        return []


def main():
    with open("processed_possessions.pkl", "rb") as f:
        possessions = pickle.load(f)

    needed_ids = set()
    for poss in possessions:
        for dp in poss:
            pid = dp.get("player_id")
            if pid:
                needed_ids.add(pid)
            bh = dp.get("ball_handler_player_id")
            if bh:
                needed_ids.add(bh)
            for tid in dp.get("teammate_player_ids", []):
                if tid:
                    needed_ids.add(tid)
    del possessions

    print(f"Need stats for {len(needed_ids)} unique players")

    # Build position lookup from nba_api
    all_players = players.get_players()
    player_positions = {}
    for p in all_players:
        player_positions[p["id"]] = "perimeter"  # default

    # Try to get positions from CommonPlayerInfo for our players
    # Fall back to event data positions
    print("Looking up player positions...")
    from nba_api.stats.endpoints import CommonPlayerInfo
    for i, pid in enumerate(sorted(needed_ids)):
        if (i + 1) % 50 == 0:
            print(f"  Position lookup {i+1}/{len(needed_ids)}...")
        try:
            info = CommonPlayerInfo(player_id=pid)
            df = info.get_data_frames()[0]
            if len(df) > 0:
                pos = df.iloc[0].get("POSITION", "")
                player_positions[pid] = get_position_group(str(pos))
        except Exception:
            pass
        time.sleep(0.3)

    print(f"Position groups: {sum(1 for v in player_positions.values() if v == 'bigs')} bigs, "
          f"{sum(1 for v in player_positions.values() if v == 'perimeter')} perimeter")

    # Pull shots for each player
    player_zone_raw = {}  # {pid: {zone: {"made": int, "attempts": int}}}
    league_zone_stats = defaultdict(lambda: {"made": 0, "attempts": 0})

    for i, pid in enumerate(sorted(needed_ids)):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Pulling shots {i+1}/{len(needed_ids)} (ID: {pid})...")

        shots = pull_player_shots(pid)
        if not shots:
            time.sleep(DELAY)
            continue

        zone_stats = defaultdict(lambda: {"made": 0, "attempts": 0})
        for shot in shots:
            dist = shot.get("SHOT_DISTANCE", 0)
            made = shot.get("SHOT_MADE_FLAG", 0)
            zone = dist_to_zone_index(dist)
            zone_stats[zone]["attempts"] += 1
            zone_stats[zone]["made"] += int(made)
            league_zone_stats[zone]["attempts"] += 1
            league_zone_stats[zone]["made"] += int(made)

        player_zone_raw[pid] = dict(zone_stats)
        time.sleep(DELAY)

    # League averages
    league_avg = {}
    for zone_idx, stats in league_zone_stats.items():
        if stats["attempts"] > 0:
            league_avg[zone_idx] = stats["made"] / stats["attempts"]

    # Group players by position for Beta-Binomial prior fitting
    group_zone_data = defaultdict(lambda: defaultdict(lambda: {"makes": [], "attempts": []}))
    for pid, zones in player_zone_raw.items():
        group = player_positions.get(pid, "perimeter")
        for zone_idx, stats in zones.items():
            if stats["attempts"] > 0:
                group_zone_data[group][zone_idx]["makes"].append(stats["made"])
                group_zone_data[group][zone_idx]["attempts"].append(stats["attempts"])

    # Fit Beta-Binomial priors per (group, zone)
    priors = {}  # {(group, zone): (alpha, beta)}
    group_averages = {}  # {(group, zone): float} — for minimum-attempts fallback
    for group in ["bigs", "perimeter"]:
        for zone in range(7):
            data = group_zone_data[group][zone]
            if data["makes"]:
                alpha, beta = fit_beta_binomial_prior(data["makes"], data["attempts"])
                priors[(group, zone)] = (alpha, beta)
                total_made = sum(data["makes"])
                total_att = sum(data["attempts"])
                group_averages[(group, zone)] = total_made / total_att if total_att > 0 else league_avg.get(zone, 0.40)
            else:
                priors[(group, zone)] = (2.0, 3.0)
                group_averages[(group, zone)] = league_avg.get(zone, 0.40)

    print("\nFitted Beta-Binomial priors:")
    for group in ["bigs", "perimeter"]:
        print(f"  {group}:")
        for zone in range(7):
            a, b = priors[(group, zone)]
            prior_mean = a / (a + b)
            gavg = group_averages[(group, zone)]
            print(f"    Zone {zone}: α={a:.1f}, β={b:.1f}, prior_mean={prior_mean:.3f}, group_avg={gavg:.3f}")

    # Apply shrinkage to compute adjusted FG%
    result = {}
    for pid, zones in player_zone_raw.items():
        result[pid] = {}
        group = player_positions.get(pid, "perimeter")
        for zone_idx, stats in zones.items():
            made = stats["made"]
            attempts = stats["attempts"]

            if attempts == 0:
                result[pid][zone_idx] = group_averages.get((group, zone_idx), league_avg.get(zone_idx, 0.40))
                continue

            # Three-point zones: minimum 20 attempts, else use group average
            if zone_idx >= 4 and attempts < THREE_PT_MIN_ATTEMPTS:
                result[pid][zone_idx] = group_averages.get((group, zone_idx), league_avg.get(zone_idx, 0.40))
                continue

            # Apply Beta-Binomial shrinkage
            alpha, beta = priors.get((group, zone_idx), (2.0, 3.0))
            adjusted = (made + alpha) / (attempts + alpha + beta)
            result[pid][zone_idx] = adjusted

    result["__league_avg__"] = league_avg

    # Print verification table
    print("\nVerification (before → after shrinkage):")
    test_players = {
        101106: "Bogut",
        201939: "Curry",
        1626157: "KAT",
        1626171: "Portis",
    }
    zone_names = ["0-5ft", "5-9ft", "10-14ft", "15-19ft", "20-24ft", "25-29ft", "30+ft"]
    for pid, name in test_players.items():
        raw = player_zone_raw.get(pid, {})
        adj = result.get(pid, {})
        group = player_positions.get(pid, "?")
        print(f"  {name} ({group}):")
        for z in range(7):
            r = raw.get(z, {})
            raw_fg = r["made"] / r["attempts"] if r.get("attempts", 0) > 0 else None
            adj_fg = adj.get(z)
            att = r.get("attempts", 0)
            raw_str = f"{raw_fg:.1%}" if raw_fg is not None else "N/A"
            adj_str = f"{adj_fg:.1%}" if adj_fg is not None else "N/A"
            flag = " ← CAPPED (3PT min)" if z >= 4 and att < THREE_PT_MIN_ATTEMPTS and att > 0 else ""
            print(f"    Zone {z} ({zone_names[z]}): {raw_str} → {adj_str} ({att} att){flag}")

    print(f"\nPlayers with data: {len(result) - 1}")

    with open("player_zone_fg.pkl", "wb") as f:
        pickle.dump(result, f)
    print("Saved to player_zone_fg.pkl")


if __name__ == "__main__":
    main()
