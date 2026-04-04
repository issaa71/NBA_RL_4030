# NBA Shot Selection — Reinforcement Learning

**AISE 4030 Phase 3 — Group 11**
Evan Park, Issa Ahmed, John Kaniuk, Trent Jones, Aaqil Kassam Murji
Western Engineering, 2026

---

## Overview

An offline RL system that learns when to shoot and who to pass to in NBA possessions. Trained on 116,928 real possessions from the 2015-16 SportVU tracking dataset (631 games). The agent observes a 78-dimensional state (all 10 player positions, velocities, shooting stats, pass lane features, and player ID embeddings) and chooses from 5 actions: shoot or pass to one of 4 teammates.

**Best model:** Dueling DQN with PBRS reward, EPSA +0.239 (55.4% agreement with NBA players, identifies 811 bad shots and 4,476 missed opportunities).

## Key Contributions

- **Potential-Based Reward Shaping (PBRS) with EPV** — Novel for basketball RL. Uses max(teammate EPVs) as potential function, theoretically preserving optimal policy (Ng et al., 1999).
- **Per-entity Q-value architecture** — Each teammate scored individually through shared Deep Sets heads. No published basketball RL paper does this.
- **Type-aware advantage normalization** — Prevents 4-vs-1 pass bias in Dueling DQN by normalizing pass advantages within the pass group only.

## Results

| Model | EPSA | Pass% | Shoot% |
|---|---|---|---|
| **Dueling DQN (PBRS)** | **+0.239** | 58% | 84% |
| DQN (PBRS) | +0.228 | 58% | 86% |

Best weights: `results_v9/pbrs_lr1e4/dueling_dqn_weights.pth`

## Project Structure

```
environment.py            # Gymnasium env, 78D state, PBRS rewards
q_network.py              # Per-entity QNetwork (Deep Sets)
dueling_q_network.py      # Per-entity DuelingQNetwork + type-aware normalization
dqn_agent.py              # DQN agent (Double DQN, CQL)
dueling_dqn_agent.py      # Dueling DQN agent
replay_buffer.py          # Experience replay with terminal oversampling
training_script.py        # Training loop with best-checkpoint saving
evaluate_and_compare.py   # Evaluation + comparison plots
app.py                    # FastAPI web visualization backend
templates/index.html      # Canvas frontend for court visualization
demo.py                   # CLI demo for presentations
00_pull_player_stats.py   # Pull per-player zone FG% from nba_api
02_segment_possessions.py # Process SportVU tracking data into possessions
config.yaml               # All hyperparameters
test_components.py        # 28 unit tests
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train
python training_script.py --agent dueling --config config.yaml

# Evaluate (generates comparison plots)
python evaluate_and_compare.py

# Web visualization
uvicorn app:app --port 8001
# Open http://localhost:8001

# CLI demo
python demo.py --agent both --num 5
```

## Architecture

**State (78D):** 73 continuous features + 5 player IDs mapped through `nn.Embedding(450, 8)`.

**Network:** Deep Sets backbone with per-entity scoring:
- Defender phi (shared): `(x,y,vx,vy)` per defender -> mean pool -> 32D
- Teammate phi (shared): `(x,y,vx,vy,openness,fg%,dist,lane features + embedding)` per teammate -> 32D each (NOT pooled)
- Shoot head: context + defender pool -> Q(shoot)
- Pass head (shared): teammate phi + context + defender pool -> Q(pass_i)

**Reward (PBRS):** Potential function = max(teammate EPVs) excluding ball handler. Shoot reward = `bh_epv - phi(s)`. Pass reward = `gamma * phi(s') - phi(s)`.

**Training:** 100K episodes, Double DQN, LR=1e-4, gamma=0.95, linear epsilon decay over 40K episodes, hard target sync every 100 episodes, CQL alpha=0.0.

## Data

SportVU 2015-16 tracking data from [linouk23/NBA-Player-Movements](https://github.com/linouk23/NBA-Player-Movements). Play-by-play from [sumitrodatta/nba-alt-awards](https://github.com/sumitrodatta/nba-alt-awards). Player FG% from [nba_api](https://github.com/swar/nba_api).

## References

- Ng, Harada & Russell (ICML 1999) — Potential-based reward shaping
- Wang et al. (ICML 2016) — Dueling DQN
- van Hasselt et al. (AAAI 2016) — Double DQN
- Cervone et al. (JASA 2014/2016) — Expected Possession Value
- Chen et al. (CIKM 2022) — ReLiable: offline RL on SportVU
- Yanai et al. (AAAI 2022) — Q-Ball: DQN for player evaluation
