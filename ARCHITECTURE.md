# NBA Shot Selection — Codebase Architecture (v10)

Guide for understanding and continuing development.

---

## Project Overview

An offline RL system that replays real NBA possessions from SportVU 2015-16 tracking data. Two agents (DQN, Dueling DQN) learn when to shoot and who to pass to using a 78-dimensional state vector with player ID embeddings. Rewards use Potential-Based Reward Shaping (PBRS) with EPV as the potential function. CQL is disabled (alpha=0.0) because it crushes terminal shoot Q-values.

Best model: **Dueling DQN, EPSA +0.239** (`results_v9/pbrs_lr1e4/dueling_dqn_weights.pth`)

---

## File Map

### Data Pipeline (run in order)

```
00_pull_player_stats.py    → player_zone_fg.pkl
    Pulls per-player, per-zone FG% from nba_api ShotChartDetail.
    7 distance zones (0-5ft through 30+ft). ~286 players.
    Volume threshold: <15 3PT attempts → 0% (kills outliers).
    Only needs to run once.

02_segment_possessions.py  → processed_possessions.pkl
    Loads raw .7z game files from local clone of
    linouk23/NBA-Player-Movements (~/Downloads/).
    Downloads play-by-play CSV and merges with tracking data.
    Processes one game at a time (memory-safe for 636 games).
    Extracts 78 state features per decision point:
    - All 10 player positions + velocities
    - Per-teammate openness, zone FG%, distance to basket
    - Pass lane features (min defender dist, corridor defenders, pass distance)
    - Player IDs (ball-handler + 4 teammates)
    Full dataset: 631 games, 116,928 possessions. Takes ~35 min.
```

### Core RL Components

```
environment.py
    NBAShootOrPassEnv (Gymnasium compatible)
    ├── __init__()     — Observation space (78D), action space (Discrete 5)
    ├── reset()        — Samples random possession, builds initial state
    ├── step(action)   — action=0: shoot (terminal), action=1-4: pass to teammate
    │                    PBRS reward shaping:
    │                      Shoot: bh_epv - Φ(s)       (terminal, Φ(s')=0)
    │                      Pass:  γ·Φ(s') - Φ(s)      (continuing)
    │                      Shot clock urgency penalty when clock < 7s
    ├── _build_state() — Extracts 78 features (73 continuous + 5 player IDs)
    │                    Continuous features normalized to [0,1]
    │                    Player IDs left as raw ints for embedding lookup
    ├── _compute_epv() — Potential function Φ(s) = max(teammate EPVs)
    │                    EXCLUDES ball handler (continuation value only)
    │                    EPV = zone_fg% × point_value × contest_factor
    ├── _compute_raw_shoot_reward() — fg% × pts × min(def_dist/6, 1)
    ├── _is_turnover() — Checks turnover flag
    └── load_and_preprocess_dataset() — Loads pickle, remaps player IDs
                                         to compact indices, splits by game_id

q_network.py
    QNetwork(nn.Module) — Per-Entity Deep Sets Architecture
    ├── player_embedding: nn.Embedding(450, 8)
    ├── phi_defender: shared [4→32→32] per defender (x,y,vx,vy)
    │   → mean pool across 5 defenders → 32D
    ├── phi_teammate: shared [18→32→32] per teammate
    │   (x,y,vx,vy,openness,fg,dist,lane,corridor,passdist + 8D embed)
    │   → 32D each, NOT pooled (per-entity scoring)
    ├── shoot_head: [ctx+def_pool → 64→32→1] → Q(shoot)
    ├── pass_head: [tm_phi+ctx+def_pool → 64→16→1] → Q(pass_i)
    │   (shared weights across all 4 teammates)
    └── Context = 11 game features + 2 BH velocity + 8D BH embedding = 21D

dueling_q_network.py
    DuelingQNetwork(nn.Module) — Same per-entity backbone + Dueling split
    ├── Same phi_defender, phi_teammate, embeddings as QNetwork
    ├── value_stream: [base → 64→32→1] → V(s)
    ├── shoot_advantage: [base → 32→1] → A(shoot)
    ├── pass_advantage: [tm_phi+ctx+def_pool → 64→16→1] → A(pass_i)
    │   (shared weights, per-entity)
    └── Type-aware normalization:
        Q(shoot)  = V(s) + A(shoot)
        Q(pass_i) = V(s) + A(pass_i) - mean(A_pass)
        Pass advantages normalized WITHIN pass group only.
        Prevents 4-vs-1 bias from standard mean subtraction.

replay_buffer.py
    ReplayBuffer
    ├── Fixed-capacity circular buffer (deque, 100K capacity)
    ├── N-step return support (default n=1, n>1 is toxic for short episodes)
    ├── Terminal transition oversampling (3x)
    ├── push() — stores transitions
    ├── sample(batch_size, device) — uniform random mini-batch
    └── is_ready(min_size)

dqn_agent.py
    DQNAgent
    ├── online_net + target_net (QNetwork)
    ├── choose_action() — epsilon-greedy
    ├── store_transition() → replay buffer
    ├── update() — Double DQN + CQL penalty (alpha=0.0 disables CQL)
    │   TD target: r + γ * Q_target(s', argmax_online(s'))
    │   CQL: loss += alpha * (logsumexp(Q) - Q_data)
    ├── decay_epsilon() — linear over 40K episodes
    └── on_episode_end() — decay epsilon, hard sync target every 100 eps

dueling_dqn_agent.py
    DuelingDQNAgent — identical interface to DQNAgent
    └── Uses DuelingQNetwork instead of QNetwork
```

### Training & Evaluation

```
training_script.py
    Main entry point. Modes: train, eval, deploy.
    ├── train() — Full training loop with best-checkpoint saving
    │   Saves model only when eval EPSA improves
    │   Tracks 4 metrics: rewards, losses, lengths, epsilons
    │   Stores terminated (not done) for correct Q-bootstrap
    ├── evaluate() — Greedy policy on test set, returns mean EPSA
    └── CLI: --config, --mode, --agent (dqn|dueling)

evaluate_and_compare.py
    Post-training evaluation and plot generation.
    ├── 3 baselines: random, always-shoot, behavior (historical)
    ├── 6 plots: learning_speed, loss_convergence, stability_variance,
    │           epsilon_decay, episode_lengths, final_performance
    ├── Decision maps: shoot-probability heatmap from real test data
    └── Output: comparison_plots/

demo.py
    CLI demo for presentations.
    └── Shows Q-values, action selection, simulated outcomes
```

### Web Visualization

```
app.py
    FastAPI backend serving real-time court visualization.
    ├── Loads .7z game files from linouk23/NBA-Player-Movements
    ├── Deduplicates moments (events overlap → 2.5x duplication)
    ├── Runs agent inference every 12 frames (2Hz, matches training)
    ├── build_state() — mirrors environment.py normalization exactly
    ├── Direction detection from Q1 settled possessions
    │   (team clustered on side = DEFENDING, inverted logic)
    ├── Score tracking from play-by-play CSV
    ├── Model selector dropdown (switch models live)
    └── SSE streaming for load progress

templates/index.html
    HTML5 Canvas frontend.
    ├── Court rendering with team-colored player dots
    ├── Agent recommendation overlay (SHOOT/PASS + target arrow)
    ├── Q-value display with player names
    ├── Scoreboard, shot clock, game clock
    └── Playback controls (play/pause, speed, frame scrub)
```

---

## 78-Feature State Vector

| # | Feature | Range | Source |
|---|---|---|---|
| 0 | grid_zone | 0-49 | Half-court 10x5 grid |
| 1 | distance_to_basket | 0-50 ft | Euclidean from basket |
| 2 | closest_defender_dist | 0-50 ft | Nearest defender |
| 3 | help_defender_dist | 0-50 ft | 2nd nearest defender |
| 4 | num_defenders_within_6ft | 0-5 | NBA "contested" definition |
| 5 | best_teammate_openness | 0-50 ft | Most open teammate's defender dist |
| 6 | num_open_teammates | 0-4 | Teammates with defender > 6ft |
| 7 | shot_clock | 0-24 s | Seconds remaining |
| 8 | shot_clock_urgency | 0-1 | max(0, (7-shot_clock)/7) |
| 9 | is_three_point_zone | 0-1 | Beyond 22ft from basket |
| 10 | ball_handler_zone_fg_pct | 0-1 | Player's FG% in this distance zone |
| 11-14 | teammate_{1-4}_openness | 0-50 ft | Each teammate's defender dist |
| 15-18 | teammate_{1-4}_zone_fg_pct | 0-1 | Each teammate's FG% at their zone |
| 19-22 | teammate_{1-4}_dist_to_basket | 0-50 ft | Each teammate's distance |
| 23-32 | defender_{1-5}_x, _y | 0-47, 0-50 | Half-court positions (sorted by dist to BH) |
| 33-40 | teammate_{1-4}_x, _y | 0-47, 0-50 | Half-court positions (sorted by dist to BH) |
| 41-42 | ball_handler_vx, vy | -20 to 20 ft/s | Frame-to-frame velocity |
| 43-52 | defender_{1-5}_vx, _vy | -20 to 20 ft/s | Defender velocities |
| 53-60 | teammate_{1-4}_vx, _vy | -20 to 20 ft/s | Teammate velocities |
| 61-64 | min_def_dist_lane_{1-4} | 0-47 ft | Closest defender to pass lane |
| 65-68 | corridor_defs_{1-4} | 0-5 | Defenders within 6ft of pass lane |
| 69-72 | pass_distance_{1-4} | 0-47 ft | BH-to-teammate distance |
| 73-77 | player IDs (BH + 4 TM) | 0-449 | Compact indices → nn.Embedding(450, 8) |

Continuous features (0-72) normalized to [0,1]. Player IDs (73-77) left as raw ints.

---

## Data Flow

```
GitHub: linouk23/NBA-Player-Movements (636 .7z game files)
  + PBP CSV: sumitrodatta/nba-alt-awards
    ↓ (02_segment_possessions.py)
processed_possessions.pkl (116,928 possessions, 78 features each)
    ↓ (environment.py: load_and_preprocess_dataset)
    ├── Remap player IDs to compact 0-449 indices
    ├── train_possessions (80%, split by game_id)
    └── test_possessions  (20%, split by game_id)
            ↓ (training_script.py)
            ├── DQN agent ←→ env.step() loop (100K episodes)
            └── Dueling DQN agent ←→ env.step() loop
                    ↓
                    results_v9/pbrs_lr1e4/ (best Dueling, EPSA +0.239)
                    results_v9/pbrs_dqn_nodist/ (best DQN, EPSA +0.228)
                            ↓ (evaluate_and_compare.py)
                            comparison_plots/ (7 plots)
```

---

## Key Design Decisions

### Why PBRS with EPV potential
Fixed baselines (any constant value) create bistable behavior — agent either always shoots or always passes. Split baselines violate reward shaping guarantees. PBRS with Φ(s) = max(teammate EPVs) automatically produces state-dependent selectivity. Shoot is positive only when ball handler has the best look; pass is positive only when it improves court position. Theoretically guaranteed to preserve optimal policy (Ng et al., 1999). Novel for basketball RL.

### Why CQL alpha = 0.0
CQL's penalty accumulates on shoot (terminal) actions with zero recovery via bootstrapping. Pass actions recover because they bootstrap from next-state values. Even α=0.05 causes shoot Q-values to be depressed ~1.26 units. PBRS already handles the reward structure; CQL's conservatism is counterproductive here.

### Why per-entity Q-values
Distance-sorted teammate ordering corrupts Bellman targets (same action index = different player each frame). Mean-pooling destroys per-teammate identity. Per-entity scoring (shared head scoring each phi output individually) solves both and gives 4x training data via weight sharing. Permutation equivariant by construction.

### Why type-aware advantage normalization
Standard Dueling DQN subtracts mean(A) across all 5 actions. With 4 pass + 1 shoot, the mean is dominated by passes, creating ~1σ bias against shooting. Normalizing pass advantages within the pass group prevents this.

### Why no pass penalty with PBRS
The -0.05 pass penalty on top of PBRS makes agents shoot immediately because it adds cumulative cost. PBRS already penalizes non-improving passes through Φ(s') < Φ(s).

### Why 1-step TD
With 5.7-step average episodes, n=3 consumes 52.6% of the episode. Creates heterogeneous return estimates and amplifies bootstrapping error in offline settings. CQL guarantees only hold for 1-step backups.

### Why hard target sync
Soft Polyak updates (τ=0.005) destabilize player embeddings. With ~450 players, many appearing rarely, embeddings need stable gradients. Hard sync every 100 episodes freezes the target for ~570 steps.

### Why volume-adjusted FG%
Players with <15 3PT attempts get 0% for zones 4-6. Bayesian shrinkage wrongly shrunk elite shooters (Curry). Simple threshold preserves high-volume stats and eliminates low-volume outliers (Bogut 1/1 = 100%).

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| state_dim | 78 (73 + 5 IDs) | |
| action_dim | 5 | shoot + 4 passes |
| hidden_layers | [512, 256, 128] | Not used directly (per-entity heads are smaller) |
| num_players | 450 | Compact ID space |
| embed_dim | 8 | Player embedding size |
| learning_rate | 1e-4 | Best hyperparameter (2x default) |
| gamma | 0.95 | |
| epsilon decay | linear, 40K episodes | |
| batch_size | 64 | Small batches = implicit regularization |
| replay_buffer | 100K | |
| target_update_freq | 100 episodes | Hard sync |
| n_step | 1 | |
| grad_clip | 1.0 | |
| num_episodes | 100K | |
| cql_alpha | 0.0 | Disabled |
| pass penalty | 0.0 | PBRS handles it |

---

## Results

### Best Models

| Model | EPSA | Shot EPV | Steps | Pass% | Shoot% |
|---|---|---|---|---|---|
| **Dueling PBRS LR=1e-4** | **+0.239** | **0.614** | 2.32 | 58% | 84% |
| DQN PBRS | +0.228 | 0.603 | 2.41 | 58% | 86% |

### Player Analysis
- 55.4% agreement rate with NBA players
- 811 bad shots identified (avg EPV 0.154 vs league avg 0.375)
- 4,476 missed shot opportunities (avg EPV 0.678)

### Key Findings
1. PBRS with EPV is the correct reward structure (fixed baselines create bistable behavior)
2. CQL must be 0.0 (any positive alpha crushes terminal Q-values)
3. Per-entity Q-values fix pass action instability
4. Network size matters less than reward design (top 8 models within +0.228 to +0.239)
5. Hyperparameters barely matter once architecture and reward are right
