# V9/V10 Complete Experiment Results
## Last updated: April 4, 2026

All v10 experiments use PBRS (Potential-Based Reward Shaping) with EPV potential,
per-entity Q-value architecture, type-aware advantage normalization, CQL=0.0,
pass penalty=0.0, pass distance penalty=0.0.

## PBRS Results (training-time eval, old-style EPSA vs 0.375 baseline)

**Note:** The report uses a separate 500-episode evaluation with consistent EPSA metric:
Dueling = +0.177, DQN = +0.123. Numbers below are from training-time periodic eval
which used a slightly different calculation. Both show the same ranking.

| Model | EPSA | Shot EPV | Steps | Pass% | Shoot% | Key Change |
|---|---|---|---|---|---|---|
| **pbrs_lr1e4** | **+0.239** | **0.614** | 2.32 | 58% | 84% | **LR=1e-4 (2x default)** |
| pbrs_smallnet | +0.236 | 0.611 | 2.34 | 58% | 88% | [256,128] network |
| pbrs_gamma099 | +0.236 | 0.611 | 2.61 | 65% | 86% | gamma=0.99 |
| pbrs_bignet | +0.232 | 0.607 | 2.29 | 52% | 88% | [512,512,256] network |
| pbrs_target200 | +0.232 | 0.607 | 2.52 | 60% | 88% | Target sync every 200 |
| pbrs_buf200k | +0.232 | 0.607 | 2.38 | 59% | 88% | 200K replay buffer |
| pbrs_nodistpen | +0.231 | 0.606 | 2.45 | 61% | 87% | No dist penalty (baseline) |
| pbrs_dqn_nodist | +0.228 | 0.603 | 2.41 | 58% | 86% | DQN (base algorithm) |
| pbrs_sloweps | +0.209 | 0.584 | 2.58 | 65% | 86% | 60K epsilon decay |
| pbrs_batch128 | +0.208 | 0.583 | 2.41 | 58% | 89% | Batch size 128 |
| pbrs_300k | +0.203 | 0.578 | 2.29 | 59% | 87% | 300K episodes |
| pbrs_lr1e5 | +0.157 | 0.532 | 2.28 | 50% | 91% | LR=1e-5 (0.2x default) |
| pbrs_gamma085 | +0.151 | 0.526 | 1.98 | 50% | 92% | gamma=0.85 |

## Earlier Results (pre-PBRS, split baseline)

| Model | EPSA | Steps | Pass% | Notes |
|---|---|---|---|---|
| Dueling ps=1.0 sb=0.75/pb=0.40 | +0.380* | 4.31 | 83% | *inflated by low baseline |
| sb=0.60/pb=0.45 ps=1.0 | +0.104* | 2.18 | 37% | |
| BCQ | -0.427 | 1.00 | 0% | Failed |
| All CQL>0 with PBRS | -0.27 | 1.01 | 1% | CQL kills shoot Q-values |

*Note: pre-PBRS EPSA not directly comparable — different reward structures.

## Key Findings

### Architecture
1. Per-entity Q-values > mean-pooling (novel for basketball RL)
2. Type-aware advantage normalization prevents 4-vs-1 bias
3. Small network [256,128] performs as well as large [512,256,128]
4. Network size matters less than reward design

### Reward Design
1. PBRS with EPV is theoretically grounded and empirically best
2. Fixed baselines create bistable behavior (all shoot or all pass)
3. CQL must be 0.0 — any positive alpha crushes terminal action Q-values
4. No auxiliary penalties needed (pass penalty, distance penalty)

### Hyperparameters
1. LR=1e-4 slightly better than default 5e-5
2. Gamma barely matters (0.90-0.99 all similar)
3. 100K episodes sufficient — 300K didn't improve
4. Bigger replay buffer (200K) marginally helps
5. Target sync frequency (100 vs 200) doesn't matter much

### Player Analysis
- 55.4% agreement rate with NBA players
- Identified 811 bad shots (avg EPV 0.154 vs league avg 0.375)
- Identified 4,476 missed opportunities (avg EPV 0.678)

## Best Models for Report/Presentation
1. **Dueling PBRS LR=1e-4** — best EPSA, best for "advanced algorithm"
2. **DQN PBRS** — best DQN, for "base vs advanced" comparison
3. **Dueling ps=1.0 split baseline** — shows pre-PBRS behavior (for evolution story)
