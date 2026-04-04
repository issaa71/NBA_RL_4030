"""
q_network.py
============
Per-Entity Q-Network with Deep Sets and player ID embeddings.

v9 architecture: Instead of mean-pooling teammate phi outputs (which destroys
per-teammate identity), each teammate's phi output is scored individually
through a shared scoring head. This produces per-teammate Q(pass_to_i) values
directly tied to each teammate's features.

Q(shoot) comes from ball-handler context + defender pool.
Q(pass_i) comes from teammate_i's phi output + ball-handler context.

This is permutation equivariant by construction — no teammate ordering needed.

AISE 4030 - Group 11
"""

import torch
import torch.nn as nn
from typing import List


class QNetwork(nn.Module):
    """
    Per-Entity Q-Network with Deep Sets backbone.

    State vector layout (78D):
      [0-10]  Game context (grid_zone, distances, shot_clock, etc.)
      [11-22] Per-teammate aggregate features (openness, fg%, dist)
      [23-32] Defender positions (5 × x,y)
      [33-40] Teammate positions (4 × x,y)
      [41-42] Ball-handler velocity
      [43-52] Defender velocities (5 × vx,vy)
      [53-60] Teammate velocities (4 × vx,vy)
      [61-72] Pass lane features (min_def×4, corridor×4, dist×4)
      [73-77] Player IDs (BH + 4 teammates)

    Architecture:
      - phi_defender: shared network per defender → mean pool → 32D
      - phi_teammate: shared network per teammate → 32D each (NOT pooled)
      - shoot_head: context + defender_pool → Q(shoot)
      - pass_head: per-teammate phi + context → Q(pass_to_i) (shared weights)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int],
        dropout_rate: float = 0.0,
        continuous_dim: int = 0,
        num_players: int = 0,
        embed_dim: int = 8,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_dim = continuous_dim if continuous_dim > 0 else state_dim
        self.num_players = num_players
        self.embed_dim = embed_dim

        # Player ID embeddings
        if num_players > 0:
            self.player_embedding = nn.Embedding(num_players, embed_dim)
        else:
            self.player_embedding = None

        # Defender phi: (x, y, vx, vy) → 32D
        self.phi_defender = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        # Teammate phi: (x, y, vx, vy, open, fg, dist, lane, corridor, passdist + embed) → 32D
        tm_input = 10 + (embed_dim if num_players > 0 else 0)
        self.phi_teammate = nn.Sequential(
            nn.Linear(tm_input, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        # Context: 11 game features + 2 BH velocity + BH embedding
        ctx_dim = 13 + (embed_dim if num_players > 0 else 0)

        # Shoot head: context + defender_pool → Q(shoot)
        shoot_input = ctx_dim + 32  # context + defender pool
        self.shoot_head = nn.Sequential(
            nn.Linear(shoot_input, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Pass scoring head (SHARED for all 4 teammates):
        # teammate_phi(32) + context(ctx_dim) + defender_pool(32) → Q(pass_to_i)
        pass_input = 32 + ctx_dim + 32
        self.pass_head = nn.Sequential(
            nn.Linear(pass_input, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.init_weights()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]

        # --- Extract state components ---
        context = state[:, :11]
        tm_open = state[:, 11:15]
        tm_fg = state[:, 15:19]
        tm_dist = state[:, 19:23]
        def_pos = state[:, 23:33]
        tm_pos = state[:, 33:41]
        bh_vel = state[:, 41:43]
        def_vel = state[:, 43:53]
        tm_vel = state[:, 53:61]
        tm_lane_min = state[:, 61:65]
        tm_corridor = state[:, 65:69]
        tm_pass_dist = state[:, 69:73]

        # Player IDs
        if self.player_embedding is not None:
            pid_start = self.continuous_dim
            player_ids = state[:, pid_start:pid_start + 5].long().clamp(
                0, self.num_players - 1)
            bh_embed = self.player_embedding(player_ids[:, 0])
            tm_embeds = [self.player_embedding(player_ids[:, i + 1]) for i in range(4)]
        else:
            bh_embed = torch.zeros(batch, 0, device=state.device)
            tm_embeds = [torch.zeros(batch, 0, device=state.device)] * 4

        # --- Defenders: phi → mean pool ---
        def_features = []
        for i in range(5):
            d_feat = torch.cat([
                def_pos[:, i*2:i*2+2], def_vel[:, i*2:i*2+2],
            ], dim=1)
            def_features.append(self.phi_defender(d_feat))
        def_pooled = torch.stack(def_features, dim=1).mean(dim=1)  # (batch, 32)

        # --- Ball-handler context ---
        ctx = torch.cat([context, bh_vel, bh_embed], dim=1)  # (batch, ctx_dim)

        # --- Q(shoot) from context + defenders ---
        shoot_input = torch.cat([ctx, def_pooled], dim=1)
        q_shoot = self.shoot_head(shoot_input)  # (batch, 1)

        # --- Q(pass_to_i) from each teammate's phi + context ---
        q_passes = []
        for i in range(4):
            t_feat = torch.cat([
                tm_pos[:, i*2:i*2+2], tm_vel[:, i*2:i*2+2],
                tm_open[:, i:i+1], tm_fg[:, i:i+1], tm_dist[:, i:i+1],
                tm_lane_min[:, i:i+1], tm_corridor[:, i:i+1], tm_pass_dist[:, i:i+1],
                tm_embeds[i],
            ], dim=1)
            tm_phi = self.phi_teammate(t_feat)  # (batch, 32)
            pass_input = torch.cat([tm_phi, ctx, def_pooled], dim=1)
            q_pass_i = self.pass_head(pass_input)  # (batch, 1)
            q_passes.append(q_pass_i)

        # Combine: [Q(shoot), Q(pass1), Q(pass2), Q(pass3), Q(pass4)]
        return torch.cat([q_shoot] + q_passes, dim=1)  # (batch, 5)

    def init_weights(self) -> None:
        for module in [self.phi_defender, self.phi_teammate,
                       self.shoot_head, self.pass_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)


def build_q_network(config: dict, device: torch.device) -> QNetwork:
    net_cfg = config["network"]
    net = QNetwork(
        state_dim=net_cfg["state_dim"],
        action_dim=net_cfg["action_dim"],
        hidden_layers=net_cfg["hidden_layers"],
        dropout_rate=net_cfg.get("dropout_rate", 0.0),
        continuous_dim=net_cfg.get("continuous_dim", net_cfg["state_dim"]),
        num_players=net_cfg.get("num_players", 0),
        embed_dim=net_cfg.get("embed_dim", 8),
    )
    return net.to(device)
