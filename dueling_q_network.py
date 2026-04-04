"""
dueling_q_network.py
====================
Per-Entity Dueling Q-Network with Deep Sets and player ID embeddings.

v9: Same per-entity architecture as q_network.py but with Dueling decomposition.
V(s) = state value (from context + defenders)
A(s, pass_i) = advantage of passing to teammate i (from their phi + context)
A(s, shoot) = advantage of shooting (from context + defenders)
Q(s, a) = V(s) + A(s, a) - mean(A)

AISE 4030 - Group 11
"""

import torch
import torch.nn as nn
from typing import List


class DuelingQNetwork(nn.Module):
    """
    Per-Entity Dueling Q-Network with Deep Sets backbone.

    Splits Q into V(s) + A(s,a) - mean(A). The value stream estimates
    how good the state is. The advantage streams estimate how much better
    each action is relative to average.
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

        if num_players > 0:
            self.player_embedding = nn.Embedding(num_players, embed_dim)
        else:
            self.player_embedding = None

        # Defender phi
        self.phi_defender = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        # Teammate phi
        tm_input = 10 + (embed_dim if num_players > 0 else 0)
        self.phi_teammate = nn.Sequential(
            nn.Linear(tm_input, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        ctx_dim = 13 + (embed_dim if num_players > 0 else 0)
        base_input = ctx_dim + 32  # context + defender pool

        # Value stream: V(s) from context + defenders
        self.value_stream = nn.Sequential(
            nn.Linear(base_input, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Shoot advantage: A(s, shoot) from context + defenders
        self.shoot_advantage = nn.Sequential(
            nn.Linear(base_input, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Pass advantage (SHARED): A(s, pass_i) from teammate phi + context + defenders
        pass_input = 32 + ctx_dim + 32
        self.pass_advantage = nn.Sequential(
            nn.Linear(pass_input, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.init_weights()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.shape[0]

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

        if self.player_embedding is not None:
            pid_start = self.continuous_dim
            player_ids = state[:, pid_start:pid_start + 5].long().clamp(
                0, self.num_players - 1)
            bh_embed = self.player_embedding(player_ids[:, 0])
            tm_embeds = [self.player_embedding(player_ids[:, i + 1]) for i in range(4)]
        else:
            bh_embed = torch.zeros(batch, 0, device=state.device)
            tm_embeds = [torch.zeros(batch, 0, device=state.device)] * 4

        # Defenders: phi → mean pool
        def_features = []
        for i in range(5):
            d_feat = torch.cat([
                def_pos[:, i*2:i*2+2], def_vel[:, i*2:i*2+2],
            ], dim=1)
            def_features.append(self.phi_defender(d_feat))
        def_pooled = torch.stack(def_features, dim=1).mean(dim=1)

        # Context
        ctx = torch.cat([context, bh_vel, bh_embed], dim=1)
        base = torch.cat([ctx, def_pooled], dim=1)

        # V(s)
        value = self.value_stream(base)  # (batch, 1)

        # A(s, shoot)
        a_shoot = self.shoot_advantage(base)  # (batch, 1)

        # A(s, pass_i) per teammate
        a_passes = []
        for i in range(4):
            t_feat = torch.cat([
                tm_pos[:, i*2:i*2+2], tm_vel[:, i*2:i*2+2],
                tm_open[:, i:i+1], tm_fg[:, i:i+1], tm_dist[:, i:i+1],
                tm_lane_min[:, i:i+1], tm_corridor[:, i:i+1], tm_pass_dist[:, i:i+1],
                tm_embeds[i],
            ], dim=1)
            tm_phi = self.phi_teammate(t_feat)
            pass_input = torch.cat([tm_phi, ctx, def_pooled], dim=1)
            a_passes.append(self.pass_advantage(pass_input))

        # Type-aware advantage normalization:
        # Normalize pass advantages within the pass group only.
        # Shoot is a singleton — no normalization needed.
        # This prevents the 4-pass group from inflating through mean subtraction.
        a_pass_stack = torch.cat(a_passes, dim=1)  # (batch, 4)
        a_pass_norm = a_pass_stack - a_pass_stack.mean(dim=1, keepdim=True)

        # Q(shoot) = V(s) + A(shoot)
        # Q(pass_i) = V(s) + A(pass_i) - mean(A_pass)
        q_shoot = value + a_shoot
        q_passes = value + a_pass_norm
        return torch.cat([q_shoot, q_passes], dim=1)  # (batch, 5)

    def init_weights(self) -> None:
        for module in [self.phi_defender, self.phi_teammate,
                       self.value_stream, self.shoot_advantage,
                       self.pass_advantage]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    nn.init.zeros_(layer.bias)


def build_dueling_q_network(config: dict, device: torch.device) -> DuelingQNetwork:
    net_cfg = config["network"]
    net = DuelingQNetwork(
        state_dim=net_cfg["state_dim"],
        action_dim=net_cfg["action_dim"],
        hidden_layers=net_cfg["hidden_layers"],
        dropout_rate=net_cfg.get("dropout_rate", 0.0),
        continuous_dim=net_cfg.get("continuous_dim", net_cfg["state_dim"]),
        num_players=net_cfg.get("num_players", 0),
        embed_dim=net_cfg.get("embed_dim", 8),
    )
    return net.to(device)
