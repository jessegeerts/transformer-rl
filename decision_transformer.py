"""Simple implementation of causal decision transformer, based on Misha Laskin's implementation of GPT.
"""
import math

import torch
from torch import nn as nn
from torch.nn import functional as F


class CausalMultiheadAttention(nn.Module):
    """Multihead attention with causal mask."""
    def __init__(self, h_dim, max_T, num_heads, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.q_net = nn.Linear(h_dim, h_dim)  # query
        self.k_net = nn.Linear(h_dim, h_dim)  # key
        self.v_net = nn.Linear(h_dim, h_dim)  # value

        self.proj_net = nn.Linear(h_dim, h_dim)  # projection

        self.att_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # apply causal mask (upper triangular matrix)
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, h_dim * n_heads

        N, D = self.num_heads, C // self.num_heads  # number of heads, hidden dimension per head

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights [B, N, T, D] * [B, N, D, T] = [B, N, T, T]
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # apply causal mask
        weights = weights.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)  # attention weights

        # attention (B, N, T, T) * (B, N, T, D) = (B, N, T, D)
        attention = self.att_dropout(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D) -> (B, T, C)
        attention = attention.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj_dropout(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = CausalMultiheadAttention(h_dim, max_T, n_heads, drop_p)
        self. mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),
            nn.GELU(),
            nn.Linear(h_dim * 4, h_dim),
            nn.Dropout(drop_p)
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p, max_timestep=4096,
                 discrete_actions=True, discrete_states=True):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        # transformer blocks
        input_seq_len = 3 * context_len  # 3 * context_len because we concatenate state, action, reward
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # projection heads
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)

        if discrete_states:
            self.embed_state = nn.Embedding(state_dim, h_dim)
        else:
            self.embed_state = torch.nn.Linear(state_dim, h_dim)

        if discrete_actions:
            self.embed_action = nn.Embedding(act_dim, h_dim)
            use_action_tanh = False
        else:
            self.embed_action = torch.nn.Linear(act_dim, h_dim)
            use_action_tanh = True

        # prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = torch.nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

    def forward(self, timesteps, states, actions, returns_to_go):
        """

        :param timesteps:
        :param states:
        :param actions:
        :param returns_to_go:
        :return:
        """
        if states.ndim == 2:
            B, T = states.shape  # batch size, sequence length
        else:
            B, T, _ = states.shape  # batch size, sequence length, state dimension

        time_embeddings = self.embed_timestep(timesteps)  # [B, T, H]

        # time embeddings are treated similar to positional embeddings in the original transformer
        # they are added to the input embeddings
        state_embeddings = self.embed_state(states) + time_embeddings  # [B, T, H]
        action_embeddings = self.embed_action(actions) + time_embeddings  # [B, T, H]
        rtg_embeddings = self.embed_rtg(returns_to_go) + time_embeddings  # [B, T, H]

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, ..., r_T, s_T, a_T)
        h = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)  # [B, 3 * T, H]

        h = self.embed_ln(h)  # layer norm

        # transformer and prediction
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x H) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # we do this such that we can predict the next state, action and rtg given the current state, action and rtg
        h = h.reshape(B, 3, T, self.h_dim)  #.permute(0, 2, 1, 3)  # [B, T, 3, H]

        # get predictions
        rtg_preds = self.predict_rtg(h[:, 2])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # predict next action given r, s

        return state_preds, action_preds, rtg_preds


if __name__ == '__main__':
    from environments.gridworld import GridWorld

    env = GridWorld()
    # collect random walk trajectories from grid world environment





    dt = DecisionTransformer(3, 2, 2, 32, 10, 4, 0.1)
    print(dt)