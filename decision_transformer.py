"""Simple implementation of causal decision transformer, based on Misha Laskin's implementation of GPT.
"""
import math
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F


def sinusoidal_embeddings(n_pos, d_hid):
    position = np.arange(n_pos)[:, np.newaxis]
    div_term = np.exp(np.arange(0., d_hid, 2) * -(np.log(10000.0) / d_hid))
    sinusoid_table = np.zeros_like(position * div_term, dtype= np.float32)
    sinusoid_table[:, 0::2] = np.sin(position * div_term)
    sinusoid_table[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(sinusoid_table)


class CausalMultiheadAttention(nn.Module):
    """Multihead attention with causal mask."""
    def __init__(self, h_dim, max_T, num_heads, dropout=0.1):
        """
        max_T: maximum sequence length (3 * context_len because we concatenate state, action, reward)
        """
        super().__init__()

        self.num_heads = num_heads
        self.q_net = nn.Linear(h_dim, h_dim)  # query
        self.k_net = nn.Linear(h_dim, h_dim)  # key
        self.v_net = nn.Linear(h_dim, h_dim)  # value

        self.proj_net = nn.Linear(h_dim, h_dim)  # projection

        self.att_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # apply causal mask
        mask = self.causal_mask(max_T)
        self.register_buffer("mask", mask)

    def causal_mask(self, max_T):
        return self.block_causal_mask(max_T)

    @staticmethod
    def standard_causal_mask(max_T):
        """Standard causal mask.
        """
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        return mask

    @staticmethod
    def block_causal_mask(max_T):
        """Causal mask with blocks of size 3 (model can attend to current rtg, s, a).
        """
        blocks = torch.kron(torch.eye(max_T//3), torch.ones((3, 3))).view(1, 1, max_T, max_T)
        tril = torch.tril(torch.ones((max_T, max_T))).view(1, 1, max_T, max_T)
        mask = torch.logical_or(blocks, tril).double()
        return mask

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
        return out, normalized_weights


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, act_fn=nn.GELU()):
        super().__init__()
        self.attention = CausalMultiheadAttention(h_dim, max_T, n_heads, drop_p)

        # Split MLP into two parts to capture activations after the first layer
        self.mlp1 = nn.Sequential(
            nn.Linear(h_dim, h_dim * 4),  # note, the dimensionality of the MLP is 4x the hidden dimension
            act_fn
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(h_dim * 4, h_dim),
            nn.Dropout(drop_p)
        )

        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        att_output, weights = self.attention(x)
        x = x + att_output
        x = self.ln1(x)
        mlp_activations = self.mlp1(x)  # Capture the MLP activations after the first layer
        x = x + self.mlp2(mlp_activations)
        x = self.ln2(x)
        return x, weights, mlp_activations  # Return the activations after the first layer


class StackedBlocks(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, num_blocks, act_fn=nn.GELU()):
        super().__init__()
        self.blocks = nn.ModuleList([Block(h_dim, max_T, n_heads, drop_p, act_fn=act_fn) for _ in range(num_blocks)])

    def forward(self, x):
        all_weights = []
        all_mlp_activations = []
        for block in self.blocks:
            x, weights, mlp_activations = block(x)
            all_weights.append(weights)
            all_mlp_activations.append(mlp_activations)
        return x, all_weights, all_mlp_activations  # Return MLP activations for each layer


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p, max_timestep=4096,
                 discrete_actions=True, discrete_states=True, act_fn=nn.GELU(),
                 action_mask_value=None, state_mask_value=None):
        super().__init__()

        self.state_dim = state_dim + 1  # +1 for masked states
        self.act_dim = act_dim + 1  # +1 for masked actions
        self.h_dim = h_dim
        self.action_mask_value = action_mask_value
        self.state_mask_value = state_mask_value

        # transformer blocks
        input_seq_len = 3 * context_len  # 3 * context_len because we concatenate state, action, reward
        self.transformer = StackedBlocks(h_dim, input_seq_len, n_heads, drop_p, n_blocks, act_fn=act_fn)

        # projection heads
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)

        if discrete_states:
            self.embed_state = nn.Embedding(self.state_dim, h_dim)
        else:
            self.embed_state = torch.nn.Linear(self.state_dim, h_dim)

        if discrete_actions:
            self.embed_action = nn.Embedding(self.act_dim, h_dim)
            use_action_tanh = False
        else:
            self.embed_action = torch.nn.Linear(self.act_dim, h_dim)
            use_action_tanh = True

        # prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, self.state_dim)
        self.predict_action = torch.nn.Sequential(
            *([nn.Linear(h_dim, self.act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
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
        if self.state_mask_value is not None:
            mask_state = states == self.action_mask_value
            state_embeddings[mask_state] = 0
        if self.action_mask_value is not None:
            mask_action = actions == self.state_mask_value
            action_embeddings[mask_action] = 0
        rtg_embeddings = self.embed_rtg(returns_to_go) + time_embeddings  # [B, T, H]

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, ..., r_T, s_T, a_T)
        h = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)  # [B, 3 * T, H]

        h = self.embed_ln(h)  # layer norm

        # transformer and prediction
        h, all_weights, all_mlp_activations = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x H) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # we do this such that we can predict the next state, action and rtg given the current state, action and rtg
        h = h.reshape(B, 3, T, self.h_dim)  # [B, T, 3, H]

        # get predictions
        rtg_preds = self.predict_rtg(h[:, 2])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # predict next action given r, s

        return state_preds, action_preds, rtg_preds, all_weights, all_mlp_activations

