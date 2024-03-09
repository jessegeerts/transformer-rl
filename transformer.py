import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np


def sinusoidal_embeddings(max_len, d_model):
    """
    Generate sinusoidal embeddings.

    Args:
    - max_len (int): Maximum sequence length.
    - d_model (int): Embedding dimension.

    Returns:
    - Tensor of shape (max_len, d_model) representing the sinusoidal embeddings.
    """

    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # Position tensor
    div_term = torch.exp(torch.arange(0.0, d_model, 2.0) * -(torch.log(torch.tensor(10000.0)) / d_model))
    embeddings = torch.zeros((max_len, d_model))

    embeddings[:, 0::2] = torch.sin(pos * div_term)
    embeddings[:, 1::2] = torch.cos(pos * div_term)

    return embeddings


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))  # fixme: check this
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)  # todo: try tiled dropout
        attention = self.att_drop(normalized_weights @ v)  # multiply weights by values and apply dropout

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    """Transformer block. Consists of masked causal attention and a feedforward layer.

    Note: we use layernorms both after the attention and after the feedforward layer.
    """
    def __init__(self, h_dim, max_T, n_heads, drop_p, widening_factor=4):
        super().__init__()

        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, widening_factor * h_dim),
            nn.GELU(),
            nn.Linear(widening_factor * h_dim, h_dim),
            nn.Dropout(drop_p),
        )

        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm

        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x


class Transformer(nn.Module):

    def __init__(self, token_dim=None, n_blocks=None, h_dim=None, max_T=None, n_heads=None, drop_p=None,
                 widening_factor=4, config=None):
        super().__init__()

        if config:
            token_dim = config.token_dim
            n_blocks = config.n_blocks
            h_dim = config.h_dim
            max_T = config.max_T
            n_heads = config.n_heads
            drop_p = config.drop_p
            widening_factor = config.widening_factor
        elif None in [token_dim, n_blocks, h_dim, max_T, n_heads, drop_p]:
            raise ValueError("Either provide a complete config or all hyperparameters individually.")

        # embed input tokens and positions
        self.proj_token = nn.Embedding(token_dim, h_dim)
        self.proj_stim = nn.Linear(token_dim, h_dim)

        # parameter = trainable weight matrix
        init_param_vals = torch.randn(1, max_T, h_dim) / math.sqrt(h_dim)
        self.position_embedding = nn.Parameter(init_param_vals)
        #self.sinusoidal = sinusoidal_embeddings(max_T, h_dim).unsqueeze(0)

        self.embed_timesteps = nn.Embedding(max_T, h_dim)

        self.dropout = nn.Dropout(drop_p)

        # transformer blocks
        blocks = [Block(h_dim, max_T, n_heads, drop_p, widening_factor=widening_factor) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # projection head
        self.ln = nn.LayerNorm(h_dim)
        self.proj_head = nn.Linear(h_dim, token_dim)

    def onehot_position_embedding(self, T, max_T):
        """Create one-hot position embeddings."""
        positions = torch.arange(T)
        return F.one_hot(positions, num_classes=max_T).float()

    def forward(self, x):
        B, T = x.shape

        # token and pos embedding
        token_h = self.proj_token(x)
        pos_h = self.position_embedding[:, :T, :]
        h = token_h + pos_h

        # transformer and prediction
        h = self.ln(self.transformer(h))
        pred = self.proj_head(h)

        return pred

    def forward_embedded(self, x, start_time=0, pos_embedding_type='sinusoidal'):
        B, T, C = x.shape

        # token and pos embedding
        token_h = self.proj_stim(x)
        # pos embedding
        if pos_embedding_type == 'sinusoidal':
            pos_h = self.sinusoidal[:, start_time:start_time + T, :]
        elif pos_embedding_type == 'param':
            pos_h = self.position_embedding[:, start_time:start_time + T, :]
        elif pos_embedding_type == 'learned':
            timesteps = torch.arange(start_time, start_time + T)
            pos_h = self.embed_timesteps(timesteps.squeeze())
        elif pos_embedding_type == 'onehot':
            pos_h = self.onehot_position_embedding(T, self.position_embedding.size(1))
            pos_h = pos_h.unsqueeze(0).expand(B, -1, -1)
        else:
            raise ValueError('pos_embedding_type must be one of: "sinusoidal", "param", "learned", "onehot"')
        h = token_h + pos_h

        # transformer and prediction
        h = self.ln(self.transformer(h))
        pred = self.proj_head(h)

        return pred

    def pred_loss(self, pred, target):
        # pred (B, T, C)  and target (B, T)
        B, T, C = pred.shape
        return F.cross_entropy(pred.view(B * T, C), target.view(B * T))
