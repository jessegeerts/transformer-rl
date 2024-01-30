import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)  # todo: we don't need this afaik (MLP can take care of this)

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
        weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):

    def __init__(self, h_dim, max_T, n_heads, drop_p, mlp_dim, include_mlp=True, layer_norm=True, activation='relu'):
        super().__init__()
        self.include_mlp = include_mlp
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.layer_norm = layer_norm
        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'gelu':
            self.activation = nn.GELU
        else:
            raise ValueError("activation must be 'relu' or 'gelu'.")

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, mlp_dim),
            self.activation(),
            nn.Linear(mlp_dim, h_dim),
            nn.Dropout(drop_p),
        )

        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm

        x = x + self.attention(x)  # residual
        if self.layer_norm:
            x = self.ln1(x)
        if self.include_mlp:
            x = x + self.mlp(x)  # residual
            if self.layer_norm:
                x = self.ln2(x)
        return x


class Transformer(nn.Module):

    def __init__(self, token_dim=None, n_blocks=None, h_dim=None, max_T=None, n_heads=None, drop_p=None,
                 mlp_dim=None, config=None, include_mlp=True, layer_norm=True):
        """
        :param token_dim: dimensionality of input tokens
        :param n_blocks: number of transformer blocks
        :param h_dim: dimensionality of hidden states
        :param max_T: maximum sequence length
        :param n_heads: number of attention heads
        :param drop_p: dropout probability
        :param mlp_dim: dimensionality of mlp hidden layer
        :param config: config object
        :param include_mlp: whether to include an mlp layer after each attention block. This should be a list of size
        n_blocks, where each element is a boolean indicating whether to include an mlp layer after the corresponding
        attention block. If None, a mlp layer is included after each attention block (default). If False, no mlp layers
        are included, which makes this a attention-only model.
        :param layer_norm: whether to include layer normalization after each attention block.
        """
        super().__init__()

        if config:
            token_dim = config.token_dim
            n_blocks = config.n_blocks
            h_dim = config.h_dim
            max_T = config.max_T
            n_heads = config.n_heads
            drop_p = config.drop_p
            mlp_dim = config.mlp_dim
            layer_norm = config.layer_norm
        elif None in [token_dim, n_blocks, h_dim, max_T, n_heads, drop_p, mlp_dim]:
            raise ValueError("Either provide a complete config or all hyperparameters individually.")

        # embed input tokens and positions
        self.proj_token = nn.Embedding(token_dim, h_dim)
        self.proj_stim = nn.Linear(token_dim, h_dim)

        # parameter = trainable weight matrix
        self.dropout = nn.Dropout(drop_p)

        # transformer blocks
        if include_mlp is True:
            include_mlp = [True] * n_blocks
        elif include_mlp is False:
            include_mlp = [False] * n_blocks
        else:
            assert len(include_mlp) == n_blocks, "include_mlp must be a list of size n_blocks."

        blocks = [Block(h_dim, max_T, n_heads, drop_p, mlp_dim, include_mlp[b], layer_norm) for b in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # projection head
        self.ln = nn.LayerNorm(h_dim)
        self.proj_head = nn.Linear(h_dim, token_dim)

        self.label_embedding = nn.Embedding(config.num_classes, config.h_dim)  # embed labels into stimulus space
        self.pos_embedding = nn.Embedding(config.max_T, config.h_dim)
        self.proj_head = nn.Linear(config.h_dim, config.num_classes)
        self.P = config.max_T  # number of possible positions

    def forward(self, stimuli, labels):
        B, T, D = stimuli.shape  # batch size, sequence length, stimulus dimension
        # (note, sequence length includes the query stimulus)
        # Embed stimuli
        # stimuli = self.proj_stim(stimuli)
        # Embed labels
        # embedded_labels = self.label_embedding(labels)
        embedded_labels = F.one_hot(labels, num_classes=D).float()
        # Embed positions
        # make positions random to learn translation-invariant computation
        # todo: random start position are now the same for all sequences in the batch. Try making them different.
        seq_len = (T - 1) * 2 + 1
        start_pos = np.random.choice(self.P - seq_len + 1)  # randomly choose a starting position
        positions = torch.arange(start_pos, start_pos + seq_len, device=stimuli.device)
        pos_embeddings = F.one_hot(positions, num_classes=self.P).float().to(stimuli.device)

        # Create interleaved sequence with an extra stimulus at the end
        ctx_stimuli = stimuli[:, :-1, :]  # Exclude the last stimulus (query stimulus)
        h = torch.cat([ctx_stimuli, embedded_labels], dim=1)
        interleave_indices = torch.arange(h.shape[1]).view(-1, h.shape[1]//2).t().reshape(-1)
        h = h[:, interleave_indices, :].view(B, -1, D)
        h = torch.cat([h, stimuli[:, -1, :].unsqueeze(1)], dim=1)  # Add the query stimulus at the end
        # h += pos_embeddings.unsqueeze(0)
        h = torch.cat([h, pos_embeddings.unsqueeze(0).expand(B, seq_len, self.P)], dim=-1)
        # Transformer and prediction
        h = self.ln(self.transformer(h))
        pred = self.proj_head(h)

        # Select the output corresponding to the last stimulus (query stimulus)
        query_pred = pred[:, -1, :]

        return query_pred
