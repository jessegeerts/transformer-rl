import math

from torch import nn as nn
from torch.nn import functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, h_dim):
        super().__init__()

        self.q_net = nn.Linear(h_dim, h_dim)  # query
        self.k_net = nn.Linear(h_dim, h_dim)  # key
        self.v_net = nn.Linear(h_dim, h_dim)  # value

    def forward(self, x):
        B, T, D = x.shape  # batch size, sequence length, hidden dimension

        q = self.q_net(x)  # [B, T, D]
        k = self.k_net(x)  # [B, T, D]
        v = self.v_net(x)  # [B, T, D]

        # weights [B, T, D] * [B, D, T] = [B, T, T]
        weights = q @ k.transpose(1, 2) / math.sqrt(D)
        normalized_weights = F.softmax(weights, dim=-1)  # attention weights

        # attention [B, T, T] * [B, T, D] = [B, T, D]
        attention = normalized_weights @ v

        return attention
