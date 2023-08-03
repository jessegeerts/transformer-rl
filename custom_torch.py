import torch
import torch.nn as nn

class SoLU(nn.Module):
    """Softmax linear unit activation function as introduced in the Transformer Circuits thread:

    https://transformer-circuits.pub/2022/solu/index.html
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.softmax(x, dim=-1)
