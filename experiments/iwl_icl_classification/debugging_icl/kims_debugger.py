"""To debug the network, and see if it can make use of positional encodings, we can try to just train the network
to output the Nth item.

TODO: try this with:
- added instead of appended position codes
- sinusoidal position codes instead of one-hot
"""

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb

from experiments.iwl_icl_classification.data import BurstyTrainingDataset
from experiments.iwl_icl_classification.model import Transformer as MyTransformer
from experiments.rules_vs_exemplars.config import TransformerConfig


def _create_positional_encodings(inputs, start_pos=0, max_time=30.0):
    """Generates positional encodings for the input. Adapted from haiku / jax (Chan et al.) to torch  by Jesse Geerts.

    Note, in Chan et al, max_time is set to 30, probably because the sequences were short.

  Args:
    inputs: A tensor of shape [batch_size, seq_len, emb_size].
    max_time: (default 10000) Constant used to scale position by in the
      encodings.

  Returns:
    pos_emb: as defined above, of size [1, seq_len, emb_size].
  """

    _, seq_len, embedding_size = inputs.shape

    if embedding_size % 2 == 1:
        raise ValueError(
            'Embedding sizes must be even if using positional encodings.')

    # Generate a sequence of positions and frequencies.
    pos = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32)
    freqs = torch.arange(0, embedding_size, 2, dtype=torch.float32)
    inverse_freqs = 1.0 / (max_time ** (freqs / embedding_size))

    # We combine [seq_len] and [emb_size / 2] to [seq_len, emb_size / 2].
    pos_emb = torch.einsum('i,j->ij', pos, inverse_freqs)

    # Concat sines and cosines and return.
    pos_emb = torch.concatenate([torch.sin(pos_emb), torch.cos(pos_emb)], -1)

    return pos_emb


class TransformerEncoderProjection(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderProjection, self).__init__()
        self.transformer = TransformerEncoder(TransformerEncoderLayer(d_model=config.h_dim,
                                                                      nhead=config.n_heads,
                                                                      dim_feedforward=config.widening_factor * config.h_dim,
                                                                      batch_first=True), num_layers=config.n_blocks)
        self.proj_head = nn.Linear(config.h_dim, config.num_classes)

    def forward(self, x):
        h = x
        mask = Transformer.generate_square_subsequent_mask(h.shape[1])
        # convert mask to boolean (-inf should go to false and 0 to True)
        mask = mask == 0
        h = self.transformer(h, is_causal=True, mask=mask)
        pred = self.proj_head(h)
        return pred[:, -1, :]  # Select the output corresponding to the last stimulus (query stimulus)


class MyTransformerWithoutEmbedding(MyTransformer):
    def __init__(self, config):
        super(MyTransformerWithoutEmbedding, self).__init__(config=config)

    def forward(self, x):
        h = x
        for index, block in enumerate(self.blocks):
            h = block(h, index=index)  # Now you pass the index to each block's forward method
        pred = self.proj_head(h)
        return pred[:, -1, :]

if __name__ == '__main__':
    # Define hyperparameters
    one_hot_pos = True
    mytransformer = True
    n_epochs = 50000
    P = 64  # number of possible positions
    D = 64  # stimulus dimension
    K = 2 ** 9  # number of classes (not to be confused with number of labels. multiple classes can have the same label)
    L = 32  # number of labels
    h_dim = P + D
    mlp_dim = 128
    n_heads = 1  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.
    alpha = 0.  # zipf parameter
    burstiness = 4  # burstiness
    seq_len = 16
    n_blocks = 2

    config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=True, n_blocks=n_blocks, n_heads=n_heads, batch_size=1,
                               max_T=P, num_classes=L, include_mlp=[True] * n_blocks, layer_norm=False, mlp_dim=mlp_dim,
                               drop_p=0., within_class_var=.75, alpha=alpha)

    if mytransformer:
        model = MyTransformerWithoutEmbedding(config)
    else:
        model = TransformerEncoderProjection(config)
    # model = MyTransformerWithoutEmbedding(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    pos_encoding = 'one_hot' if one_hot_pos else 'sinusoidal'
    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb.init(project="RulesExemplars",
                   name=f'KimsDebugger-{pos_encoding}PosEncoding-MyTransformer={mytransformer}')

    model.train()
    for i in range(n_epochs):
        optimizer.zero_grad()
        x = torch.randint(low=0, high=config.num_classes - 1, size=(config.batch_size, seq_len))
        one_hot_x = F.one_hot(x, num_classes=D).float()

        pos_to_predict = seq_len - 1  # this one should always be in the sequence even with random start position

        # add position encoding and randomize start position for each sequence
        start_pos = np.random.choice(seq_len - 1, size=config.batch_size)  # randomly choose a starting position
        positions = [torch.arange(start, start + seq_len) for start in start_pos]
        if one_hot_pos:
            pos_embeddings = torch.stack(
                [F.one_hot(pos, num_classes=P).float() for pos in positions])
        else:
            pos_embeddings = torch.stack(
                [_create_positional_encodings(one_hot_x, start_pos=s, max_time=seq_len)
                 for s in start_pos])
        # we want to predict position pos_to_predict, which might occur at random actual position in the sequence because
        # of the random start position.
        position_in_sequence = np.where(positions[0] == pos_to_predict)[0][0]
        true_label_at_pos = x[0, position_in_sequence]

        x = torch.cat([one_hot_x, pos_embeddings], dim=-1)
        y = true_label_at_pos

        if i == n_epochs - 2:
            print(f'x: {x}')
            print(f'one_hot_x: {one_hot_x}')
            print(f'y: {y}')

        pred = model(x)
        loss = F.cross_entropy(pred, torch.tensor([y]))
        loss.backward()
        optimizer.step()

        if config.log_to_wandb:
            wandb.log({'loss': loss.item()})



