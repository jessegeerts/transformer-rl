"""Given that the network can use the one-hot position encodings to retrieve the Nth item, now we can try to see if
it can return the label associated to item coming after the occurence of a specific item. For example, the target item
is A, and the network should return the label associated to the item that comes after the first occurence of A.
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

from experiments.iwl_icl_classification.model import Transformer as MyTransformer
from experiments.rules_vs_exemplars.config import TransformerConfig
from experiments.iwl_icl_classification.debugging_icl.kims_debugger import _create_positional_encodings

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


# Define hyperparameters
one_hot_pos = False
n_epochs = 100000
P = 64  # number of possible positions
D = 64  # stimulus dimension
L = 32  # number of labels
h_dim = P + D
mlp_dim = 128
n_heads = 1  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.
alpha = 0.  # zipf parameter
burstiness = 4  # burstiness
seq_len = L
n_blocks = 2

config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=True, n_blocks=n_blocks, n_heads=n_heads, batch_size=1,
                           max_T=P, num_classes=L, include_mlp=[True, True], layer_norm=False, mlp_dim=mlp_dim,
                           drop_p=0., within_class_var=.75, alpha=alpha)


model = TransformerEncoderProjection(config)
# model = MyTransformerWithoutEmbedding(config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

if config.log_to_wandb:
    wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
    wandb.init(project="RulesExemplars",
               name=f'DebugRelativePosition')

model.train()
for i in range(n_epochs):
    match_pos = 'before'  # 'before' or 'after

    token_to_match = 0  # the task will be to return the label of the item that comes after this token in the sequence

    optimizer.zero_grad()
    x = torch.randperm(L)
    # ensure that the last item is not token_to_match because we want to predict the item that comes after it
    # also ensure the first item isn't in case we want to predict the item that comes before it
    while x[-1] == token_to_match or x[0] == token_to_match:
        x = torch.randperm(L)
    x = torch.unsqueeze(x, 0)
    one_hot_x = F.one_hot(x, num_classes=D).float()

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
    position_in_sequence = np.where(x == token_to_match)[1][0]
    assert x[0, position_in_sequence] == token_to_match, 'The token to match should be in the sequence.'

    if match_pos == 'after':
        true_label_at_pos_plus_1 = x[0, position_in_sequence + 1]
    elif match_pos == 'before':
        true_label_at_pos_plus_1 = x[0, position_in_sequence - 1]
    else:
        raise ValueError('match_pos must be either "before" or "after"')

    x = torch.cat([one_hot_x, pos_embeddings], dim=-1)
    y = true_label_at_pos_plus_1

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



