""" train decision transformer on some random walks in a GridWorld environment
"""
from prepare_trajectory_data import TrajectoryDataset
from torch.utils.data import DataLoader
from environments.environments import GridWorld
import torch
import pickle
from decision_transformer import DecisionTransformer
import matplotlib.pyplot as plt
import numpy as np

# set some hyperparameters
n_blocks = 2
embed_dim = 32
context_len = 10
n_heads = 4
dropout_p = 0.1

batch_size = 64             # training batch size
lr = 1e-4                   # learning rate
wt_decay = 1e-4             # weight decay
warmup_steps = 10000        # warmup steps for lr scheduler


env = GridWorld()

# prepare data
# load trajectories
trajectories = pickle.load(open('trajectories/trajectories_onehot.pkl', 'rb'))
dataset = TrajectoryDataset(trajectories, context_len=10, rtg_scale=1.0)
traj_data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)

data_iter = iter(traj_data_loader)

state_dim = env.n_states
act_dim = 1  # probably should be 4?

# train decision transformer

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    n_blocks=n_blocks,
    h_dim=embed_dim,
    context_len=context_len,
    n_heads=n_heads,
    drop_p=dropout_p,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=wt_decay
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda steps: min((steps + 1) / warmup_steps, 1)
)

