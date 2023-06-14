from prepare_trajectory_data import TrajectoryDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from environments.gridworld import GridWorld
import torch
import pickle
from decision_transformer import DecisionTransformer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns

from utils import evaluate_on_env
from definitions import model_save_dir, ROOT_FOLDER
import os
from datetime import datetime

# set some hyperparameters
n_training_iters = 100
n_updates_per_iter = 10

# model hyperparameters
n_blocks = 1
embed_dim = 32
context_len = 1
n_heads = 4
dropout_p = 0.1

# training hyperparameters
batch_size = 1  # training batch size
lr = 1e-2  # learning rate
wt_decay = 1e-4  # weight decay
warmup_steps = 10000  # warmup steps for lr scheduler

state_dim = 19
n_time_steps = 10
act_dim = 2

timesteps = torch.tensor(range(n_time_steps)).unsqueeze(0)  # B x T
states_plus = torch.tensor(range(n_time_steps)).unsqueeze(0).flip(-1)  # B x T x state_dim
states_minus = torch.tensor(range(9, n_time_steps + 9)).unsqueeze(0)
actions_plus = torch.ones_like(states_plus).unsqueeze(0)  # B x T x act_dim
actions_minus = torch.zeros_like(states_minus).unsqueeze(0)

returns_to_go_plus = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]).unsqueeze(0).to(torch.float32)  # B x T
returns_to_go_minus = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0]).unsqueeze(0).to(torch.float32)  # B x T

# traj_mask = torch.ones_like(states).unsqueeze(0) # B x T

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

loss_func = nn.CrossEntropyLoss()

mean_action_losses = []
mean_state_losses = []

model.train()
for train_i in tqdm(range(n_training_iters)):
    log_action_losses = []
    log_state_losses = []

    for update_i in range(n_updates_per_iter):
        t = timesteps[:, update_i].unsqueeze(0)

        if np.random.rand() > .5:
            s = states_plus[:, update_i].unsqueeze(0)
            a = actions_plus[:, :, update_i]
            r = returns_to_go_plus[:, update_i].unsqueeze(0)
        else:
            s = states_minus[:, update_i].unsqueeze(0)
            a = actions_minus[:, :, update_i]
            r = returns_to_go_minus[:, update_i].unsqueeze(0)

        optimizer.zero_grad()

        action_target = torch.clone(a).detach().to(torch.int64)  # B x T
        state_target = torch.clone(s).detach().to(torch.int64)  # B x T x state_dim

        state_preds, action_preds, return_preds = model.forward(
            timesteps=t,
            states=s,
            actions=a,
            returns_to_go=r
        )

        action_preds = action_preds.view(-1, act_dim)
        action_target = action_target.squeeze(0)

        action_loss = loss_func(action_preds, action_target)

        state_preds = state_preds.view(-1, state_dim)
        state_target = state_target.squeeze(0)

        state_loss = loss_func(state_preds, state_target)

        loss = action_loss + state_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        log_action_losses.append(action_loss.detach().cpu().item())
        log_state_losses.append(state_loss.detach().cpu().item())

    mean_action_losses.append(np.mean(log_action_losses))
    mean_state_losses.append(np.mean(log_state_losses))

# plot losses
plt.plot(mean_action_losses)
plt.plot(mean_state_losses)
plt.legend(['action loss', 'state loss'])
plt.show()

# evaluate model
ap = []
sp = []
best_actions = []

for rtg_target in [1., -1.]:

    if rtg_target == 1.:
        best_action = 1
    elif rtg_target == -1.:
        best_action = 0

    # set starting state and RTG and predict best action
    t = torch.tensor([[0]])  # B x T
    s = torch.tensor([[9]])  # B x T x state_dim
    a = torch.tensor([[0]])  # B x T x act_dim
    r = torch.tensor([[rtg_target]])  # B x T

    for idx in range(10):
        state_preds, action_preds, return_preds = model.forward(
            timesteps=t,
            states=s,
            actions=a,
            returns_to_go=r
        )

        ap.append(torch.multinomial(action_preds.detach().softmax(-1).squeeze(), 1).item())
        sp.append(torch.multinomial(state_preds.detach().softmax(-1).squeeze(), 1).item())
        best_actions.append(best_action)

cf_matrix = confusion_matrix(best_actions, ap, labels=[0, 1])
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in [0, 1]],
                     columns=[i for i in [0, 1]])
plt.figure(figsize=(12, 7))
sns.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix')
plt.show()
