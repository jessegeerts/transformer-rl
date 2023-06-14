""" train decision transformer on some random walks in a GridWorld environment
"""
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

from utils import evaluate_on_env
from definitions import model_save_dir
import os
from datetime import datetime


torch.set_default_dtype(torch.float32)

# set some hyperparameters
n_training_iters = 20
n_updates_per_iter = 100


# model hyperparameters
n_blocks = 1
embed_dim = 32
context_len = 10
n_heads = 4
dropout_p = 0.1

# training hyperparameters
batch_size = 64             # training batch size
lr = 1e-4                   # learning rate
wt_decay = 1e-4             # weight decay
warmup_steps = 10000        # warmup steps for lr scheduler

# eval hyperparameters
rtg_target = -40.
rtg_scale = 1.0
num_eval_episodes = 10

map_fn = "map2.txt"
env = GridWorld(file_name=map_fn, terminal_reward=0.0, move_reward=-1.0, bump_reward=-1.)

# prepare data
# load trajectories
trajectories = pickle.load(open('trajectories/trajectories_map2_onehot_100_epochs.pkl', 'rb'))
dataset = TrajectoryDataset(trajectories, context_len=10, rtg_scale=1.0)
traj_data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)

data_iter = iter(traj_data_loader)
# each item from data_iter is a list of the form:
# timesteps, states, actions, returns_to_go, traj_mask

state_dim = env.n_states
act_dim = 4  # probably should be 4?

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

loss_func = nn.CrossEntropyLoss()


eval_avg_rewards = []

for train_i in range(n_training_iters):
    log_action_losses = []
    for _ in range(n_updates_per_iter):
        # sample a trajectory
        try:
            timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(traj_data_loader)
            timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

        timesteps = timesteps.type(torch.int32)  # B x T
        states = states.type(torch.float32)  # B x T x state_dim
        actions = actions.type(torch.int32)  # B x T x act_dim
        returns_to_go = returns_to_go.type(torch.float32).unsqueeze(-1)  # B x T x 1
        traj_mask = traj_mask.type(torch.int32)  # B x T
        action_target = F.one_hot(torch.clone(actions).detach().to(torch.int64), num_classes=act_dim)  # B x T x act_dim

        state_preds, action_preds, return_preds = model.forward(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go
        )

        # calculate loss
        # only consider non-padded elements
        action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
        action_target = action_target.view(-1, act_dim)[traj_mask.view(-1, ) > 0].to(torch.float32)

        action_loss = loss_func(action_preds, action_target)

        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        log_action_losses.append(action_loss.detach().cpu().item())

    # evaluate action accuracy
    results = evaluate_on_env(model, context_len, env, rtg_target, rtg_scale, num_eval_episodes)

    eval_avg_reward = results['eval/avg_reward']
    eval_avg_ep_len = results['eval/avg_ep_len']

    mean_action_loss = np.mean(log_action_losses)

    eval_avg_rewards.append(eval_avg_reward)

    print('Training iteration: {}'.format(train_i))
    print('Mean action loss: {:.4f}'.format(mean_action_loss))
    print('Eval avg reward: {:.4f}'.format(eval_avg_reward))


print("=" * 60)
print("finished training!")
print("=" * 60)

plt.plot(eval_avg_rewards)
plt.show()

torch.save(model.state_dict(), os.path.join(model_save_dir, 'GridWorldTransformer_{}_{}_{}.pt'.format(
           datetime.now().month,
           datetime.now().day,
           datetime.now().hour)))


# TODO: what's it learning here? it's trying to predict the next action given previous states and rewards-to-go.
# TODO: make a calibration plot for the model's initial RTG versus the actual rewards it gets