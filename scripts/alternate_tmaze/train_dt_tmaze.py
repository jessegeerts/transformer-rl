from prepare_trajectory_data import TrajectoryDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from environments.mazes import AltTmaze
import torch
import pickle
from decision_transformer import DecisionTransformer
import matplotlib.pyplot as plt
import numpy as np
import random

from utils import evaluate_on_env
from definitions import model_save_dir, ROOT_FOLDER
import os
from datetime import datetime
import wandb

torch.manual_seed(1)

traj_dir = os.path.join(ROOT_FOLDER, 'trajectories', 'Tmaze')

torch.set_default_dtype(torch.float32)

# set some hyperparameters
n_training_iters = 30
n_updates_per_iter = 100

# model hyperparameters
n_blocks = 1
embed_dim = 32
context_len = 179
n_heads = 4
dropout_p = 0.1

# training hyperparameters
batch_size = 1              # training batch size
lr = 1e-2                   # learning rate
wt_decay = 1e-4             # weight decay
warmup_steps = 10000        # warmup steps for lr scheduler

# eval hyperparameters
rtg_target = 9.
rtg_scale = 1.0
num_eval_episodes = 10
max_ep_len = 180
render = False

map_fn = "map4.txt"
env = AltTmaze(render_mode='human')

# prepare data
# load trajectories
trajectories = pickle.load(open(os.path.join(traj_dir, 'trajectories_alt_tmaze_100.pkl'), 'rb'))
dataset = TrajectoryDataset(trajectories, context_len=context_len, rtg_scale=1.0)
traj_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

data_iter = iter(traj_data_loader)
# each item from data_iter is a list of the form:
# timesteps, states, actions, returns_to_go, traj_mask

state_dim = env.n_states
act_dim = 4

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
rtg_loss_func = nn.MSELoss()

eval_avg_rewards = []
log_action_losses = []
log_state_losses = []
log_rtg_losses = []

for train_i in range(n_training_iters):

    model.train()
    for _ in range(n_updates_per_iter):
        # sample a trajectory
        try:
            timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)
        except:
            data_iter = iter(traj_data_loader)
            timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)


        optimizer.zero_grad()

        timesteps = timesteps.type(torch.int32)  # B x T
        states = states.type(torch.int)  # B x T x state_dim
        actions = actions.type(torch.int32)  # B x T x act_dim
        returns_to_go = returns_to_go.type(torch.float32).unsqueeze(-1)  # B x T x 1
        traj_mask = traj_mask.type(torch.int32)  # B x T

        action_target = torch.clone(actions).detach().to(torch.int64)  # B x T
        state_target = torch.clone(states).detach().to(torch.int64)  # B x T x state_dim

        state_preds, action_preds, return_preds = model.forward(
            timesteps=timesteps,
            states=states,
            actions=actions,
            returns_to_go=returns_to_go
        )

        # calculate loss
        # only consider non-padded elements
        def calculate_loss(predictions, targets, mask, loss_func, dim):
            predictions = predictions.view(-1, dim)[mask.view(-1) > 0]
            targets = targets.view(-1).squeeze(0)[mask.view(-1) > 0]
            loss = loss_func(predictions, targets)
            return loss


        # Now call it for each data set
        action_loss = calculate_loss(action_preds, action_target, traj_mask, loss_func, act_dim)
        state_loss = calculate_loss(state_preds, state_target, traj_mask, loss_func, state_dim)
        rtg_loss = calculate_loss(return_preds, returns_to_go, traj_mask, rtg_loss_func, 1)  # rtg has dimension 1

        loss = action_loss + state_loss + rtg_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        log_action_losses.append(action_loss.detach().cpu().item())
        log_state_losses.append(state_loss.detach().cpu().item())
        log_rtg_losses.append(rtg_loss.detach().cpu().item())

    # evaluate action accuracy
    results = evaluate_on_env(model, context_len, env, rtg_target, rtg_scale, num_eval_episodes, render=render,
                              max_ep_len=max_ep_len)

    eval_avg_reward = results['eval/avg_reward']
    eval_avg_ep_len = results['eval/avg_ep_len']

    mean_action_loss = np.mean(log_action_losses)
    mean_state_loss = np.mean(log_state_losses)
    mean_rtg_loss = np.mean(log_rtg_losses)

    eval_avg_rewards.append(eval_avg_reward)

    print('Training iteration: {}'.format(train_i))
    print('Mean action loss: {:.4f}'.format(mean_action_loss))
    print('Mean state loss: {:.4f}'.format(mean_state_loss))
    print('Mean rtg loss: {:.4f}'.format(mean_rtg_loss))
    print('Eval avg reward: {:.4f}'.format(eval_avg_reward))


print("=" * 60)
print("finished training!")
print("=" * 60)

plt.plot(eval_avg_rewards)
plt.title('Eval Avg Reward')
plt.show()

plt.plot(log_state_losses)
plt.plot(log_action_losses)
plt.plot(log_rtg_losses)
plt.title('Losses')
plt.legend(['state', 'action', 'rtg'])
plt.show()

torch.save(model.state_dict(), os.path.join(model_save_dir, 'TmazeTransformer_{}_{}_{}.pt'.format(
           datetime.now().month,
           datetime.now().day,
           datetime.now().hour)))
