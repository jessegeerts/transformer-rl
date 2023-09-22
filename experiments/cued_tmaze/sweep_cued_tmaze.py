from prepare_trajectory_data import TrajectoryDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from environments.mazes import CuedTmaze
import torch
import pickle
from decision_transformer import DecisionTransformer
import matplotlib.pyplot as plt
import numpy as np

from utils import evaluate_on_env, mask_data
from definitions import model_save_dir, ROOT_FOLDER
import os
import wandb


sweep_config = {
    'method': 'bayes', # can be 'grid', 'random' or 'bayes'
    'metric': {
      'name': 'eval_avg_reward',
      'goal': 'maximize'
    },
    'parameters': {
        'n_blocks': {
            'values': [1, 2]
        },
        'embed_dim': {
            'values': [32, 64]
        },
        'context_len': {
            'values': [8, 10]
        },
        'n_heads': {
            'values': [1, 2, 4, 8]
        },
        'dropout_p': {
            'values': [0.1, 0.2]
        },
        'batch_size': {
            'values': [1, 4]
        },
        'lr': {
            'values': [1e-3, 1e-2]
        },
        'wt_decay': {
            'values': [1e-5, 1e-4]
        },
        'warmup_steps': {
            'values': [5000, 10000]
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="AlternatingTmaze")

wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
# Initialize wandb

torch.manual_seed(2)

traj_dir = os.path.join(ROOT_FOLDER, 'trajectories', 'CuedTmaze')
torch.set_default_dtype(torch.float32)

# set some training hyperparameters
n_training_iters = 30
n_updates_per_iter = 100

# eval hyperparameters
rtg_target = .9
rtg_scale = 1.0
num_eval_episodes = 10
max_ep_len = 180
render = False

env = CuedTmaze(render_mode='human', terminal_reward=1.0, move_reward=0.0, bump_reward=0., bomb_reward=-1.0)


# train decision transformer

def train():

    with wandb.init() as run:
        # prepare data
        # load trajectories
        trajectories = pickle.load(open(os.path.join(traj_dir, 'cued_tmaze_trajectories.pkl'), 'rb'))
        dataset = TrajectoryDataset(trajectories, context_len=wandb.config.context_len, rtg_scale=1.0, random_truncation=True)
        traj_data_loader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=False, pin_memory=True, drop_last=True)

        action_mask_value = env.n_actions
        state_mask_value = env.n_states

        data_iter = iter(traj_data_loader)
        # each item from data_iter is a list of the form:
        # timesteps, states, actions, returns_to_go, traj_mask

        state_dim = env.n_states
        act_dim = 4

        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            n_blocks=wandb.config.n_blocks,
            h_dim=wandb.config.embed_dim,
            context_len=wandb.config.context_len,
            n_heads=wandb.config.n_heads,
            drop_p=wandb.config.dropout_p,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=wandb.config.lr,
            weight_decay=wandb.config.wt_decay
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / wandb.config.warmup_steps, 1)
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

                # mask some data
                masked_actions = mask_data(actions, action_mask_value)
                masked_states = mask_data(states, state_mask_value)

                # create targets
                action_target = torch.clone(masked_actions).detach().to(torch.int64)  # B x T
                state_target = torch.clone(masked_states).detach().to(torch.int64)  # B x T x state_dim

                # forward pass
                state_preds, action_preds, return_preds, weights = model.forward(
                    timesteps=timesteps,
                    states=masked_states,
                    actions=masked_actions,
                    returns_to_go=returns_to_go
                )

                # calculate loss
                # only consider non-padded elements
                def calculate_loss(predictions, targets, mask, loss_func, dim, mask_value=np.nan):
                    valid = (mask.view(-1) > 0) & (targets.view(-1) != mask_value)
                    predictions = predictions.view(-1, dim)[valid].squeeze()
                    targets = targets.view(-1)[valid]
                    loss = loss_func(predictions, targets)
                    return loss

                # Now call it for each data set
                action_loss = calculate_loss(action_preds, action_target, traj_mask, loss_func, act_dim + 1,
                                             mask_value=action_mask_value)
                state_loss = calculate_loss(state_preds, state_target, traj_mask, loss_func, state_dim + 1,
                                            mask_value=state_mask_value)
                rtg_loss = calculate_loss(return_preds, returns_to_go.unsqueeze(-1), traj_mask, rtg_loss_func,
                                          1)  # rtg has dimension 1

                loss = action_loss + state_loss + rtg_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()

                log_action_losses.append(action_loss.detach().cpu().item())
                log_state_losses.append(state_loss.detach().cpu().item())
                log_rtg_losses.append(rtg_loss.detach().cpu().item())

                # Log losses to wandb
                wandb.log({
                    "Action Loss": action_loss.detach().cpu().item(),
                    "State Loss": state_loss.detach().cpu().item(),
                    "RTG Loss": rtg_loss.detach().cpu().item()
                }, step=train_i * n_updates_per_iter + _)

            # evaluate action accuracy
            results = evaluate_on_env(model, wandb.config.context_len, env, rtg_target, rtg_scale, num_eval_episodes, render=render,
                                      max_ep_len=max_ep_len)

            eval_avg_reward = results['eval/avg_reward']
            eval_avg_ep_len = results['eval/avg_ep_len']

            mean_action_loss = np.mean(log_action_losses)
            mean_state_loss = np.mean(log_state_losses)
            mean_rtg_loss = np.mean(log_rtg_losses)

            eval_avg_rewards.append(eval_avg_reward)
            wandb.log({"eval_avg_reward": eval_avg_reward})

            print('Training iteration: {}'.format(train_i))
            print('Mean action loss: {:.4f}'.format(mean_action_loss))
            print('Mean state loss: {:.4f}'.format(mean_state_loss))
            print('Mean rtg loss: {:.4f}'.format(mean_rtg_loss))
            print('Eval avg reward: {:.4f}'.format(eval_avg_reward))
            print("-" * 60)

wandb.agent(sweep_id, train)

# TODO: perhaps log the attention weights to wandb



