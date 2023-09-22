from prepare_trajectory_data import TrajectoryDataset
from torch.utils.data import DataLoader
from torch import nn
from environments.mazes import CuedTmaze
import torch
import pickle
from decision_transformer import DecisionTransformer
import matplotlib.pyplot as plt
import numpy as np

from utils import calculate_loss, seed_everything, post_process, mask_data, evaluate_on_env_array, evaluate_on_env_append
from definitions import model_save_dir, ROOT_FOLDER
import os
from datetime import datetime
import wandb
from config import TransformerConfig
from logging_utils import log_attention_weights


map_name = 'cued_t_maze'
trajectories_fn = 'cued_tmaze_trajectories.pkl'

# set some hyperparameters
config = TransformerConfig(log_to_wandb=True, save_model=True, n_training_iters=30, n_updates_per_iter=100,
                           random_truncation=False, mask_prob=.1, rtg_target=.9, context_len=10)
# TODO: find out why longer context lengths don't work

# choose evaluate_on_env function
evaluate_on_env = evaluate_on_env_append

if config.log_to_wandb:
    wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
    # Initialize wandb
    wandb.init(project="AlternatingTmaze", name="TransformerCuedMaze")

seed_everything(config.seed)
traj_dir = os.path.join(ROOT_FOLDER, 'trajectories', 'CuedTmaze')
torch.set_default_dtype(torch.float32)

env = CuedTmaze(render_mode='human', file_name='cued_t_maze', terminal_reward=1.0, move_reward=0.0, bump_reward=0.,
                bomb_reward=-1.0, map_name=map_name)

# prepare data
# load trajectories
trajectories = pickle.load(open(os.path.join(traj_dir, trajectories_fn), 'rb'))
dataset = TrajectoryDataset(trajectories, context_len=config.context_len, rtg_scale=1.0, random_truncation=config.random_truncation)
traj_data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, drop_last=True)

action_mask_value = env.n_actions
state_mask_value = env.n_states

data_iter = iter(traj_data_loader)
# each item from data_iter is a list of the form:
# timesteps, states, actions, returns_to_go, traj_mask

state_dim = env.n_states
act_dim = 4

# train decision transformer

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    n_blocks=config.n_blocks,
    h_dim=config.embed_dim,
    context_len=config.context_len,
    n_heads=config.n_heads,
    drop_p=config.dropout_p,
    act_fn=config.act_fn,
    action_mask_value=action_mask_value,
    state_mask_value=state_mask_value
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.wt_decay
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda steps: min((steps + 1) / config.warmup_steps, 1)
)

loss_func = nn.CrossEntropyLoss()
rtg_loss_func = nn.MSELoss()

eval_avg_rewards = []
log_action_losses = []
log_state_losses = []
log_rtg_losses = []

for train_i in range(config.n_training_iters):
    model.train()
    for _ in range(config.n_updates_per_iter):
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

        masked_actions = mask_data(actions, action_mask_value, mask_prob=config.mask_prob)
        masked_states = mask_data(states, state_mask_value, mask_prob=config.mask_prob)

        # create targets
        action_target = torch.clone(masked_actions).detach().to(torch.int64)  # B x T
        state_target = torch.clone(masked_states).detach().to(torch.int64)  # B x T x state_dim

        # forward pass
        state_preds, action_preds, return_preds, weights, mlp_activations = model.forward(
            timesteps=timesteps,
            states=masked_states,
            actions=masked_actions,
            returns_to_go=returns_to_go
        )

        state_preds, action_preds = post_process(state_preds, action_preds)

        # calculate loss
        # only consider non-padded elements
        # Now call it for each data set
        action_loss = calculate_loss(action_preds, action_target, traj_mask, loss_func, act_dim, mask_value=action_mask_value)
        state_loss = calculate_loss(state_preds, state_target, traj_mask, loss_func, state_dim, mask_value=state_mask_value)
        rtg_loss = calculate_loss(return_preds, returns_to_go.unsqueeze(-1), traj_mask, rtg_loss_func, 1)  # rtg has dimension 1

        loss = action_loss + state_loss + rtg_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        log_action_losses.append(action_loss.detach().cpu().item())
        log_state_losses.append(state_loss.detach().cpu().item())
        log_rtg_losses.append(rtg_loss.detach().cpu().item())

        # Log losses to wandb
        if config.log_to_wandb:
            wandb.log({
                "Action Loss": action_loss.detach().cpu().item(),
                "State Loss": state_loss.detach().cpu().item(),
                "RTG Loss": rtg_loss.detach().cpu().item()
            }, step=train_i*config.n_updates_per_iter+_)


    # evaluate action accuracy
    results = evaluate_on_env(model, config.context_len, env, config.rtg_target, config.rtg_scale,
                              config.num_eval_episodes, render=config.render,
                              max_ep_len=config.max_ep_len)


    eval_avg_reward = results['eval/avg_reward']
    eval_avg_ep_len = results['eval/avg_ep_len']

    mean_action_loss = np.mean(log_action_losses)
    mean_state_loss = np.mean(log_state_losses)
    mean_rtg_loss = np.mean(log_rtg_losses)

    eval_avg_rewards.append(eval_avg_reward)
    if config.log_to_wandb:
        wandb.log({"eval_avg_reward": eval_avg_reward})
        log_attention_weights(results['eval/attention_weights'], train_i)

        #if train_i % 10 == 0:
        #    # log ratemaps to wandb
        #    ratemaps_1, ratemaps_2 = compute_ratemaps(model, config.context_len, env, config.rtg_target,
        #                                              config.rtg_scale, num_eval_episodes=100)
        #    n_mlp_units = model.transformer.blocks[0].mlp1[0].out_features

        #    for i in range(math.ceil(n_mlp_units / 32)):
        #        units = list(range(32 * i, min(32 * (i + 1), n_mlp_units)))
        #        fig, axes = plot_tmaze_ratemaps(units, ratemaps_1, ratemaps_2, env)
        #        wandb.log({"ratemaps_units_{}-{}".format(units[0], units[-1]): fig})
        #        plt.close(fig)

    print('Training iteration: {}'.format(train_i))
    print('Mean action loss: {:.4f}'.format(mean_action_loss))
    print('Mean state loss: {:.4f}'.format(mean_state_loss))
    print('Mean rtg loss: {:.4f}'.format(mean_rtg_loss))
    print('Eval avg reward: {:.4f}'.format(eval_avg_reward))
    print('Eval avg ep len: {:.4f}'.format(eval_avg_ep_len))
    print("-" * 60)

print("=" * 60)
print("finished training!")
print("=" * 60)

plt.plot(eval_avg_rewards)
plt.title('Eval Avg Reward')

plt.plot(log_state_losses)
plt.plot(log_action_losses)
plt.plot(log_rtg_losses)
plt.title('Losses')
plt.legend(['state', 'action', 'rtg'])


if config.save_model:
    model_fn = 'CuedTmazeTransformer_{}_{}_{}_{}head.pt'.format(
               datetime.now().month,
               datetime.now().day,
               datetime.now().hour,
               config.n_heads)
    torch.save(model.state_dict(), os.path.join(model_save_dir, model_fn))
    print("Saved model to {}".format(model_fn))
    print("=" * 60)

