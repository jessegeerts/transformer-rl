import torch
from torch.nn import functional as F
import numpy as np


def evaluate_on_env(model, context_len, env, rtg_target, rtg_scale, num_eval_episodes=10, max_ep_len=1000,
                    discrete=True, render=False):
    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.n

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1)

    model.eval()

    with torch.no_grad():
        attention_weights = []
        for _ in range(num_eval_episodes):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_ep_len), dtype=torch.int32)
            if discrete:
                states = torch.zeros((eval_batch_size, max_ep_len), dtype=torch.int32)
            else:
                states = torch.zeros((eval_batch_size, max_ep_len, state_dim), dtype=torch.int32)
            rewards_to_go = torch.zeros((eval_batch_size, max_ep_len, 1), dtype=torch.float32)

            # add same mask as used for training
            states[:] = env.n_states
            actions[:] = env.n_actions

            # init episode
            running_state = env.reset()
            # convert to one-hot
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_ep_len):

                total_timesteps += 1

                # add state in placeholder
                if discrete:
                    states[0, t] = running_state
                else:
                    states[0, t] = F.one_hot(torch.tensor(running_state), num_classes=state_dim).to(
                        torch.float32)  # running_state # torch.from_numpy(running_state)  # .to(device)

                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    _, act_preds, _, all_weights = model.forward(timesteps[:, :context_len],
                                                    states[:, :context_len],
                                                    actions[:, :context_len],
                                                    rewards_to_go[:, :context_len])
                    act_preds = act_preds[0, t].detach()
                else:
                    _, act_preds, _, all_weights = model.forward(timesteps[:, t - context_len + 1:t + 1],
                                                    states[:, t - context_len + 1:t + 1],
                                                    actions[:, t - context_len + 1:t + 1],
                                                    rewards_to_go[:, t - context_len + 1:t + 1])
                    act_preds = act_preds[0, -1].detach()

                softmax_act = torch.multinomial(act_preds[:-1].softmax(-1), 1).item()
                act = torch.argmax(act_preds[:-1]).item()  # TODO: why not use softmax?
                running_state, running_reward, done, _ = env.step(act)

                if render:
                    env.render()
                # add action in placeholder
                actions[0, t] = act

                total_reward += running_reward

                if done:
                    stacked_weights = torch.stack(all_weights, dim=0)
                    # average over batches
                    stacked_weights = torch.mean(stacked_weights, dim=1) # (num_layers, num_heads, seq_len, seq_len)
                    # keep running average of attention weights at last time step
                    attention_weights.append(stacked_weights.cpu().numpy())
                    break

    results['eval/avg_reward'] = total_reward / num_eval_episodes
    results['eval/avg_ep_len'] = total_timesteps / num_eval_episodes
    results['eval/attention_weights'] = np.mean(attention_weights, axis=0)  # average over episodes

    return results
