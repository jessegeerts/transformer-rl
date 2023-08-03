import torch
from torch.nn import functional as F
import numpy as np
import random


def evaluate_on_env_append(model, context_len, env, rtg_target, rtg_scale, num_eval_episodes=10, max_ep_len=1000, discrete=True, render=False):
    """Evaluate a model on an environment.

    """
    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.n

    # same as timesteps used for training the transformer
    timesteps = torch.arange(start=0, end=max_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1)

    model.eval()

    with torch.no_grad():
        attention_weights = []
        for ep in range(num_eval_episodes):

            actions = torch.zeros((eval_batch_size, max_ep_len), dtype=torch.int32)
            if discrete:
                states = torch.zeros((eval_batch_size, max_ep_len), dtype=torch.int32)
            else:
                states = torch.zeros((eval_batch_size, max_ep_len, state_dim), dtype=torch.int32)
            rewards_to_go = torch.zeros((eval_batch_size, max_ep_len, 1), dtype=torch.float32)

            # add same mask as used for training
            states[:] = env.n_states
            actions[:] = env.n_actions

            running_state = env.reset()

            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            for t in range(max_ep_len):

                total_timesteps += 1

                if discrete:
                    states[0, t] = running_state
                else:
                    states[0, t] = F.one_hot(torch.tensor(running_state), num_classes=state_dim).to(torch.float32)

                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                input_timesteps = timesteps[:, max(0, t - context_len + 1):t + 1]
                input_states = states[:, max(0, t - context_len + 1):t + 1]
                input_actions = actions[:, max(0, t - context_len + 1):t + 1]
                input_rtg = rewards_to_go[:, max(0, t - context_len + 1):t + 1]

                st_preds, act_preds, rtg_preds, all_weights, mlp_activations = model.forward(input_timesteps,
                                                                              input_states,
                                                                              input_actions,
                                                                              input_rtg)

                act_preds = act_preds[0, min(t, context_len-1)].detach()  # Get the last action prediction

                # softmax_act = torch.multinomial(act_preds[:-1].softmax(-1), 1).item()
                act = torch.argmax(act_preds[:-1]).item()
                running_state, running_reward, done, _ = env.step(act)

                if render:
                    env.render()

                actions[0, t] = act

                total_reward += running_reward

                if done or t == max_ep_len - 1:
                    stacked_weights = torch.stack(all_weights, dim=0)
                    stacked_weights = torch.mean(stacked_weights, dim=1)
                    attention_weights.append(stacked_weights.cpu().numpy())
                    break

    results['eval/avg_reward'] = total_reward / num_eval_episodes
    results['eval/avg_ep_len'] = total_timesteps / num_eval_episodes
    results['eval/attention_weights'] = np.mean(attention_weights, axis=0)

    return results


def per_token_loss(data_iter, model, context_len, env, n_examples, token_ids):
    """Calculate per token loss for each example in the dataset, following the procedure in Olah et al. (2021).
    Randomly sample a token from each example and calculate the loss for that token.

    https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#results-in-context-score

    TODO: implement this
    """
    for timesteps, states, actions, returns_to_go, traj_mask in data_iter:
        # randomly sample a token from each example
        pass


def calculate_loss(predictions, targets, mask, loss_func, dim, mask_value=np.nan):
    valid = (mask.view(-1) > 0) & (targets.view(-1) != mask_value)
    predictions = predictions.view(-1, dim)[valid].squeeze()
    targets = targets.view(-1)[valid]
    if len(targets) == 0:  # no valid targets, return 0 loss
        return torch.tensor(0.0, requires_grad=True)
    loss = loss_func(predictions, targets.squeeze())
    return loss


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def post_process(state_preds, action_preds):
    """Post process the logits to remove the dummy action and state"""
    state_preds = state_preds[:, :, :-1]
    action_preds = action_preds[:, :, :-1]
    return state_preds, action_preds


def mask_data(data, mask_value, mask_prob=0.1):
    mask = torch.rand(data.shape) < mask_prob
    masked_data = data.masked_fill(mask, mask_value)
    return masked_data
