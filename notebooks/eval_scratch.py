import torch
import numpy
import matplotlib.pyplot as plt
from definitions import model_save_dir
import os
from decision_transformer import DecisionTransformer
from utils import evaluate_on_env_array, evaluate_on_env_append
from environments.mazes import AltTmaze, CuedTmaze

from definitions import model_save_dir, ROOT_FOLDER

model_fn = 'CuedTmazeTransformer_7_31_16_4head.pt'

target = 1.

env = CuedTmaze(render_mode='human')

# model hyperparameters
n_blocks = 1
embed_dim = 32
context_len = 10
n_heads = 4
dropout_p = 0.1
state_dim = env.n_states
act_dim = 4


# load model
model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    n_blocks=n_blocks,
    h_dim=embed_dim,
    context_len=context_len,
    n_heads=n_heads,
    drop_p=dropout_p,
    action_mask_value=env.n_actions,
    state_mask_value=env.n_states
)


model.load_state_dict(torch.load(os.path.join(model_save_dir, model_fn)))
model.eval()

# zeros place holders
max_ep_len = 50

timesteps = torch.arange(start=0, end=max_ep_len, step=1).unsqueeze(0)
actions = torch.zeros((1, max_ep_len), dtype=torch.int32)
states = torch.zeros((1, max_ep_len), dtype=torch.int32)
rewards_to_go = torch.zeros((1, max_ep_len, 1), dtype=torch.float32)

# add same mask as used for training
states[:] = env.n_states
actions[:] = env.n_actions
#rewards_to_go[:] = target

# init episode
running_state = 50  # 48 is the start state, optimal action is 1
env.reset(running_state)

padding_len = 0


def select_inputs(timesteps, states, actions, rewards_to_go, t, padding_len=0):
    if padding_len == 0:
        input_timesteps = timesteps[:, max(0, t - context_len + 1):t + 1]
        input_states = states[:, max(0, t - context_len + 1):t + 1]
        input_actions = actions[:, max(0, t - context_len + 1):t + 1]
        input_rewards_to_go = rewards_to_go[:, max(0, t - context_len + 1):t + 1]

    else:
        input_timesteps = timesteps[:, :padding_len+1]
        input_states = states[:, :padding_len+1]
        input_actions = actions[:, :padding_len+1]
        input_rewards_to_go = rewards_to_go[:, :padding_len+1]
    return input_timesteps, input_states, input_actions, input_rewards_to_go


env.render()
done = False
t = 0
while not done:

    states[:, t] = running_state
    rewards_to_go[:, t] = target

    # select inputs
    input_timesteps, input_states, input_actions, input_rewards_to_go = \
        select_inputs(timesteps, states, actions, rewards_to_go, t, padding_len=padding_len)

    _, act_preds, _, all_weights, mlp_activations = model.forward(input_timesteps,
                                                                  input_states,
                                                                  input_actions,
                                                                  input_rewards_to_go)

    # get action prediction
    if t < context_len:
        act_pred = act_preds[:, t, :].squeeze(0)
    else:
        act_pred = act_preds[:, -1, :].squeeze(0)
    a = torch.argmax(act_pred, dim=-1).item()

    new_state, reward, done, _ = env.step(a)

    probs = torch.softmax(act_pred, dim=-1)

    logits = {a:l for a, l in zip(range(env.n_actions), probs[:-1].detach().numpy())}

    env.render(logits, t)
    actions[:, t] = a
    running_state = new_state
    t += 1
env.close()

