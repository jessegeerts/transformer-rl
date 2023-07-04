import torch
import numpy
import matplotlib.pyplot as plt
from definitions import model_save_dir
import os
from decision_transformer import DecisionTransformer
from utils import evaluate_on_env
from environments.mazes import AltTmaze, CuedTmaze

from definitions import model_save_dir, ROOT_FOLDER


model_fn = 'CuedTmazeTransformer_7_4_11.pt'

target = .95

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
)


model.load_state_dict(torch.load(os.path.join(model_save_dir, model_fn)))
model.eval()

results = evaluate_on_env(model, context_len, env, target, 1., num_eval_episodes=10, max_ep_len=180, render=False)
print(f'RTG target: {target}, avg reward: {results["avg_reward"]}, avg timesteps: {results["avg_timesteps"]}')
