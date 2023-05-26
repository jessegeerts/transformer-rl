import torch
import numpy
import matplotlib.pyplot as plt
from definitions import model_save_dir
import os
from decision_transformer import DecisionTransformer
from utils import evaluate_on_env
from environments.environments import GridWorld

from definitions import model_save_dir, ROOT_FOLDER

context_len = 5

map_fn = "map4.txt"
env = GridWorld(file_name=map_fn, terminal_reward=1.0, move_reward=0.0, bump_reward=0., bomb_reward=-1.0)

# model hyperparameters
n_blocks = 1
embed_dim = 32
n_heads = 4
dropout_p = 0.1

state_dim = env.n_states
act_dim = 4  # probably should be 4?


model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    n_blocks=n_blocks,
    h_dim=embed_dim,
    context_len=context_len,
    n_heads=n_heads,
    drop_p=dropout_p,
)


model.load_state_dict(torch.load(os.path.join(model_save_dir, 'GridWorldTransformer_5_23_18.pt')))
model.eval()


rtg_targets = [.999, -1]


for target in rtg_targets:
    results = evaluate_on_env(model, context_len, env, target, 1., num_eval_episodes=10, max_ep_len=1000)
    print(f'RTG target: {target}, avg reward: {results["avg_reward"]}, avg timesteps: {results["avg_timesteps"]}')
