import torch
import numpy
import matplotlib.pyplot as plt
from definitions import model_save_dir
import os
from decision_transformer import DecisionTransformer
from utils import evaluate_on_env
from environments.gridworld import GridWorld


context_len = 10


map_fn = "map3.txt"
env = GridWorld(file_name=map_fn, terminal_reward=0.0, move_reward=-1.0, bump_reward=-1., bomb_reward=0.)

# model hyperparameters
n_blocks = 2
embed_dim = 32
context_len = 10
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

model.load_state_dict(torch.load(os.path.join(model_save_dir, 'GridWorldTransformer_5_22_13.pt')))
model.eval()


rtg_targets = [-60., -50., -40., -30., -20., -10.]


for target in rtg_targets:
    results = evaluate_on_env(model, context_len, env, target, 1., num_eval_episodes=10, max_ep_len=1000)
    print(f'RTG target: {target}, avg reward: {results["avg_reward"]}, avg timesteps: {results["avg_timesteps"]}')
