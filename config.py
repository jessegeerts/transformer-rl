import os
import torch
from custom_torch import SoLU
from definitions import ROOT_FOLDER


class TransformerConfig:
    def __init__(self, seed=2, n_training_iters=30, n_updates_per_iter=100, n_blocks=1,
                 embed_dim=32, context_len=10, n_heads=4, dropout_p=0.1,
                 act_fn=SoLU(), batch_size=1, mask_prob=0.0, random_truncation=False,
                 lr=1e-2, wt_decay=1e-4, warmup_steps=10000,
                 rtg_target=.95, rtg_scale=1.0, num_eval_episodes=10, max_ep_len=180, render=False,
                 log_to_wandb=True, save_model=True):
        self.seed = seed
        self.traj_dir = os.path.join(ROOT_FOLDER, 'trajectories', 'CuedTmaze')
        self.dtype = torch.float32
        self.n_training_iters = n_training_iters
        self.n_updates_per_iter = n_updates_per_iter
        self.n_blocks = n_blocks
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.act_fn = act_fn
        self.batch_size = batch_size
        self.mask_prob = mask_prob
        self.random_truncation = random_truncation
        self.lr = lr
        self.wt_decay = wt_decay
        self.warmup_steps = warmup_steps
        self.rtg_target = rtg_target
        self.rtg_scale = rtg_scale
        self.num_eval_episodes = num_eval_episodes
        self.max_ep_len = max_ep_len
        self.render = render
        self.log_to_wandb = log_to_wandb
        self.save_model = save_model
