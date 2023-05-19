"""Collect some trajectories from a random policy and save in right format for TrajectoryDataset"""

import numpy as np
from environments.environments import GridWorld
import pickle
import os

np.random.seed(3)

n_epochs = 100
one_hot = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], 'trajectories')

if one_hot:
    fn = 'trajectories_onehot.pkl'
else:
    fn = 'trajectories.pkl'


env = GridWorld(terminal_reward=0.0, move_reward=-1.0, bump_reward=-1.)
n_states = env.n_states


def to_onehot(idx):
    onehot = np.zeros(env.n_states)
    onehot[idx] = 1
    return onehot


def state_encoding(state, one_hot=False):
    if one_hot:
        return to_onehot(state)
    else:
        return state


trajectories = []
for epoch in range(n_epochs):
    env.reset()


    action = env.action_space.sample()

    traj = {
        'observations': [state_encoding(env.state, one_hot=one_hot)],
        'actions': [action],
        'rewards': [0.],
    }

    while not env.done:

        state, reward, done, _ = env.step(action)

        action = env.action_space.sample()

        traj['actions'].append(action)
        traj['observations'].append(state_encoding(state, one_hot=one_hot))
        traj['rewards'].append(reward)

    traj['actions'] = np.array(traj['actions'])
    traj['observations'] = np.array(traj['observations'])
    traj['rewards'] = np.array(traj['rewards'])
    trajectories.append(traj)

# save trajectories to pickle
with open(os.path.join(data_dir, fn), 'wb') as f:
    pickle.dump(trajectories, f)

