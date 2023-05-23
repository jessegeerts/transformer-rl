"""Collect some trajectories from a random policy and save in right format for TrajectoryDataset"""

import numpy as np
from environments.environments import GridWorld
import pickle
import os
import matplotlib.pyplot as plt


np.random.seed(3)

n_epochs = 200
one_hot = False


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], 'trajectories')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


map_fn = "map4.txt"
env = GridWorld(file_name=map_fn, terminal_reward=1.0, move_reward=0.0, bump_reward=0., bomb_reward=-1.0,
                max_steps=250)

n_states = env.n_states


if one_hot:
    fn = 'trajectories_{}_onehot_{}_epochs.pkl'.format(map_fn.split('.')[0], n_epochs)
else:
    fn = 'trajectories_{}_discrete_{}_epochs.pkl'.format(map_fn.split('.')[0], n_epochs)


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


print('='*50)
print('Saved {} trajectories to {}'.format(n_epochs, os.path.join(data_dir, fn)))
print('Average reward: {}'.format(np.mean([np.sum(traj['rewards']) for traj in trajectories])))
print('='*50)


plt.hist([np.sum(traj['rewards']) for traj in trajectories])
plt.title('Histogram of total rewards')
plt.show()

plt.title('Histogram of trajectory lengths')
plt.hist([len(traj['rewards']) for traj in trajectories])
plt.show()

# save trajectories to pickle
with open(os.path.join(data_dir, fn), 'wb') as f:
    pickle.dump(trajectories, f)
