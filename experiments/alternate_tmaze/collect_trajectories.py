from definitions import ROOT_FOLDER
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from environments.mazes import AltTmaze, collect_trajectories_altmaze


n_epochs = 100

data_dir = os.path.join(ROOT_FOLDER, 'trajectories', 'Tmaze')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
fn = 'trajectories_alt_tmaze_{}.pkl'.format(n_epochs)

env = AltTmaze(render_mode='human')
env.reset()
# env.render()
error_rate = 0  # 20% chance of making an error
trajectories = collect_trajectories_altmaze(env, n_epochs, error_rate, render=False)
env.close()

print('=' * 50)
print('Saved {} trajectories to {}'.format(n_epochs, os.path.join(data_dir, fn)))
print('Average reward: {}'.format(np.mean([np.sum(traj['rewards']) for traj in trajectories])))
print('=' * 50)

plt.hist([np.sum(traj['rewards']) for traj in trajectories])
plt.title('Histogram of total rewards')
plt.show()

plt.title('Histogram of trajectory lengths')
plt.hist([len(traj['rewards']) for traj in trajectories])
plt.show()

# save trajectories to pickle
with open(os.path.join(data_dir, fn), 'wb') as f:
    pickle.dump(trajectories, f)
