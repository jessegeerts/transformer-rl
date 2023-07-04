import os
from itertools import product

import numpy as np

from definitions import ROOT_FOLDER
from environments.mazes import CuedTmaze


def collect_cue_maze_traj(render=False):
    env = CuedTmaze(render_mode='human')
    trajectories = []
    for start, goal in product(env.start_states, env.goals):
        path = env.bfs(start, goal)
        env.reset(start)
        if render:
            env.render()
        traj = {'observations': [], 'actions': [], 'rewards': []}
        for state, action in path:
            traj['observations'].append(state)
            traj['actions'].append(action)
            new_state, reward, done, _ = env.step(action)
            traj['rewards'].append(reward)
            if render:
                env.render()

        for key, val in traj.items():
            traj[key] = np.array(val)
        trajectories.append(traj)
    env.close()
    return trajectories


if __name__ == '__main__':
    from itertools import product
    import pickle
    import os
    import matplotlib.pyplot as plt

    data_dir = os.path.join(ROOT_FOLDER, 'trajectories', 'CuedTmaze')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fn = 'cued_tmaze_trajectories.pkl'

    trajectories = collect_cue_maze_traj(render=False)

    with open(os.path.join(data_dir, fn), 'wb') as f:
        pickle.dump(trajectories, f)

    print('=' * 50)
    print('Saved {} trajectories to {}'.format(len(trajectories), os.path.join(data_dir, fn)))
    print('Average reward: {}'.format(np.mean([np.sum(traj['rewards']) for traj in trajectories])))
    print('=' * 50)

    plt.hist([np.sum(traj['rewards']) for traj in trajectories])
    plt.title('Histogram of total rewards')
    plt.show()

    plt.title('Histogram of trajectory lengths')
    plt.hist([len(traj['rewards']) for traj in trajectories])
    plt.show()
