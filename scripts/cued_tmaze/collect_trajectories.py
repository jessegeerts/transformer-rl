import os
from itertools import product

import numpy as np

from definitions import ROOT_FOLDER
from environments.mazes import CuedTmaze


def collect_cue_maze_traj(render=False, map_name='cued_t_maze'):
    env = CuedTmaze(render_mode='human', map_name=map_name)
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


def collect_epsilon_optimal_cue_maze_traj(render=False, map_name='cued_t_maze', epsilon=0.2, n_trajectories=200):
    env = CuedTmaze(render_mode='human', map_name=map_name, terminal_reward=1.0, move_reward=0.0, bump_reward=0.,
                    bomb_reward=-1.0)
    trajectories = []
    for _ in range(n_trajectories):
        for start, goal in product(env.start_states, env.goals):
            env.reset(start)
            state = start

            if render:
                env.render()
            traj = {'observations': [], 'actions': [], 'rewards': []}
            done = False
            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, env.n_actions)
                else:
                    path = env.bfs(state, goal)
                    _, action = path[0]
                traj['observations'].append(state)
                traj['actions'].append(action)
                new_state, reward, done, _ = env.step(action)
                traj['rewards'].append(reward)

                state = new_state
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
    fn2 = 'cued_tmaze_epsilon_optimal_trajectories.pkl'

    trajectories = collect_cue_maze_traj(render=False)

    noisy_trajectories = collect_epsilon_optimal_cue_maze_traj(render=False, map_name='cued_t_maze_2')

    with open(os.path.join(data_dir, fn), 'wb') as f:
        pickle.dump(trajectories, f)

    with open(os.path.join(data_dir, fn2), 'wb') as f:
        pickle.dump(noisy_trajectories, f)


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


    print('=' * 50)
    print('Saved {} trajectories to {}'.format(len(noisy_trajectories), os.path.join(data_dir, fn2)))
    print('Average reward: {}'.format(np.mean([np.sum(traj['rewards']) for traj in noisy_trajectories])))
    print('=' * 50)

    plt.hist([np.sum(traj['rewards']) for traj in noisy_trajectories], bins=100)
    plt.title('Histogram of total rewards')
    plt.show()

    plt.title('Histogram of trajectory lengths')
    plt.hist([len(traj['rewards']) for traj in noisy_trajectories])
    plt.show()
