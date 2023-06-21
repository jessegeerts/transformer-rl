from environments.gridworld import GridWorld
import random


class AltTmaze(GridWorld):
    """Grid world environment implementing the alternating T-maze.

    Agent gets rewarded for taking the opposite path to previous trajectory.
    """
    def __init__(self, max_steps=200, **kwargs):
        super().__init__(file_name='alt_t_maze.txt', **kwargs)
        self._prev_trajectory = []
        self.max_steps = max_steps

    def get_reward(self, new_state):
        """Reward is 1 if agent takes the opposite path to previous trajectory, 0 otherwise."""
        reward = 0
        if new_state in self.goals:
            if new_state not in self._prev_trajectory:
                reward = 1
            self._prev_trajectory = []
        self._prev_trajectory.append(new_state)
        return reward

    def step(self, action):
        assert self.action_space.contains(action)
        new_state = self.take_action(action)
        reward = self.get_reward(new_state)
        self.state = new_state
        if self.step_count >= self.max_steps:
            self.done = True
        self.step_count += 1
        return self.state, reward, self.done, None


def right_way_around(env, render=False):
    actions = [0, 1, 2, 3]
    num_steps = [5, 4, 5, 4]  # Number of steps for each action
    states = []
    rewards = []
    action_seq = []
    total_reward = 0

    for action, num_step in zip(actions, num_steps):
        for _ in range(num_step):
            s, r, done, _ = env.step(action)
            total_reward += r
            states.append(s)
            rewards.append(r)
            action_seq.append(action)
            if render:
                env.render()

    return states, action_seq, rewards, total_reward

def left_way_around(env, render=False):
    actions = [0, 3, 2, 1]
    num_steps = [5, 4, 5, 4]  # Number of steps for each action
    states = []
    rewards = []
    action_seq = []
    total_reward = 0

    for action, num_step in zip(actions, num_steps):
        for _ in range(num_step):
            s, r, done, _ = env.step(action)
            total_reward += r
            states.append(s)
            rewards.append(r)
            action_seq.append(action)
            if render:
                env.render()

    return states, action_seq, rewards, total_reward


def collect_trajectories(env, num_trials, error_rate, render=False):
    """Collects simulated trajectories through the maze, alternating between left and right way around unless an error is made."""
    directions = [right_way_around, left_way_around]
    direction_index = 0  # start with the first direction in the list
    trajectories = []

    for _ in range(num_trials):
        # occasionally make an error
        if random.random() < error_rate:
            # repeat the last direction
            direction = directions[direction_index]
        else:
            # alternate direction
            direction = directions[direction_index]
            direction_index = (direction_index + 1) % 2  # toggle between 0 and 1

        states, actions, rewards, total_reward = direction(env, render)

        trajectories.append({'observations': states, 'actions': actions, 'rewards': rewards})

    return trajectories


if __name__ == '__main__':
    from definitions import ROOT_FOLDER
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle

    n_epochs = 100

    data_dir = os.path.join(ROOT_FOLDER, 'trajectories')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    fn = 'trajectories_alt_tmaze_{}.pkl'.format(n_epochs)

    env = AltTmaze(render_mode='human')
    env.reset()
    env.render()
    error_rate = 0.2  # 20% chance of making an error
    trajectories = collect_trajectories(env, n_epochs, error_rate, render=False)
    env.close()

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
