from environments.gridworld import GridWorld
import random
import numpy as np


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


def collect_trajectories_altmaze(env, num_trials, error_rate, render=False):
    """Collects simulated trajectories through the maze, alternating between left and right way around unless an error is made.

    Store trajectores in list of dictionaries with keys 'observations', 'actions', 'rewards', and values are numpy arrays.
    """
    directions = [right_way_around, left_way_around]
    direction_index = 0  # start with the first direction in the list
    trajectories = []
    n_laps_per_trajectory = 10

    for _ in range(num_trials):
        env.reset()

        states, actions, rewards = [], [], []
        for lap in range(n_laps_per_trajectory):
            # occasionally make an error
            if random.random() < error_rate:
                # repeat the last direction
                direction = directions[direction_index]
            else:
                # alternate direction
                direction = directions[direction_index]
                direction_index = (direction_index + 1) % 2  # toggle between 0 and 1

            lap_states, lap_actions, lap_rewards, _ = direction(env, render)
            states += lap_states
            actions += lap_actions
            rewards += lap_rewards
        trajectories.append({'observations': np.array(states), 'actions': np.array(actions), 'rewards': np.array(rewards)})
    return trajectories


class CuedTmaze(GridWorld):
    def __init__(self, max_steps=200, **kwargs):
        super().__init__(file_name='cued_t_maze', **kwargs)
        self.max_steps = max_steps

        self.start_states = [48, 50]
        self.rewarded_goal = None
        self.unrewarded_goal = None
        self.reset()

    def reset(self, start_state=None):
        self.done = False
        if start_state is not None:
            self.start = start_state
        else:
            self.start = random.choice(self.start_states)
        self.state = self.start
        if self.start == 48:
            self.rewarded_goal = self.goals[0]
            self.unrewarded_goal = self.goals[1]
        elif self.start == 50:
            self.rewarded_goal = self.goals[1]
            self.unrewarded_goal = self.goals[0]
        self.step_count = 0
        return self.state

    def get_reward(self, new_state):
        if new_state in self.goals:
            self.done = True
            if new_state == self.rewarded_goal:
                return 1 - 0.9 * (self.step_count / self.max_steps)
            else:
                return -1
        elif new_state in self.bombs:
            return self.bomb_reward
        elif new_state == self.state:
            return self.bump_reward
        return self.move_reward



