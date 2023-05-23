import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

import numpy as np
import os

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridWorld(gym.Env):
    def __init__(self, file_name="map3.txt", fail_rate=0.0, terminal_reward=1.0, move_reward=0.0, bump_reward=-0.5,
                 bomb_reward=-1.0):
        self.n = None
        self.m = None
        self.bombs = []
        self.walls = []
        self.goals = []
        self.start = None
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(this_file_path, file_name)
        with open(file_name, "r") as f:
            for i, row in enumerate(f):
                row = row.rstrip('\r\n')
                if self.n is not None and len(row) != self.n:
                    raise ValueError("Map's rows are not of the same dimension...")
                self.n = len(row)
                for j, col in enumerate(row):
                    if col == "x" and self.start is None:
                        self.start = self.n * i + j
                    elif col == "x" and self.start is not None:
                        raise ValueError("There is more than one starting position in the map...")
                    elif col == "G":
                        self.goals.append(self.n * i + j)
                    elif col == "B":
                        self.bombs.append(self.n * i + j)
                    elif col == "1":
                        self.walls.append(self.n * i + j)
            self.m = i + 1
        if len(self.goals) == 0:
            raise ValueError("At least one goal needs to be specified...")
        self.n_states = self.n * self.m
        self.n_actions = 4
        self.fail_rate = fail_rate
        self.state = self.start
        self.terminal_reward = terminal_reward
        self.move_reward = move_reward
        self.bump_reward = bump_reward
        self.bomb_reward = bomb_reward
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)
        self.done = False

    def step(self, action):
        assert self.action_space.contains(action)
        if self.state in self.goals or np.random.rand() < self.fail_rate:
            self.done = True
            return self.state, 0.0, self.done, None
        elif self.state in self.bombs:
            self.done = True
            return self.state, 0.0, self.done, None
        else:
            new_state = self.take_action(action)
            reward = self.get_reward(new_state)
            self.state = new_state
            if self.state in self.goals or self.state in self.bombs:
                self.done = True
            return self.state, reward, self.done, None

    def reset(self):
        self.done = False
        self.state = self.start
        return self.state

    def render(self, mode='human', close=False):
        pass

    def take_action(self, action):
        row = self.state // self.n
        col = self.state % self.n
        if action == DOWN and (row + 1) * self.n + col not in self.walls:
            row = min(row + 1, self.m - 1)
        elif action == UP and (row - 1) * self.n + col not in self.walls:
            row = max(0, row - 1)
        elif action == RIGHT and row * self.n + col + 1 not in self.walls:
            col = min(col + 1, self.n - 1)
        elif action == LEFT and row * self.n + col - 1 not in self.walls:
            col = max(0, col - 1)
        new_state = row * self.n + col
        return new_state

    def get_reward(self, new_state):
        if new_state in self.goals:
            self.done = True
            return self.terminal_reward
        elif new_state in self.bombs:
            return self.bomb_reward
        elif new_state == self.state:
            return self.bump_reward
        return self.move_reward


if __name__ == '__main__':
    env = GridWorld(terminal_reward=1.0, move_reward=-1.0)
    env.reset()
    while not env.done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(state, reward, done)
