import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import pygame
#from gym.envs.classic_control.rendering import SimpleImageViewer

import numpy as np
import os
from collections import deque


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


# Define the four possible directions for the arrows
UP_ARROW = np.array([0, -1])
DOWN_ARROW = np.array([0, 1])
LEFT_ARROW = np.array([-1, 0])
RIGHT_ARROW = np.array([1, 0])
ARROWS = [UP_ARROW, DOWN_ARROW, LEFT_ARROW, RIGHT_ARROW]

# Function to draw arrow
def draw_arrow(surface, color, pos, direction, scale):
    arrow_size = np.array([0.2, 0.5]) * scale  # This defines the size of the arrow, adjusted for agent size
    pos = np.array(pos)
    points = [
        pos + ((np.array([0.5, 0.5]) + direction * 0.5 - direction * arrow_size) * scale),
        pos + ((np.array([0.5, 0.5]) + direction * 0.5 + direction * arrow_size) * scale),
        pos + ((np.array([0.5, 0.5]) + direction * 0.5 + np.array([-1, 1]) * direction * arrow_size / 2) * scale),
    ]
    pygame.draw.polygon(surface, color, points)


class GridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, file_name="map3.txt", fail_rate=0.0, terminal_reward=1.0, move_reward=0.0, bump_reward=-0.5,
                 bomb_reward=-1.0, max_steps=1000, render_mode=None):
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
        self.max_steps = max_steps
        self.step_count = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = 512  # The size of the PyGame window
        self.render_mode = render_mode
        self.window = None  # will be a reference to the window that we draw to.
        self.clock = None  # used to ensure that the environment is rendered at the correct framerate in human-mode.

    def step(self, action):
        assert self.action_space.contains(action)
        new_state = self.take_action(action)
        reward = self.get_reward(new_state)
        self.state = new_state
        if self.state in self.goals or self.state in self.bombs:
            self.done = True
        self.step_count += 1
        return self.state, reward, self.done, None

    def reset(self):
        self.done = False
        self.state = self.start
        self.step_count = 0
        return self.state

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

    def get_new_state(self, state, action):
        row = state // self.n
        col = state % self.n
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


    def get_state_loc(self, state):
        row = state // self.n
        col = state % self.n
        return np.array([row, col])

    def get_reward(self, new_state):
        if new_state in self.goals:
            self.done = True
            return 1 - 0.9 * (self.step_count / self.max_steps)
        elif new_state in self.bombs:
            return self.bomb_reward
        elif new_state == self.state:
            return self.bump_reward
        return self.move_reward

    def render(self, logits=None, t=None):
        if self.render_mode == "rgb_array":
            return self._render_frame(logits, t)
        elif self.render_mode == "human":
            self._render_frame(logits, t)
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _render_frame(self, logits=None, t=None):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.n
        )  # The size of a single grid square in pixels

        # First we draw the goal location(s)
        for goal in self.goals:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * self.get_state_loc(goal),
                    (pix_square_size, pix_square_size),
                ),
            )
        # Next we draw the bomb location(s)
        for bomb in self.bombs:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * self.get_state_loc(bomb),
                    (pix_square_size, pix_square_size),
                ),
            )
        # Next we draw the wall location(s)
        for wall in self.walls:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * self.get_state_loc(wall),
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.get_state_loc(self.state) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        if logits is not None:
            # Where the agent's arrows should be drawn in the `_render_frame` function
            # Now we draw the action indicators (as circles) in the squares adjacent to the agent
            # Now we draw the action indicators (as circles) in the squares adjacent to the agent
            for action in range(self.n_actions):
                current_state_loc = self.get_state_loc(self.state)

                if action == 0:  # Up
                    new_state_loc = (current_state_loc[0], current_state_loc[1] - 1)
                elif action == 1:  # Right
                    new_state_loc = (current_state_loc[0] + 1, current_state_loc[1])
                elif action == 2:  # Down
                    new_state_loc = (current_state_loc[0], current_state_loc[1] + 1)
                elif action == 3:  # Left
                    new_state_loc = (current_state_loc[0] - 1, current_state_loc[1])

                # Ensuring that the new state is within the grid
                if new_state_loc[0] > self.m - 1 or new_state_loc[1] > self.n - 1:
                    continue

                # Color the circle based on the action
                color = (logits[action]*255, 0, 0)  # use different colors for different actions as per your requirement
                pygame.draw.circle(
                    canvas,
                    color,
                    (np.array(new_state_loc) + 0.5) * pix_square_size,
                    pix_square_size / 5,
                )
        # Finally, add some gridlines
        for x in range(self.n + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.m *pix_square_size , pix_square_size * x),
                width=3,
            )
        for x in range(self.m + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.n * pix_square_size),
                width=3,
            )

        if t is not None:
            font = pygame.font.Font(None, 36)
            text = font.render("Time: %d" % t, 1, (10, 10, 10))
            textpos = text.get_rect(topright=(self.window_size - 20, 10))  # You can change the position as needed
            canvas.blit(text, textpos)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def get_neighbors_and_actions(self, state):
        neighbors = []
        row, col = self.get_state_loc(state)
        for action in [UP, DOWN, LEFT, RIGHT]:
            new_row, new_col = self.get_state_loc(self.get_new_state(state, action))
            if (new_row, new_col) != (row, col):  # If the agent has moved, then it's a valid neighbor.
                neighbors.append((new_row * self.n + new_col, action))
        return neighbors

    def bfs(self, start_state, goal_state):
        visited = [False] * self.n_states
        queue = deque([(start_state, [])])  # The queue holds tuples of (state, path).

        while queue:
            state, path = queue.popleft()

            if state == goal_state:  # If we've reached the goal, return the path and actions to it.
                return path  # We return the path only, as the action that reached the goal is already included in the path.

            if not visited[state]:  # If the state has not been visited before.
                visited[state] = True

                for neighbor, action in self.get_neighbors_and_actions(state):
                    if not visited[neighbor]:  # If the neighbor has not been visited, enqueue it with the path to it.
                        new_path = path + [
                            (state, action)]  # Add the current state and the action leading to the neighbor.
                        queue.append((neighbor, new_path))

        return None  # If there's no path to the goal, return None.


if __name__ == '__main__':
    env = GridWorld(terminal_reward=1.0, move_reward=0.0, render_mode='human', file_name='map4.txt')
    env.reset()
    while not env.done:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(state, reward, done)
