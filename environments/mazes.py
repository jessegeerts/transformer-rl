from environments.gridworld import GridWorld


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


if __name__ == '__main__':
    """Collect some optimal trajectories and save in right format for TrajectoryDataset."""
    env = AltTmaze(render_mode='human')


    env.step(0)

    env.reset()
    env.render()
    for _ in range(50):
        env.step(env.action_space.sample())
        env.render()
    env.close()