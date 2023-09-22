"""Q learning agents to generate trajectories. We will use these trajectories to train decision transformers."""

import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0,
                 exploration_decay=0.995, min_exploration_prob=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay
        self.min_exploration_prob = min_exploration_prob
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_prob:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q_table[state, :])

    def choose_action_softmax(self, state):
        logits = self.q_table[state, :]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.random.choice(self.n_actions, p=probs)

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        # Decay the exploration probability
        self.exploration_prob = max(self.min_exploration_prob,
                                    self.exploration_prob * self.exploration_decay)


if __name__ == '__main__':
    from environments.gridworld import ProcGenGrid
    from tqdm import tqdm
    n_episodes = 500

    grid = ProcGenGrid(n=10, m=10, num_walls=10, min_distance=5, render_mode='human', bump_reward=0., max_steps=10000)
    agent = QLearningAgent(n_states=grid.n * grid.m, n_actions=4, exploration_decay=.999999999999)

    for episode in tqdm(range(n_episodes)):
        state = grid.reset()
        done = False
        t = 0
        total_reward = 0
        while not done and t <= grid.max_steps:
            action = agent.choose_action_softmax(state)
            next_state, reward, done, _ = grid.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            t += 1

        if episode % 10 == 0:
            print(f"Episode {episode}, total reward {total_reward}")

    # Test the trained agent
    state = grid.reset()
    done = False
    grid.render(logits=agent.q_table[state]/np.max(agent.q_table[state]))
    while not done:
        action = agent.choose_action_softmax(state)
        state, _, done, _ = grid.step(action)
        grid.render()
