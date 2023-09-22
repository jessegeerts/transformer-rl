"""
Idea: procedurally generate grid worlds with different obstacles. We would like to know if training decision
tansfomers on these procedurally generated grid worlds will generalize to unseen grid worlds, and whether predicting
observations acts as a useful auxiliary task for this generalization.

1. First we need to collect some trajectories. Do this by training an RL agent on a batch of procedurally generated
grid worlds. we can train separate agents for each grid world.
2. Then we need to train a decision transformer on these trajectories.
3. Then we need to evaluate the decision transformer on a batch of procedurally generated grid worlds.
"""

# collect trajectories from procedurally generated grid worlds
n_envs = 50