import numpy as np
import torch
from matplotlib import pyplot as plt


def compute_ratemaps(model, context_len, env, rtg_target, rtg_scale,
                                num_eval_episodes=100, max_ep_len=1000, start_state1=48, start_state2=50):
    # TODO: make this work for more layers

    mlp_dim = model.transformer.blocks[0].mlp1[0].out_features
    # Placeholders for grid size, initialize as zero arrays
    grid_size = env.observation_space.n
    mlp_activations_map1 = np.zeros((grid_size, 128))  # Assuming MLP has 128 units
    mlp_activations_map2 = np.zeros((grid_size, 128))  # Assuming MLP has 128 units
    visit_counts1 = np.zeros(grid_size)
    visit_counts2 = np.zeros(grid_size)

    # Timesteps tensor
    timesteps = torch.arange(start=0, end=max_ep_len, step=1).repeat(1, 1)

    model.eval()

    with torch.no_grad():
        for _ in range(num_eval_episodes):
            # Initialize placeholders for states and actions
            states = torch.zeros((1, max_ep_len), dtype=torch.int32)
            actions = torch.zeros((1, max_ep_len), dtype=torch.int32)
            rewards_to_go = torch.zeros((1, max_ep_len, 1), dtype=torch.float32)

            states[:] = env.n_states
            actions[:] = env.n_actions

            # Init episode
            running_state = env.reset()
            running_reward = 0
            running_rtg = rtg_target / rtg_scale

            # Select the corresponding activations map and visit counts
            if running_state == start_state1:
                mlp_activations_map_current = mlp_activations_map1
                visit_counts_current = visit_counts1
            elif running_state == start_state2:
                mlp_activations_map_current = mlp_activations_map2
                visit_counts_current = visit_counts2

            for t in range(max_ep_len):
                # Add state to placeholder
                states[0, t] = running_state

                # Calculate running rtg and add to placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                rewards_to_go[0, t] = running_rtg

                if t < context_len:
                    _, act_preds, _, all_weights, mlp_activations = model.forward(timesteps[:, :context_len],
                                                    states[:, :context_len],
                                                    actions[:, :context_len],
                                                    rewards_to_go[:, :context_len])
                    act_preds = act_preds[0, t].detach()
                else:
                    _, act_preds, _, all_weights, mlp_activations = model.forward(timesteps[:, t - context_len + 1:t + 1],
                                                    states[:, t - context_len + 1:t + 1],
                                                    actions[:, t - context_len + 1:t + 1],
                                                    rewards_to_go[:, t - context_len + 1:t + 1])
                    act_preds = act_preds[0, -1].detach()

                act = torch.argmax(act_preds[:-1]).item()
                running_state, running_reward, done, _ = env.step(act)

                # Update activations map and visit counts
                mlp_activations_map_current[running_state] += mlp_activations[0][0, -1, :].cpu().numpy()
                visit_counts_current[running_state] += 1

                actions[0, t] = act

                if done:
                    break

    # Normalize the activations by visit counts
    mlp_activations_map1 /= np.maximum(visit_counts1.reshape(-1, 1), 1)  # To prevent division by zero
    mlp_activations_map2 /= np.maximum(visit_counts2.reshape(-1, 1), 1)  # To prevent division by zero

    # set wall locations to nan:
    mlp_activations_map1[env.walls] = np.nan
    mlp_activations_map2[env.walls] = np.nan

    # reshape to 2D:
    mlp_activations_map1 = mlp_activations_map1.reshape(env.m, env.n, mlp_dim).transpose(2, 0, 1)
    mlp_activations_map2 = mlp_activations_map2.reshape(env.m, env.n, mlp_dim).transpose(2, 0, 1)

    return mlp_activations_map1, mlp_activations_map2


def plot_tmaze_ratemaps(unit_ids, ratemaps_1, ratemaps_2, env=None):
    max_units = 32
    num_units = len(unit_ids)
    if num_units > max_units:
        print(f'Only plotting the first {max_units} units')

    # Create a 4 by 8 grid of subplots
    num_rows = 8
    num_cols = 8
    # Create the figure and the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    # Reshape axs for easier indexing
    axs = np.reshape(axs, -1)
    # Plot the ratemaps
    for i, unit in zip(range(max_units), unit_ids):
        # Condition 1
        axs[2 * i].imshow(ratemaps_1[unit], cmap='viridis', interpolation='nearest')
        axs[2 * i].set_title(f'Unit {unit + 1} Condition 1')
        # Condition 2
        axs[2 * i + 1].imshow(ratemaps_2[unit], cmap='viridis', interpolation='nearest')
        axs[2 * i + 1].set_title(f'Unit {unit + 1} Condition 2')

        if env is not None:
            # plot the start locations
            sx1, sy1 = env.get_state_loc(env.start_states[0])
            axs[2 * i].scatter(sy1, sx1, marker='*', s=100, c='red')
            sx2, sy2 = env.get_state_loc(env.start_states[1])
            axs[2 * i + 1].scatter(sy2, sx2, marker='*', s=100, c='red')

    # Remove empty subplots if there are less than num_units * 2 plots
    for i in range(num_units * 2, num_rows * num_cols):
        fig.delaxes(axs[i])
    plt.tight_layout()
    return fig, axs
