import torch
from torch.utils.data import Dataset
import numpy as np


def discount_cumsum(x, gamma):
    """Discounted cumulative sum.
    """
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


class TrajectoryDataset(Dataset):
    """Class to hold a dataset of trajectories. Can be used with pytorch DataLoader.

    Usage example:

    from torch.utils.data import DataLoader

    traj_data_loader = DataLoader(
                        traj_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True
                    )

    data_iter = iter(traj_data_loader)
    timesteps, states, actions, returns_to_go, traj_mask = next(data_iter)

    """
    def __init__(self, trajectories, context_len, rtg_scale, random_truncation=False):
        """

        :param trajectories: list of trajectories, each is a dict with keys 'observations', 'actions', 'reward'
        :param context_len: length of context for the transformer
        :param rtg_scale: scale of reward-to-go
        """
        self.trajectories = trajectories
        self.context_len = context_len
        self.random_truncation = random_truncation

        # calculate min len of trajectory, state mean and std
        # and reward-to-go for all trajectories
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])
            # calculate returns to go and rescale them
            traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states  (TODO: maybe dont do this for one-hot states)
        #for traj in self.trajectories:
        #    traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            if traj_len == self.context_len:
                si = 0
            else:
                si = np.random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations'])
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=states.dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions'])
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=actions.dtype)],
                               dim=0)

            returns_to_go = torch.from_numpy(traj['returns_to_go'])
            returns_to_go = torch.cat([returns_to_go,
                                torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
                                dtype=returns_to_go.dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        if self.random_truncation:
            # sample a random subsequence of the trajectory
            min_traj_len = 2
            sample_len = np.random.randint(min_traj_len, len(timesteps)+1)
            sample_len = 1
            if sample_len == len(timesteps):
                si = 0
            else:
                si = np.random.randint(0, traj_len - sample_len)
            ei = si + sample_len

            timesteps = timesteps[si:ei]
            states = states[si:ei]
            actions = actions[si:ei]
            returns_to_go = returns_to_go[si:ei]
            traj_mask = traj_mask[si:ei]
        return timesteps, states, actions, returns_to_go, traj_mask
