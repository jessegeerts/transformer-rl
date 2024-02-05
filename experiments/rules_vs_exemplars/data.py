import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GeneralizationDataset(Dataset):
    """Generate few-shot dataset for pre-training in-context transformer model, from the loose description in Chan
    et al.
    """
    def __init__(self, size, mode='fewshot', D=64):
        self.size = size
        self.mode = mode
        self.D = D  # stimulus dimensionality

    def __getitem__(self, idx):
        if self.mode == 'fewshot':
            sequence, labels = self.generate_sequence()
        else:
            raise ValueError("Invalid mode. Mode should be 'train' or 'eval'.")

        return torch.tensor(np.array(sequence), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)

    def __len__(self):
        return self.size

    def set_mode(self, mode):
        self.mode = mode

    def generate_sequence(self):
        GH, SQ, RD = self.sample_classes(3)
        stim_sequence = [GH, SQ, RD] * 4  # 4 repetitions of each class
        labels_sequence = [0, 1, 2] * 4
        query_id = np.random.randint(0, 12)
        query_stim = stim_sequence[query_id]
        query_label = labels_sequence[query_id]
        # add gaussian noise to the stimuli
        stim_sequence = np.array(stim_sequence) + np.random.normal(0, 0.1, (12, self.D))
        # randomly shuffle the sequence
        idx = np.random.permutation(12)
        stim_sequence = stim_sequence[idx]
        labels_sequence = np.array(labels_sequence)[idx]
        # append query stimulus and label
        query_stim = query_stim + np.random.normal(0, 0.1, (self.D,))
        stim_sequence = np.vstack((stim_sequence, query_stim))
        labels_sequence = np.append(labels_sequence, query_label)
        return stim_sequence, labels_sequence

    def sample_classes(self, n_classes):
        return np.random.choice([-1, 1], (n_classes, self.D))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = GeneralizationDataset(100)
    stim, labl = dataset.generate_sequence()