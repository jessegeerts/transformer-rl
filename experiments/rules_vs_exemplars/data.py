import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GeneralizationDataset(Dataset):
    """Generate few-shot dataset for pre-training in-context transformer model, from the loose description in Chan
    et al.
    """
    def __init__(self, size, mode='fewshot', subvector_len=32, within_class_var=0.1, n_classes_per_feature=10):
        self.size = size
        self.mode = mode
        self.subvector_dim = subvector_len
        self.stim_dim = subvector_len * 2  # stimulus dimensionality
        self.seq_len = 12  # sequence length (excluding query stimulus)
        self.within_class_var = within_class_var  # within-class variance

        self.A, self.B, self.C, self.W, self.X, self.Y, self.Z = self.sample_classes(7, self.subvector_dim)
        self.G, self.H, self.S, self.Q, self.R, self.D = self.sample_classes(6, self.subvector_dim)
        n_rand_classes = 2**7
        self.class_means_random = np.random.choice([-1, 1], (n_rand_classes, self.stim_dim))
        self.labels_random = np.random.choice([0, 1, 2], n_rand_classes)

    def __getitem__(self, idx):
        if self.mode == 'fewshot':
            sequence, labels = self.generate_sequence_fewshot()
        elif self.mode == 'iw_training':
            sequence, labels = self.generate_iw_training_seq()
        elif self.mode == 'iw_generalization_test':
            sequence, labels = self.generate_iw_generalization_seq()
        elif self.mode == 'icl_training':
            sequence, labels = self.generate_icl_training_seq()
        elif self.mode == 'icl_generalization_test':
            sequence, labels = self.generate_icl_generalization_seq()
        elif self.mode == 'icl_eval':
            sequence, labels = self.generate_icl_eval_seq()
        else:
            raise ValueError("Invalid mode. Mode should be 'train' or 'eval'.")

        return torch.tensor(np.array(sequence), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)

    def __len__(self):
        return self.size

    def set_mode(self, mode):
        self.mode = mode

    def generate_sequence_fewshot(self):
        """For few-shot learning, we have 3 classes (G, H, S) and 3 labels. According to the paper, "classes and labels
        were randomly assigned for each sequence". This would be true few-shot learning but it doesn't work at all.
        """
        GH, SQ, RD = self.sample_classes(3)
        stim_sequence = [GH, SQ, RD] * 4  # 4 repetitions of each class
        labels_sequence = [0, 1, 2] * 4
        query_id = np.random.randint(0, 12)
        query_stim = stim_sequence[query_id]
        query_label = labels_sequence[query_id]
        # add gaussian noise to the stimuli
        stim_sequence = np.array(stim_sequence) + np.random.normal(0, self.within_class_var, (12, self.stim_dim))
        # randomly shuffle the sequence
        idx = np.random.permutation(12)
        stim_sequence = stim_sequence[idx]
        labels_sequence = np.array(labels_sequence)[idx]
        # append query stimulus and label
        query_stim = query_stim + np.random.normal(0, 0.1, (self.stim_dim,))
        stim_sequence = np.vstack((stim_sequence, query_stim))
        labels_sequence = np.append(labels_sequence, query_label)
        return stim_sequence, labels_sequence

    def generate_iw_training_seq(self):
        """For in-weight training, we have random stimuli and labels in the context (keep stim-label mapping constant?).
        Then the queries follow the partial exposure paradigm:

            AW --> 0
            AX --> 0
            BW --> 1
            BW --> 1 (note that there are twice as many presentations of BW)
            CY --> 2
            CZ --> 2

        During evaluation (not this function), we will again have a context of random stimuli, with a query of BX, the
        unseen combination of subvectors B and X. The expectation is that this training paradigm leads to rule-based
        generalisation (i.e. using only the previously relevant dimension of A vs B).
        """
        # random context stimuli and labels (but keep the mapping constant so that the model can learn the mapping)
        # note: actually it might not matter, since the model is trained only on the queries and not the context
        # todo: check if this matters
        ids = np.random.randint(0, len(self.labels_random), self.seq_len)
        stim_sequence = self.class_means_random[ids]
        labels_sequence = self.labels_random[ids]
        # choose query from the partial exposure paradigm
        query_id = np.random.choice(['AW', 'AX', 'BW', 'BW', 'CY', 'CZ'])
        if query_id == 'AW':
            query_stim = np.concatenate((self.A, self.W))
            query_label = 0
        elif query_id == 'AX':
            query_stim = np.concatenate((self.A, self.X))
            query_label = 0
        elif query_id == 'BW':
            query_stim = np.concatenate((self.B, self.W))
            query_label = 1
        elif query_id == 'CY':
            query_stim = np.concatenate((self.C, self.Y))
            query_label = 2
        elif query_id == 'CZ':
            query_stim = np.concatenate((self.C, self.Z))
            query_label = 2
        else:
            raise ValueError("Invalid query id.")
        # add gaussian noise to the stimuli
        stim_sequence = stim_sequence + np.random.normal(0, self.within_class_var, (self.seq_len, self.stim_dim))
        query_stim = query_stim + np.random.normal(0, self.within_class_var, (self.stim_dim,))
        return np.vstack((stim_sequence, query_stim)), np.append(labels_sequence, query_label)

    def generate_iw_generalization_seq(self):
        """For in-weight evaluation, we have random context stimuli and labels, and a query of BX. The expectation is
        that this training paradigm leads to rule-based generalisation (i.e. using only the previously relevant dimension
        of A vs B).
        """
        # random context stimuli and labels (but keep the mapping constant so that the model can learn the mapping)
        ids = np.random.randint(0, len(self.labels_random), self.seq_len)
        stim_sequence = self.class_means_random[ids]
        labels_sequence = self.labels_random[ids]
        # choose query from the partial exposure paradigm
        query_stim = np.concatenate((self.B, self.X))
        query_label = 1  # the label is 1 because the relevant dimension is B according to rule-based generalisation.
        # add gaussian noise to the stimuli
        stim_sequence = stim_sequence + np.random.normal(0, self.within_class_var, (self.seq_len, self.stim_dim))
        query_stim = query_stim + np.random.normal(0, self.within_class_var, (self.stim_dim,))
        return np.vstack((stim_sequence, query_stim)), np.append(labels_sequence, query_label)

    def generate_icl_training_seq(self):
        """Since we can't get ICL to work from pure few-shot training, we copy the general approach from other papers,
         which is to train on sequences where both IWL and ICL are valid. We can measure the amount of ICL it's doing
         by evaluation on a separate set of sequences with the same structure but different stimuli.

         fixme: this is not true according to stephanie. we need to do PURE fewshot learning here.

         For assessing generalization, we can be sure that it is doing ICL because the query is a novel combination of
         subvectors that have only been seen in the context.
         """
        # random context stimuli and labels (but keep the mapping constant so that the model can learn the mapping)
        ids = np.random.randint(0, len(self.labels_random), 3)
        stim_sequence = self.class_means_random[ids]
        labels_sequence = self.labels_random[ids]
        # choose query from the context
        query_id = np.random.choice([0, 1, 2])
        query_stim = stim_sequence[query_id]
        query_label = labels_sequence[query_id]
        # repeat the context 4 times and shuffle
        stim_sequence = np.repeat(stim_sequence, 4, axis=0)
        labels_sequence = np.repeat(labels_sequence, 4)
        idx = np.random.permutation(12)
        stim_sequence = stim_sequence[idx]
        labels_sequence = labels_sequence[idx]
        # append query stimulus and label
        stim_sequence = np.vstack((stim_sequence, query_stim))
        labels_sequence = np.append(labels_sequence, query_label)
        # add gaussian noise to the stimuli
        stim_sequence = stim_sequence + np.random.normal(0, self.within_class_var, (13, self.stim_dim))
        return stim_sequence, labels_sequence

    def generate_icl_generalization_seq(self):
        """For evaluating in-context generalisation, we let the context follow the partial exposure paradigm. The query
        is then BX, the unseen combination of subvectors B and X. The expectation is that this training paradigm leads
        to exemplar-based generalisation (i.e. using both the stimulus dimensions of B and X)."""

        stim_sequence = np.vstack(
            (np.concatenate((self.A, self.W)),
             np.concatenate((self.A, self.X)),
             np.concatenate((self.B, self.W)),
             np.concatenate((self.B, self.W)),
             np.concatenate((self.C, self.Y)),
             np.concatenate((self.C, self.Z))))

        labels_sequence = np.array([0, 0, 1, 1, 2, 2])

        # repeat 2 times and shuffle
        stim_sequence = np.repeat(stim_sequence, 2, axis=0)
        labels_sequence = np.repeat(labels_sequence, 2)
        idx = np.random.permutation(12)
        stim_sequence = stim_sequence[idx]
        labels_sequence = labels_sequence[idx]
        # add gaussian noise to the stimuli
        stim_sequence = stim_sequence + np.random.normal(0, self.within_class_var, (12, self.stim_dim))

        # choose query from the partial exposure paradigm
        query_stim = np.concatenate((self.B, self.X))
        query_label = 1  # the label is 1 because the relevant dimension is B according to rule-based generalisation.

        # add gaussian noise to the query stimulus
        query_stim = query_stim + np.random.normal(0, self.within_class_var, (self.stim_dim,))

        # append query stimulus and label
        stim_sequence = np.vstack((stim_sequence, query_stim))
        labels_sequence = np.append(labels_sequence, query_label)

        return stim_sequence, labels_sequence

    def generate_icl_eval_seq(self):
        """we also just want to test how much ICL the network is doing. We can do this by testing on a separate set of
        sequences with the same structure (4 shot 3 way classification) but completely novel stimuli."""
        classes = self.sample_classes(3)
        labels = np.array([0, 1, 2])
        # choose query from the classes
        query_id = np.random.choice([0, 1, 2])
        query_stim = classes[query_id]
        query_label = labels[query_id]
        # repeat 4 times and shuffle
        stim_sequence = np.repeat(classes, 4, axis=0)
        labels_sequence = np.repeat(labels, 4)
        idx = np.random.permutation(12)
        stim_sequence = stim_sequence[idx]
        labels_sequence = labels_sequence[idx]
        # append query stimulus and label
        stim_sequence = np.vstack((stim_sequence, query_stim))
        labels_sequence = np.append(labels_sequence, query_label)
        # add gaussian noise to the stimuli
        stim_sequence = stim_sequence + np.random.normal(0, self.within_class_var, (13, self.stim_dim))
        return stim_sequence, labels_sequence

    def sample_classes(self, n_classes, dim=None):
        if dim is None:
            dim = self.stim_dim
        return np.random.choice([-1, 1], (n_classes, dim))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = GeneralizationDataset(100)
    dataset.set_mode('iw_training')
    stim, labl = dataset[0]
