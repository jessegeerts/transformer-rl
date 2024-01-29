import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BurstyTrainingDataset(Dataset):
    """Custom dataset for testing ICL and IWL strategies (as described in Reddy et al., 2023).

    The dataset consists of sequences of stimuli and labels, where each stimulus is a D-dimensional vector. Each
    sequence consists of a context set of N stimuli and N labels, followed by a query stimulus. The task is to predict
    the label of the query stimulus.
    """
    P = 65  # dimension of position embeddings
    N = 8  # number of stimuli in a sequence

    def __init__(self, size=1024, K=128, B=4, alpha=1.0, D=63, Pb=1., L=32, epsilon=0.1):
        self.size = size
        self.K = K  # number of classes
        self.B = B  # burstiness of sequences
        self.D = D  # dimension of stimuli
        self.L = L
        self.epsilon = epsilon  # magnitude of within-class noise
        self.alpha = alpha  # zipf parameter
        self.Pb = Pb  # proportion of bursty sequences
        self.class_means = np.random.normal(0, 1 / self.D, (
        K, self.D))  # Mean vectors for each class (note: could change this to the chan et al. means)
        self.class_labels = np.random.choice(self.L, size=K)  # Labels for each class
        self.mode = 'train'

    def __len__(self):
        return self.size

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'train':
            sequence, labels = self.generate_training_sequence()
        elif self.mode == 'eval_iwl':
            sequence, labels = self.generate_eval_sequence_iwl()
        elif self.mode == 'eval_icl':
            sequence, labels = self.generate_eval_sequence_icl()
        else:
            raise ValueError("Invalid mode. Mode should be 'train' or 'eval'.")

        return torch.tensor(np.array(sequence), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)

    def generate_training_sequence(self):
        # todo: ensure that each label appears the same number of times in each sequence
        # first, determine whether the sequence is bursty or not
        is_bursty = np.random.rand() < self.Pb
        if is_bursty:
            # generate a bursty sequence
            sequence, labels = self.generate_bursty_sequence()
        else:
            # generate a iid sequence
            sequence, labels = self.generate_iid_sequence()
        return sequence, labels

    def generate_iid_sequence(self):
        # generate a sequence of N stimuli and labels
        sequence = []
        labels = []
        classes = []
        for i in range(self.N):
            # sample a class according to the zipf distribution
            k = self.sample_class()
            x, y = self.sample_item(k)
            sequence.append(x)
            labels.append(y)
            classes.append(k)
        # append the query stimulus
        k = np.random.randint(self.K + 1)
        x, y = self.sample_item(k)
        sequence.append(x)
        labels.append(y)
        return sequence, labels

    def generate_bursty_sequence(self):
        """Generate a bursty sequence of N stimuli and labels.

        The burstiness B is the number of occurrences of items from a particular class in an input sequence (N is a
        multiple of B). pB is the fraction of bursty sequences. Specifically, the burstiness is B for a fraction pB of
        the training data.
        """
        sequence = []
        labels = []
        if self.B == 0:  # if B = 0, then the sequence is IID, and the query stimulus is sampled from the same distribution
            n_classes = self.N
        else:  # if B = 1, the
            n_classes = self.N // self.B  # number of classes in the sequence

        equal_n_labels = False
        while not equal_n_labels:
            context_classes = [self.sample_class() for _ in range(
                n_classes)]  # choose the classes fixme: this should ensure that each label appears the same number of times
            unique_labels, counts = np.unique(self.class_labels[context_classes], return_counts=True)


        if self.B > 0:
            context_classes = np.repeat(context_classes, self.B)  # repeat each class B times
        else:
            context_classes = np.array(context_classes)
        context_classes = context_classes[np.random.permutation(self.N)]  # randomly permute the order of the classes

        for k in context_classes:
            x, y = self.sample_item(k)
            sequence.append(x)
            labels.append(y)

        # append the query stimulus
        if self.B == 0:
            k = np.random.randint(self.K)
        else:  # for bursty sequences, the query stimulus is in the context
            k = np.random.choice(context_classes)
        x, y = self.sample_item(k)
        sequence.append(x)
        labels.append(y)
        return sequence, labels

    def generate_eval_sequence_iwl(self):
        """Generate sequence for evaluating in weights learning.

        In IWL evaluation, target and item classes are sampled independently from the rank-frequency distribution used
        during training. K >>N, so it's unlikely the target's class appears in the context (i.e. it has to rely on IWL).

        TODO: should this also be bursty? (i.e. should we sample a class and then repeat it B times?)
        """
        # generate a sequence of N stimuli and labels
        sequence = []
        labels = []
        classes = []
        for i in range(self.N):
            # sample a class according to the zipf distribution
            k = self.sample_class()
            x, y = self.sample_item(k)
            sequence.append(x)
            labels.append(y)
            classes.append(k)
        # append the query stimulus
        k = self.sample_class()
        x, y = self.sample_item(k)
        sequence.append(x)
        labels.append(y)
        return sequence, labels

    def generate_eval_sequence_icl(self):
        """
        In ICL evaluation, we draw completely new classes which are assigned to existing labels.
        """
        if self.B == 0:
            n_classes = self.N
        else:
            n_classes = self.N // self.B  # number of classes in the sequence = sequence length / burstiness
        class_means = np.random.normal(0, 1 / self.D, (n_classes, self.D))
        class_labels = np.random.choice(self.L, size=n_classes)  # assign random labels to the classes
        sequence = []
        labels = []
        if self.B > 0:
            classes = np.arange(n_classes).repeat(self.B)
        else:
            classes = np.arange(n_classes)
        classes = classes[np.random.permutation(self.N)]  # randomly permute the order of the classes

        for k in classes:
            x = class_means[k] + self.epsilon * np.random.normal(0, 1 / self.D, self.D)
            y = class_labels[k]
            sequence.append(x)
            labels.append(y)

        # append the query stimulus
        k = np.random.choice(classes)
        x = class_means[k] + self.epsilon * np.random.normal(0, 1 / self.D, self.D)
        y = class_labels[k]
        sequence.append(x)
        labels.append(y)
        return sequence, labels

    def sample_item(self, k):
        x = self.class_means[k] + self.epsilon * np.random.normal(0, 1 / self.D, self.D)
        #  x /= np.sqrt(1 + self.epsilon ** 2)  todo: should we even norm? it won't be a mixture of gaussians anymore
        y = self.class_labels[k]
        return x, y

    def sample_class(self):
        """Sample a class according to the zipf distribution.
        """
        p = np.arange(1, self.K + 1) ** - self.alpha / np.sum(np.arange(1, self.K + 1) ** - self.alpha)
        return np.random.choice(self.K, p=p)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ---------------------------------
    # example with D = 2, K = 2**4 classes
    xx = []
    yy = []
    dataset = BurstyTrainingDataset(K=2 ** 4, D=2)
    for k in range(dataset.K):
        for i in range(20):
            x, y = dataset.sample_item(k)
            xx.append(x)
            yy.append(y)

    plt.scatter(*np.array(xx).T, c=yy)

    # ---------------------------------
    # bursty sequence example

    dataset = BurstyTrainingDataset(K=2 ** 4, D=2)
    sequence, labels = dataset.generate_training_sequence()

    plt.figure()
    plt.scatter(*np.array(sequence).T, c=labels)
    plt.scatter(*np.array(sequence[-1]).T, c='r', marker='x')
    plt.title('Bursty sequence')

    # ---------------------------------
    # iid sequence example
    dataset = BurstyTrainingDataset(K=2 ** 4, D=2, Pb=0.0)
    sequence, labels = dataset.generate_training_sequence()

    plt.figure()
    plt.scatter(*np.array(sequence).T, c=labels)
    plt.scatter(*np.array(sequence[-1]).T, c='r', marker='x')
    plt.title('IID sequence')

    # ---------------------------------
    # IWL evaluation sequence example
    dataset = BurstyTrainingDataset(K=2 ** 4, D=2, Pb=1.0)
    dataset.set_mode('eval_iwl')
    sequence, labels = dataset.generate_training_sequence()

    plt.figure()
    plt.scatter(*np.array(sequence).T, c=labels)
    plt.scatter(*np.array(sequence[-1]).T, c='r', marker='x')
    plt.title('IWL evaluation sequence')

    # ---------------------------------
    # ICL evaluation sequence example
    dataset = BurstyTrainingDataset(K=2 ** 4, D=2, Pb=1.0)
    dataset.set_mode('eval_icl')
    sequence, labels = dataset.generate_training_sequence()

    plt.figure()
    plt.scatter(*np.array(sequence).T, c=labels)
    plt.scatter(*np.array(sequence[-1]).T, c='r', marker='x')
    plt.title('ICL evaluation sequence')

    # ---------------------------------
    # check that dataloader works

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        break

    # then switch mode
    dataset.set_mode('eval_icl')
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for x, y in eval_loader:
        print(x.shape)
        print(y.shape)
        break

    # ---------------------------------

    # ---------------------------------

    from torch.utils.data import DataLoader
    import torch

    B_values = [0, 1, 2, 4]
    K_values = [2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11]

    for i, B in enumerate(B_values):
        for j, K in enumerate(K_values):
            dataset = BurstyTrainingDataset(K=K, D=63, Pb=1.0, B=B)
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            for x, y in loader:
                assert np.all(x.shape == np.array([4, 9, 63]))  # check that the shape is correct
                break

    # ---------------------------------
    # check that the labels are balanced
    dataset = BurstyTrainingDataset(K=2 ** 8, D=2, Pb=1.0, B=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        # does each label appear the same number of times?
        labels, counts = np.unique(y[:, :-1], return_counts=True)
