import numpy as np
import torch
from torch.utils.data import Dataset


# μk = np.random.normal(0, 1/D, (K, D))  # Mean vectors for each class


class BurstyDataset(Dataset):
    """Custom dataset for testing ICL and IWL strategies (as described in Reddy et al., 2023).

    The dataset consists of sequences of stimuli and labels, where each stimulus is a D-dimensional vector. Each
    sequence consists of a context set of N stimuli and N labels, followed by a query stimulus. The task is to predict
    the label of the query stimulus.
    """
    P = 65  # dimension of position embeddings
    L = 32  # number of labels
    N = 8  # number of stimuli in a sequence
    epsilon = 0.1  # within class noise

    def __init__(self, size=1000, K=128, B=4, alpha=1.0, D=63, Pb=1.):
        self.size = size
        self.K = K  # number of classes
        self.B = B  # burstiness of sequences
        self.D = D  # dimension of stimuli
        self.alpha = alpha  # zipf parameter
        self.Pb = Pb  # proportion of bursty sequences
        self.class_means = np.random.normal(0, 1/self.D, (K, self.D))  # Mean vectors for each class (note: could change this to the chan et al. means)
        self.class_labels = np.random.choice(self.L, size=K)  # Labels for each class

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sequence, labels = self.generate_training_sequence()
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def generate_training_sequence(self):
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
        k = np.random.choice(classes)
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
        n_classes = self.N // self.B  # number of classes in the sequence
        classes = [self.sample_class() for _ in range(n_classes)]  # choose the classes fixme: this should be sampled from the zipf distribution
        classes = np.repeat(classes, self.B)  # repeat each class B times
        classes = classes[np.random.permutation(self.N)]  # randomly permute the order of the classes

        for k in classes:
            x, y = self.sample_item(k)
            sequence.append(x)
            labels.append(y)

        # append the query stimulus
        k = np.random.choice(classes)
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
        n_classes = self.N // self.B  # number of classes in the sequence = sequence length / burstiness
        class_means = np.random.normal(0, 1 / self.D, (n_classes, self.D))
        class_labels = np.random.choice(self.L, size=n_classes)  # assign random labels to the classes
        sequence = []
        labels = []
        classes = np.arange(n_classes).repeat(self.B)
        classes = classes[np.random.permutation(self.N)]  # randomly permute the order of the classes

        for k in classes:
            x = class_means[k] + self.epsilon * np.random.normal(0, 1/self.D, self.D)
            y = class_labels[k]
            sequence.append(x)
            labels.append(y)

        # append the query stimulus
        k = np.random.choice(classes)
        x = class_means[k] + self.epsilon * np.random.normal(0, 1/self.D, self.D)
        y = class_labels[k]
        sequence.append(x)
        labels.append(y)
        return sequence, labels
        
    def sample_item(self, k):
        x = self.class_means[k] + self.epsilon * np.random.normal(0, 1/self.D, self.D)
        y = self.class_labels[k]
        return x, y

    def sample_class(self):
        """Sample a class according to the zipf distribution."""
        p = np.arange(1, self.K+1) ** - self.alpha / np.sum(np.arange(1, self.K+1) ** - self.alpha)
        return np.random.choice(self.K, p=p)




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ---------------------------------
    # example with D = 2, K = 2**4 classes
    xx = []
    yy = []
    dataset = BurstyDataset(K=2**4, D=2)
    for k in range(dataset.K):
        for i in range(20):
            x, y = dataset.sample_item(k)
            xx.append(x)
            yy.append(y)

    plt.scatter(*np.array(xx).T, c=yy)

    # ---------------------------------
    # bursty sequence example

    dataset = BurstyDataset(K=2**4, D=2)
    sequence, labels = dataset.generate_training_sequence()
