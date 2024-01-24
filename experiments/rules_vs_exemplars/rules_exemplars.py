import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations


class TwoDimensionalDataGen(object):
    """Generate data according to the 2D example in Dasgupta et al (2022) ICML.

    Inputs are generated by a composition of categorical attributes, with two latent binary features, z_disc and z_dist.
    The discrimative feature z_disc is a binary feature that determines the class label, and the distractor feature
    z_dist is purely distraction and does not affect the class label.

    We enforce that (p(z_disc = 0) = p(z_disc = 1) =.5 and we fix the number of training examples to be n.

    With these constraints, we can vary p(z_disc, z_dist) via two degrees of freedom:
    pi_0 = p(z_dist = 1| z_disc = 0)
    pi_1 = p(z_dist = 1| z_disc = 1)

    The data is generated as follows:
    p(x|z_disc, z_dist) = N(mu, 1)
    mu = alpha * [2 * z_disc -1, 2 * z_dist - 1]
    """

    def __init__(self, n=300, seed=0):
        self.n = n
        self.seed = seed
        self.train_x, self.train_y = self.generate_data()
        self.test_x, self.test_y = self.generate_data(pi_0=0, pi_1=1)
        self.test_x = self.test_x[self.test_y == 1]
        self.test_y = self.test_y[self.test_y == 1]

    def generate_data(self, pi_0=.5, pi_1=0., alpha=3):
        """Generate data according to the above description."""
        np.random.seed(self.seed)
        z_disc = np.random.binomial(1, .5, self.n)
        z_dist = np.random.binomial(1, z_disc * pi_1 + (1 - z_disc) * pi_0, self.n)
        mu = alpha * np.vstack([2 * z_disc - 1, 2 * z_dist - 1]).T
        x = np.random.normal(mu, 1)
        return x, z_disc


class SequenceDataGen(object):
    def __init__(self, num_classes=10, feature_length=32, num_values_per_class=100, covariance_scale=0.1):
        self.num_classes = num_classes
        self.feature_length = feature_length
        self.stimulus_dim = feature_length * 2
        self.num_values_per_class = num_values_per_class
        self.covariance_scale = covariance_scale

        # Define the 12 subvectors with elements as 1 or -1
        self.subvectors = {label: np.random.choice([-1, 1], feature_length) for label in "GHSQRDABCXYZW"}

        self.stimulus_classes = {'GH': 0,
                                 'SQ': 1,
                                 'RD': 2,
                                 'AW': 0,
                                 'AX': 0,
                                 'BW': 1,
                                 'CY': 2,
                                 'CZ': 2,
                                 'BX': None}

        # Define the centroids as the sum of two subvectors (for now only some of the combinations)
        self.centroids = {"{}{}".format(i, j): np.concatenate([self.subvectors[i], self.subvectors[j]]) for i, j in
                          self.stimulus_classes}

        # Create a mapping from subvector class pairs to unique integer labels
        self.class_pair_to_label = {stim_class: label for stim_class, label in zip(self.stimulus_classes, range(9))}

    def sample_stimulus(self, class_id):
        """Generate a subvector for a given class."""
        mean = self.centroids[class_id]
        covariance = np.eye(self.stimulus_dim) * self.covariance_scale
        return np.random.multivariate_normal(mean, covariance)

    def new_stimulus_centroid(self):
        """Generate a new stimulus centroid, ensuring it doesn't ."""
        while True:
            v1 = np.random.choice([-1, 1], self.feature_length)
            if not any(np.array_equal(v1, sv) for sv in self.subvectors.values()):
                break
        while True:
            v2 = np.random.choice([-1, 1], self.feature_length)
            if not any(np.array_equal(v2, sv) for sv in self.subvectors.values()):
                break
        return np.concatenate([v1, v2])

    def generate_stimulus_and_label(self):
        """Generate a stimulus and its corresponding single integer label."""
        class_ids = np.random.choice(self.num_classes, 2, replace=False)
        subvectors = [self.generate_subvector(class_id) for class_id in class_ids]
        stimulus = np.concatenate(subvectors)
        label = self.class_pair_to_label[tuple(class_ids)]  # Single integer label
        return stimulus, label

    def generate_stimuli_few_shot_random_context(self, n_reps=4, batch_size=1):
        context_labels = np.array([0, 1, 2])

        # Pre-allocate memory
        stim_sequence = np.zeros((batch_size, n_reps * 3, self.stimulus_dim))  # Assuming centroid has a specific dimension
        labels_sequence = np.zeros((batch_size, n_reps * 3), dtype=int)
        query_stimuli = np.zeros((batch_size, self.stimulus_dim))
        query_labels = np.zeros(batch_size, dtype=int)

        for trial in range(batch_size):
            #np.random.shuffle(context_labels)
            centroids = np.array([self.new_stimulus_centroid() for _ in range(3)])
            query_id = np.random.choice(3)
            query_stimulus = np.random.normal(centroids[query_id], scale=self.covariance_scale)
            query_label = context_labels[query_id]

            for i in range(3):
                stim_idx = slice(n_reps * i, n_reps * (i + 1))
                stim_sequence[trial, stim_idx] = np.random.normal(centroids[i], size=(n_reps, self.stimulus_dim),
                                                                  scale=self.covariance_scale)
                labels_sequence[trial, stim_idx] = context_labels[i]

            # randomly permute the order of the stimuli
            perm = np.random.permutation(n_reps * 3)
            stim_sequence[trial] = stim_sequence[trial][perm]
            labels_sequence[trial] = labels_sequence[trial][perm]
            # append query stimulus
            query_stimuli[trial] = query_stimulus
            query_labels[trial] = query_label

        return stim_sequence, labels_sequence, query_stimuli, query_labels

    def generate_stimuli_few_shot(self, n_reps=4, n_trials=100, return_stim_names=False):
        # first context stimuli

        stim_names = []
        stim_sequence = []
        labels_sequence = []
        query_stimuli = []
        query_labels = []
        query_names = []

        context_labels = [0, 1, 2]
        context_stimuli = ['GH', 'SQ', 'RD']

        for trial in range(n_trials):
            # shuffle the context labels, such that the mapping between context stimuli and labels is randomly switched
            # every n_reps trials. This means the model needs to learn few-shot in-context
            np.random.shuffle(context_labels)
            # also shuffle the context stimuli order (#FIXME: model should be invariant to this but breaks if randomized)
            np.random.shuffle(context_stimuli)
            # 3 context stimuli, 3 context labels, 1 query stimulus, 1 query label
            query_id = np.random.choice(3)
            query_stimulus = self.sample_stimulus(context_stimuli[query_id])
            query_name = context_stimuli[query_id]
            query_label = context_labels[query_id]

            stimuli = []
            labels = []
            names = []
            for rep in range(n_reps):
                for i, stim in enumerate(context_stimuli):
                    stimuli.append(self.sample_stimulus(stim))
                    names.append(stim)
                    labels.append(context_labels[i])

            stim_sequence.append(
                stimuli)  # we shouldn't have exact repetitions of the same stimulus but samples from the same class
            stim_names.append(names)
            labels_sequence.append(labels)
            query_stimuli.append([query_stimulus])
            query_labels.append([query_label])
            query_names.append([query_name])

        return stim_sequence, stim_names, labels_sequence, query_stimuli, query_labels, query_names

    def generate_sequence(self, context_length=12):
        """Generate a sequence of alternating stimuli and labels."""
        stimuli = []
        labels = []
        for _ in range(context_length):
            stimulus, label = self.generate_stimulus_and_label()
            stimuli.append(stimulus)
            labels.append(label)
        query_stimulus, _ = self.generate_stimulus_and_label()
        return stimuli, labels, query_stimulus


def classify_data(x, y, clf):
    """Classify data using clf and return the accuracy."""
    # first randomly shuffle the data
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    # fit the classifier
    clf.fit(x, y)

    # predict the labels to compute the decision boundary
    # Create a mesh
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # predict the labels
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    return clf, Z
