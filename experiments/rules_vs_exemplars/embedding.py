import torch
from torch import nn


def _create_positional_encodings(inputs, max_time=30.0):
    """Generates positional encodings for the input. Adapted from haiku / jax (Chan et al.) to torch  by Jesse Geerts.

    Note, in Chan et al, max_time is set to 30, probably because the sequences were short.

  Args:
    inputs: A tensor of shape [batch_size, seq_len, emb_size].
    max_time: (default 10000) Constant used to scale position by in the
      encodings.

  Returns:
    pos_emb: as defined above, of size [1, seq_len, emb_size].
  """

    _, seq_len, embedding_size = inputs.shape

    if embedding_size % 2 == 1:
        raise ValueError(
            'Embedding sizes must be even if using positional encodings.')

    # Generate a sequence of positions and frequencies.
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.arange(0, embedding_size, 2, dtype=torch.float32)
    inverse_freqs = 1.0 / (max_time ** (freqs / embedding_size))

    # We combine [seq_len] and [emb_size / 2] to [seq_len, emb_size / 2].
    pos_emb = torch.einsum('i,j->ij', pos, inverse_freqs)

    # Concat sines and cosines and return.
    pos_emb = torch.concatenate([torch.sin(pos_emb), torch.cos(pos_emb)], -1)

    return pos_emb


class InputEmbedder(nn.Module):
    """Input embedder.
  Adapted from haiku / jax (Chan et al.) to torch  by Jesse Geerts. For now I've deleted the resnet option.
  """

    def __init__(self,
                 num_classes=1623,
                 emb_dim=64,
                 token_dim=64,
                 example_encoding='linear',
                 flatten_superpixels=False,
                 example_dropout_prob=0.0,
                 concatenate_labels=False,
                 use_positional_encodings=True,
                 positional_dropout_prob=0.1,
                 name=None):
        """Initialize the input embedder.

    Args:
      num_classes: Total number of output classes.
      emb_dim: Dimensionality of example and label embeddings.
      example_encoding: How to encode example inputs.
        'resnet': simple resnet encoding  (not supported for now)
        'linear': flatten and pass through a linear layer
        'embedding': pass through an embedding layer
      flatten_superpixels: Whether to flatten the output of the resnet (instead
        of taking a mean over superpixels).
      example_dropout_prob: Dropout probability on example embeddings. Note that
        these are applied at both train and test.
      concatenate_labels: Whether to concatenate example and label embeddings
        into one token for each (example, label) pair, rather than being fed to
        the transformer as two separate tokens.
      use_positional_encodings: Whether to use positional encoding.
      positional_dropout_prob: Positional dropout probability.
      name: Optional name for the module.
    """
        super(InputEmbedder, self).__init__()
        self._num_classes = num_classes
        self._emb_dim = emb_dim
        self._example_encoding = example_encoding
        self._flatten_superpixels = flatten_superpixels
        self._example_dropout_prob = example_dropout_prob
        self._concatenate_labels = concatenate_labels
        self._use_positional_encodings = use_positional_encodings
        self._positional_dropout_prob = positional_dropout_prob

        # define trainable parameters below:
        # ================================
        # Initialize example embeddings
        if self._example_encoding == 'embedding':
            self.example_embedding = nn.Embedding(num_classes, emb_dim)
        elif self._example_encoding == 'linear':
            self.example_embedding = nn.Linear(token_dim, self._emb_dim)
        else:
            raise ValueError('Invalid example_encoding: %s' % self._example_encoding)

        # initialize label embeddings
        # Embed the labels.  # todo: add back functionality for concatenating labels
        n_emb_classes = self._num_classes
        # labels_to_embed = labels
        # if self._concatenate_labels:
        #     # Dummy label for final position, where we don't want the label
        #     # information to be available.
        #     n_emb_classes += 1
        #     labels_to_embed[:, -1] = n_emb_classes - 1

        # Initialize label embeddings
        self.embs = nn.Parameter(torch.Tensor(n_emb_classes, self._emb_dim))
        nn.init.trunc_normal_(self.embs, mean=0.0, std=0.02)

    def __call__(self, examples, labels, is_training=True):
        """Call to the input embedder.

        Args:
          examples: input sequence of shape
            [batch_size, seq_len, token_dim]
          labels: input sequence of shape [batch_size, seq_len]
          is_training: if is currently training.

        Returns:
          outputs: output of the transformer tower
            of shape [batch_size, seq_len, channels].
        """
        # Encode the example inputs into shape (B, SS, E)
        h_example = self.example_embedding(examples)

        # Add dropout to example embeddings.
        # Note that this is not restricted to training, because the purpose is to
        # add noise to the examples, not for regularization.
        if self._example_dropout_prob:
            h_example = nn.Dropout(self._example_dropout_prob)(h_example)

        labels_to_embed = labels
        h_label = self.embs[labels_to_embed]  # (B, SS, E)

        if self._concatenate_labels:
            # Concatenate example and label embeddings
            hh = torch.cat((h_example, h_label), dim=2)  # (B,SS,E*2)
        else:
            # Interleave example and label embeddings
            B, SS, E = h_example.shape
            # Create an empty tensor of the desired shape. Note: Use torch.empty_like to match dtype.
            hh = torch.empty(B, SS * 2 - 1, E, dtype=h_example.dtype)

            # Interleave example and label embeddings
            # Set slices for h_example and h_label alternately
            hh[:, 0::2, :] = h_example
            hh[:, 1::2, :] = h_label[:, :-1, :]  # Assuming h_label has the same second dimension as h_example
            # hh is (B,S,E) where S=SS*2-1

        # Create positional encodings.
        if self._use_positional_encodings:
            positional_encodings = _create_positional_encodings(hh)
            if is_training:
                positional_encodings = nn.Dropout(self._positional_dropout_prob)(positional_encodings)
            # Add on the positional encoding.
            hh += positional_encodings
        return hh


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    examples = torch.rand(1, 500, 20)
    labels = torch.randint(0, 20, (1, 500))
    input_embedder = InputEmbedder()
    h = input_embedder(examples, labels)

    plt.imshow(h.detach().numpy().squeeze(), aspect='auto')
