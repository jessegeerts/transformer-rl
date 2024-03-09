import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import math

from experiments.rules_vs_exemplars.rules_exemplars import SequenceDataGen
from transformer import Transformer
from experiments.rules_vs_exemplars.config import TransformerConfig
from experiments.rules_vs_exemplars.embedding import InputEmbedder

# Define hyperparameters
# note that token_dim and h_dim are the same for now but they don't have to be at all

# From paper (though note that Reddy (2024) shows ICL with 2 layers attention-only):
# Num layers: 12
# Embedding size: 64
# Optimizer: Adam
# Batch size: 32
# Learning rate schedule: Linear warmup and square root decay, described as
# min(3e-4 / 4000 * global_step, power(4000, 0.5) * 3e-4 * power(global_step, -0.5))


P = 64  # number of possible positions
D = 64  # stimulus dimension
h_dim = P + D

config = TransformerConfig(token_dim=64, h_dim=h_dim, log_to_wandb=True, n_blocks=12, n_heads=4, batch_size=1,
                           max_T=64)
n_reps = 4
n_epochs = 200000  # i.e. number of sequences


class CustomLRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.global_step = 0

    def step(self):
        self.global_step += 1
        lr = self.calculate_lr(self.global_step)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_lr(self, global_step):
        global_step = max(global_step, 1)

        base_lr = 3e-4
        warmup_steps = 4000
        lr_warmup = base_lr / warmup_steps * global_step
        lr_decay = math.sqrt(warmup_steps) * base_lr * math.pow(global_step, -0.5)
        return min(lr_warmup, lr_decay)


class TransformerClassifier(Transformer):
    def __init__(self, token_dim=None, n_blocks=None, h_dim=None, max_T=None, n_heads=None, drop_p=None, config=None):
        super().__init__(token_dim=token_dim, n_blocks=n_blocks, h_dim=h_dim, max_T=max_T, n_heads=n_heads,
                         drop_p=drop_p, config=config)
        self.label_embedding = nn.Embedding(config.num_classes, config.h_dim)  # embed labels into stimulus space
        self.pos_embedding = nn.Embedding(config.max_T, config.h_dim)
        self.proj_head = nn.Linear(config.h_dim, config.num_classes)
        self.P = config.max_T  # number of possible positions

    def forward(self, stimuli, labels):
        B, T, D = stimuli.shape  # batch size, sequence length, stimulus dimension
        # (note, sequence length includes the query stimulus)

        # Embed stimuli  todo: try no embedding
        stimuli = self.proj_stim(stimuli)
        # Embed labels  todo: try one-hot embedding for labels
        # embedded_labels = self.label_embedding(labels)
        embedded_labels = F.one_hot(labels, num_classes=D).float()
        # Embed positions
        # make positions random to learn translation-invariant computation
        # todo: try appending position embeddings to stimuli instead of adding them
        # todo: random start position are now the same for all sequences in the batch. Try making them different.
        seq_len = (T - 1) * 2 + 1
        start_pos = np.random.choice(self.P - seq_len + 1)  # randomly choose a starting position
        positions = torch.arange(start_pos, start_pos + seq_len)
        pos_embeddings = F.one_hot(positions, num_classes=self.P).float()

        # Create interleaved sequence with an extra stimulus at the end  todo: check if this is correct
        ctx_stimuli = stimuli[:, :-1, :]  # Exclude the last stimulus (query stimulus)
        h = torch.cat([ctx_stimuli, embedded_labels], dim=1)
        interleave_indices = torch.arange(h.shape[1]).view(-1, h.shape[1]//2).t().reshape(-1)
        h = h[:, interleave_indices, :].view(B, -1, D)
        h = torch.cat([h, stimuli[:, -1, :].unsqueeze(1)], dim=1)  # Add the query stimulus at the end
        # h += pos_embeddings.unsqueeze(0)
        h = torch.cat([h, pos_embeddings.unsqueeze(0).expand(B, seq_len, self.P)], dim=-1)
        # Transformer and prediction
        h = self.ln(self.transformer(h))
        pred = self.proj_head(h)

        # Select the output corresponding to the last stimulus (query stimulus)
        query_pred = pred[:, -1, :]

        return query_pred


class TransformerClassifierV2(Transformer):
    def __init__(self, token_dim=None, n_blocks=None, h_dim=None, max_T=None, n_heads=None, drop_p=None,
                 config=None):
        super().__init__(token_dim=token_dim, n_blocks=n_blocks, h_dim=h_dim, max_T=max_T, n_heads=n_heads,
                         drop_p=drop_p, config=config)
        self._input_embedder = InputEmbedder(num_classes=config.num_classes, emb_dim=config.h_dim,
                                             example_encoding=config.example_encoding)
        self.proj_head = nn.Linear(config.h_dim, config.num_classes)
        self.P = config.max_T  # number of possible positions

    def forward(self, stimuli, labels, is_training=True):

        h = self._input_embedder(stimuli, labels, is_training)

        # Transformer and prediction
        h = self.ln(self.transformer(h))
        pred = self.proj_head(h)

        # Select the output corresponding to the last stimulus (query stimulus)
        query_pred = pred[:, -1, :]

        return query_pred


def exemplar_strategy(stim, labels, query):
    """Exemplar strategy for classification.

    This is a simple strategy that classifies the query stimulus as the label of the most similar stimulus in the
    sequence.
    """
    similarity = query.dot(stim.T)
    max_similar = np.argmax(similarity)
    max_label = labels[max_similar]
    return max_label


if __name__ == '__main__':

    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb.init(project="RulesExemplars", name='TransformerClassification-AppendedPositions')

    # data preparation
    # ----------------------------------
    generator = SequenceDataGen(covariance_scale=0.1)
    # stim_sequence, stim_names, labels_sequence, query_stimuli, query_labels, query_names = \
    #     generator.generate_stimuli_few_shot(n_reps=n_reps, n_trials=n_trials, return_stim_names=True)


    # model preparation
    # ----------------------------------
    model = TransformerClassifier(config=config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CustomLRScheduler(optimizer)
    criterion = nn.CrossEntropyLoss()

    # training
    # ----------------------------------
    for epoch in range(n_epochs):

        # generate data
        stim_sequence, labels_sequence, query_stimuli, query_labels = \
            generator.generate_stimuli_few_shot_random_context(n_reps=4, batch_size=config.batch_size)

        # assert that exemplar strategy works
        for seq in range(config.batch_size):
            pred = exemplar_strategy(stim_sequence[seq], labels_sequence[seq], query_stimuli[seq])
            target = query_labels[seq]
            assert pred == target

        # convert to torch tensors
        stimuli_sequence = torch.from_numpy(stim_sequence).float()
        labels_sequence = torch.from_numpy(labels_sequence).long()
        query_stimuli = torch.from_numpy(query_stimuli).float()
        query_labels = torch.from_numpy(query_labels).long()

        # append query stimulus to stimuli_sequence
        stimuli_sequence = torch.cat([stimuli_sequence, query_stimuli.unsqueeze(1)], dim=1)

        batch_size, n_ctx_stim, stim_dim = stimuli_sequence.shape

        # Compute global step (assuming 1 step per batch)
        global_step = epoch  # * len(train_loader) + batch_idx

        # Compute learning rate
        scheduler.step()

        # training step
        optimizer.zero_grad()

        pred = model(stimuli_sequence, labels_sequence)
        loss = criterion(pred.squeeze(), query_labels.squeeze())

        if epoch % 100 == 0:
            current_lr = scheduler.calculate_lr(scheduler.global_step)
            print(f"Epoch: {epoch}, Step: {epoch}, Learning Rate: {current_lr}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        if config.log_to_wandb:
            # log to wandb
            wandb.log({'loss': loss.item()})
            wandb.log({'learning_rate': scheduler.calculate_lr(scheduler.global_step)})
