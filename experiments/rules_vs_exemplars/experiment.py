from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.rules_vs_exemplars.data import GeneralizationDataset
from experiments.iwl_icl_classification.model import Transformer
from experiments.rules_vs_exemplars.config import TransformerConfig
import wandb


def exemplar_strategy(stim, labels, query):
    """Exemplar strategy for classification.

    This is a simple strategy that classifies the query stimulus as the label of the most similar stimulus in the
    sequence.
    """
    similarity = query @ stim.T
    max_similar = torch.argmax(similarity)
    max_label = labels[max_similar]
    return max_label


def exemplar_strategy_batch(stim, labels, query):
    """
    Exemplar strategy for classification for a batch of data, with 1D labels.

    Args:
    - stim (Tensor): Stimuli tensor of shape [B, T, dim].
    - labels (Tensor): Labels tensor of shape [B, T].
    - query (Tensor): Query tensor of shape [B, dim].

    Returns:
    - Tensor: The predicted label for each query in the batch.
    """
    # Calculate similarity for each example in the batch
    similarity = torch.bmm(query.unsqueeze(1), stim.transpose(1, 2)).squeeze(1)

    # Find the index of the most similar stimulus for each example in the batch
    max_similar_indices = torch.argmax(similarity, dim=1)

    # Gather the corresponding labels for each batch item
    batch_indices = torch.arange(labels.size(0)).to(labels.device)
    max_labels = labels[batch_indices, max_similar_indices]

    return max_labels


def exemplar_strat_batch(stim, labels, query):
    """
    Exemplar strategy for classification for a batch of data, using Euclidean distance.

    Args:
    - stim (Tensor): Stimuli tensor of shape [B, T, dim].
    - labels (Tensor): Labels tensor of shape [B, T].
    - query (Tensor): Query tensor of shape [B, dim].

    Returns:
    - Tensor: The predicted label for each query in the batch.
    """
    # Reshape stim and query for Euclidean distance computation
    # stim is [B, T, dim], query is [B, dim]
    # Expand query to [B, T, dim] to match stim
    query_expanded = query.unsqueeze(1).expand(-1, stim.size(1), -1)

    # Calculate Euclidean distance
    euclidean_distance = torch.sqrt(torch.sum((query_expanded - stim) ** 2, dim=2))

    # Find the index of the stimulus with minimum distance for each example in the batch
    min_distance_indices = torch.argmin(euclidean_distance, dim=1)

    # Gather the corresponding labels for each batch item
    batch_indices = torch.arange(labels.size(0)).to(labels.device)
    min_labels = labels[batch_indices, min_distance_indices]

    return min_labels


def run_iwl_experiment():
    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb.init(project="RulesExemplars", name=f'InWeightGeneralization{config.n_blocks}_layers')
    dataset = GeneralizationDataset(config.batch_size, subvector_len=subvector_dim)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = Transformer(config=config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    epochs_below_threshold = 0
    for epoch in range(max_epochs):
        # train in-weight learning
        dataset.set_mode('iw_training')
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x, y[:, :-1])
            loss = criterion(y_hat, y[:, -1])
            loss.backward()
            optimizer.step()
            if config.log_to_wandb:
                # log to wandb
                wandb.log({'training_loss': loss.item()})

        # evaluate generalization from in-weight learning
        dataset.set_mode('iw_generalization_test')
        model.eval()
        with torch.no_grad():
            for x, y in train_loader:
                y_hat = model(x, y[:, :-1])
                predicted_labels = torch.argmax(y_hat, dim=1)
                accuracy = (predicted_labels == y[:,
                                                -1]).float().mean()  # assuming rule-based is correct and exemplar-based is wrong
                prop_exemplar = (predicted_labels == 0).sum() / len(predicted_labels)
                prop_rule = (predicted_labels == 1).sum() / len(predicted_labels)
                prop_rand = (predicted_labels == 2).sum() / len(predicted_labels)
                if config.log_to_wandb:
                    # log to wandb
                    wandb.log({'rule_based_accuracy': accuracy.item()})
                    wandb.log({'prop_exemplar': prop_exemplar.item()})
                    wandb.log({'prop_rule': prop_rule.item()})
                    wandb.log({'prop_rand': prop_rand.item()})

        if loss <= config.loss_threshold:
            epochs_below_threshold += 1
            if epochs_below_threshold >= config.duration_threshold:
                print(
                    f"Loss  below threshold for {config.duration_threshold} consecutive epochs. Stopping training."
                    f"The final loss was {loss}.")
                break
        else:
            epochs_below_threshold = 0  # Reset counter if loss goes above threshold
    wandb.finish()


def run_icl_experiment():
    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb.init(project="RulesExemplars", name=f'InContextGeneralization{config.n_blocks}_layers')

    dataset = GeneralizationDataset(1024, subvector_len=subvector_dim)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = Transformer(config=config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    epochs_below_threshold = 0

    exemplar_strat_results = []
    for epoch in range(max_epochs):
        # train in-weight learning
        dataset.set_mode('fewshot')
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            y_hat = model(x, y[:, :-1])
            loss = criterion(y_hat, y[:, -1])
            loss.backward()
            optimizer.step()
            if config.log_to_wandb:
                # log to wandb
                wandb.log({'training_loss': loss.item()})

        # first evaluate in-context learning. is this happening at all
        model.eval()
        dataset.set_mode('icl_eval')
        with torch.no_grad():
            for x, y in train_loader:
                y_hat = model(x, y[:, :-1])
                predicted_labels = torch.argmax(y_hat, dim=1)
                accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    # log to wandb
                    wandb.log({'icl_accuracy': accuracy.item()})
                break

        # then evaluate generalization from in-context learning with the partial exposure paradigm
        dataset.set_mode('icl_generalization_test')
        with torch.no_grad():
            for x, y in train_loader:
                # verify that the exemplar strategy labels about 50% of datapoints as 0 and 50% as 1
                predicted_from_exemplar = exemplar_strategy_batch(x[:, :-1], y[:, :-1], x[:, -1])
                predicted_from_exemplar_euc = exemplar_strat_batch(x[:, :-1], y[:, :-1], x[:, -1])
                exemplar_strat_results.append(predicted_from_exemplar_euc)
                # now see what model does
                y_hat = model(x, y[:, :-1])
                predicted_labels = torch.argmax(y_hat, dim=1)
                accuracy = (predicted_labels == y[:,
                                                -1]).float().mean()
                prop_0 = (predicted_labels == 0).sum() / len(predicted_labels)
                prop_1 = (predicted_labels == 1).sum() / len(predicted_labels)
                prop_rand = (predicted_labels == 2).sum() / len(predicted_labels)
                if config.log_to_wandb:
                    # log to wandb
                    wandb.log({'rule_based_accuracy': accuracy.item()})
                    wandb.log({'prop_0': prop_0.item()})
                    wandb.log({'prop_1': prop_1.item()})
                    wandb.log({'prop_rand': prop_rand.item()})
                break

        if loss <= config.loss_threshold:
            epochs_below_threshold += 1
            if epochs_below_threshold >= config.duration_threshold:
                print(
                    f"Loss  below threshold for {config.duration_threshold} consecutive epochs. Stopping training."
                    f"The final loss was {loss}.")
                break
        else:
            epochs_below_threshold = 0

    wandb.finish()


if __name__ == '__main__':
    # Define hyperparameters
    max_epochs = 20000
    P = 64  # number of possible positions
    subvector_dim = 32  # subvector dimension
    h_dim = P + subvector_dim * 2
    mlp_dim = 128
    n_heads = 4  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.

    config = TransformerConfig(token_dim=subvector_dim * 2, h_dim=h_dim, log_to_wandb=True, n_blocks=12,
                               n_heads=n_heads, batch_size=32,
                               max_T=P, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim, drop_p=0.,
                               duration_threshold=4, within_class_var=0.)

    # first run the in-weight learning experiment
    #run_iwl_experiment()

    # then the in-context learning experiment
    run_icl_experiment()

    print('done')