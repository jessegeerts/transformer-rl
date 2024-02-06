from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.rules_vs_exemplars.data import GeneralizationDataset
from experiments.iwl_icl_classification.model import Transformer
from experiments.rules_vs_exemplars.config import TransformerConfig
import wandb


def run_iwl_experiment():
    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb.init(project="RulesExemplars", name=f'InWeightGeneralization{config.n_blocks}_layers')
    dataset = GeneralizationDataset(1000, subvector_len=subvector_dim)
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
        dataset.set_mode('iw_eval')
        model.eval()
        with torch.no_grad():
            for x, y in train_loader:
                y_hat = model(x, y[:, :-1])
                predicted_labels = torch.argmax(y_hat, dim=1)
                accuracy = (predicted_labels == y[:,
                                                -1]).float().mean()  # assuming rule-based is correct and exemplar-based is wrong
                if config.log_to_wandb:
                    # log to wandb
                    wandb.log({'rule_basedness': accuracy.item()})

        if loss <= config.loss_threshold:
            epochs_below_threshold += 1
            if epochs_below_threshold >= config.duration_threshold:
                print(
                    f"Loss  below threshold for {config.duration_threshold} consecutive epochs. Stopping training."
                    f"The final loss was {loss}.")
                break
        else:
            epochs_below_threshold = 0  # Reset counter if loss goes above threshold


if __name__ == '__main__':
    # Define hyperparameters
    max_epochs = 20000
    P = 64  # number of possible positions
    subvector_dim = 32  # subvector dimension
    h_dim = P + subvector_dim *2
    mlp_dim = 128
    n_heads = 4  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.

    config = TransformerConfig(token_dim=subvector_dim * 2, h_dim=h_dim, log_to_wandb=True, n_blocks=12,
                               n_heads=n_heads, batch_size=32,
                               max_T=P, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim, drop_p=0.,
                               duration_threshold=2)

    run_iwl_experiment()

    print('done')