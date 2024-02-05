from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.rules_vs_exemplars.data import GeneralizationDataset
from experiments.iwl_icl_classification.model import Transformer
from experiments.rules_vs_exemplars.config import TransformerConfig
import wandb


if __name__ == '__main__':
    # Define hyperparameters
    n_epochs = 20000
    P = 64  # number of possible positions
    D = 64  # stimulus dimension
    h_dim = P + D
    mlp_dim = 128
    n_heads = 4  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.

    config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=True, n_blocks=12, n_heads=n_heads, batch_size=32,
                               max_T=P, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim, drop_p=0.)

    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb.init(project="RulesExemplars", name=f'RandomizedIncontextPretraining_{config.n_blocks}_layers')


    dataset = GeneralizationDataset(1000, D=config.token_dim)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = Transformer(config=config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
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

    print('done')