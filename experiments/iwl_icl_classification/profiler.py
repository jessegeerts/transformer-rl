import time

time_start = time.time()

import wandb
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from experiments.rules_vs_exemplars.config import TransformerConfig
from experiments.iwl_icl_classification.data import BurstyTrainingDataset
from experiments.iwl_icl_classification.model import Transformer

time_end = time.time()

print(f"Time taken for imports: {time_end - time_start}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# data parameters
alpha = 0.
L = 32  # number of labels
D = 63  # stimulus dimension

# Define network hyperparameters
max_epochs = 10
loss_threshold = 0.05
duration_threshold = 10

P = 64  # number of possible positions
h_dim = P + D
mlp_dim = 128
n_heads = 1  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.
epsilon = 0.1

config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=False, n_blocks=2, n_heads=n_heads, batch_size=128,
                           max_T=P, num_classes=L, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim,
                           drop_p=0., loss_threshold=loss_threshold, duration_threshold=duration_threshold)

K = 2**11
B = 0

# data preparation
# ----------------------------------
# note, we set the size of the dataset to be the same as the batch size so that we can generate data on the fly
dataset = BurstyTrainingDataset(K=K, D=config.token_dim, size=config.batch_size, alpha=alpha, epsilon=epsilon, B=B)
train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
eval_loader_icl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
eval_loader_iwl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# model preparation
# ----------------------------------
model = Transformer(config=config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


# measure time taken for data generation (train_loader)

start_time = time.time()
for _ in range(max_epochs):
    for x, y in train_loader:
        pass
total_time = time.time() - start_time
print(f"Time taken for data generation: {total_time}")


# measure time taken for data generation (eval_loader_icl)

start_time = time.time()
for _ in range(max_epochs):
    for x, y in eval_loader_icl:
        pass

total_time = time.time() - start_time
print(f"Time taken for data generation (ICL): {total_time}")

# measure time taken for data generation (eval_loader_iwl)

start_time = time.time()
for _ in range(max_epochs):
    for x, y in eval_loader_iwl:
        pass

total_time = time.time() - start_time
print(f"Time taken for data generation (IWL): {total_time}")

# measure time taken for training


start_time = time.time()
for epoch in range(max_epochs):
    dataset.set_mode('train')
    model.train()

    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x, y[:, :-1])
        loss = criterion(y_hat, y[:, -1])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

total_time = time.time() - start_time
print(f"Time taken for training (including data loading): {total_time}")

