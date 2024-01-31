"""This is a replication of figure 2 in Reddy et al. 2021. We are going to loop over values of B and K and monitor the
ICL and IWL accuracies.
"""

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.iwl_icl_classification.data import BurstyTrainingDataset
from experiments.iwl_icl_classification.model import Transformer
from experiments.iwl_icl_classification.config import TransformerConfig


def run_experiment(config, max_epochs, alpha, epsilon, K, B):
    if config.log_to_wandb:
        wandb.init(project="RulesExemplars", name='iwl-icl-tradeoff-B{}-K{}'.format(B, K))

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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # training loop
    # ----------------------------------
    epochs_below_threshold = 0
    train_iter = 0
    for epoch in range(max_epochs):
        print(f'\rEpoch {epoch}', end="")

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

            if config.log_to_wandb:
                # log to wandb
                wandb.log({'training_loss': loss.item()})
            train_iter += 1

        avg_loss = total_loss / len(train_loader)
        if config.log_to_wandb:
            wandb.log({'avg_training_loss': avg_loss})

        # eval loop
        model.eval()
        with torch.no_grad():
            # eval on ICL
            dataset.set_mode('eval_icl')
            for x, y in eval_loader_icl:
                x, y = x.to(device), y.to(device)
                y_hat = model(x,  y[:, :-1])
                loss = criterion(y_hat, y[:, -1])
                # calculate accuracy
                predicted_labels = torch.argmax(y_hat, dim=1)
                icl_accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'icl_loss': loss.item()})
                    wandb.log({'icl_accuracy': icl_accuracy.item()})
                break

            # eval on IWL
            dataset.set_mode('eval_iwl')
            for x, y in eval_loader_iwl:
                x, y = x.to(device), y.to(device)
                # eval on IWL
                y_hat = model(x,  y[:, :-1])
                loss = criterion(y_hat, y[:, -1])
                # calculate accuracy
                predicted_labels = torch.argmax(y_hat, dim=1)
                iwl_accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'iwl_loss': loss.item()})
                    wandb.log({'iwl_accuracy': iwl_accuracy.item()})
                break

            # eval on training distribution
            dataset.set_mode('train')
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x,  y[:, :-1])
                loss = criterion(y_hat, y[:, -1])
                # calculate accuracy
                predicted_labels = torch.argmax(y_hat, dim=1)
                test_accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'test_loss': loss.item()})
                    wandb.log({'test_accuracy': test_accuracy.item()})
                break

        if avg_loss <= config.loss_threshold:
            epochs_below_threshold += 1
            if epochs_below_threshold >= config.duration_threshold:
                print(
                    f"Loss  below threshold for {config.duration_threshold} consecutive epochs. Stopping training."
                    f"The final loss was {avg_loss}.")
                break
        else:
            epochs_below_threshold = 0  # Reset counter if loss goes above threshold

    if config.log_to_wandb:
        wandb.finish()
    return icl_accuracy, iwl_accuracy, test_accuracy


if __name__ == '__main__':
    import wandb
    from tqdm import tqdm
    import numpy as np
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt

    wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data parameters
    alpha = 0.  # zipf exponent
    L = 32  # number of labels
    D = 63  # stimulus dimension

    # Define network hyperparameters
    max_epochs = 50000
    loss_threshold = 0.05
    duration_threshold = 10

    P = 64  # number of possible positions
    h_dim = P + D
    mlp_dim = 128
    n_heads = 1  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.
    epsilon = 0.1

    config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=True, n_blocks=2, n_heads=n_heads, batch_size=128,
                               max_T=P, num_classes=L, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim,
                               drop_p=0., loss_threshold=loss_threshold, duration_threshold=duration_threshold)

    # we are going to loop over values of B and K and monitor the ICL and IWL accuracies

    B_values = [0, 1, 2, 4]
    K_values = [2**7, 2**8, 2**9, 2**10, 2**11]

    if os.path.exists('icl_accuracy_matrix.npy'):
        ic_accuracy_matrix = np.load('icl_accuracy_matrix.npy')
    else:
        ic_accuracy_matrix = np.zeros((len(B_values), len(K_values)))

    if os.path.exists('iwl_accuracy_matrix.npy'):
        iw_accuracy_matrix = np.load('iwl_accuracy_matrix.npy')
    else:
        iw_accuracy_matrix = np.zeros((len(B_values), len(K_values)))

    if os.path.exists('test_accuracy_matrix.npy'):
        test_accuracy_matrix = np.load('test_accuracy_matrix.npy')
    else:
        test_accuracy_matrix = np.zeros((len(B_values), len(K_values)))

    for i, B in tqdm(enumerate(B_values)):
        for j, K in tqdm(enumerate(K_values), leave=False):
            print('-' * 50)
            print('B = {}, K = {}'.format(B, K))
            # Check if this iteration has already been completed
            if ic_accuracy_matrix[i, j] != 0 and iw_accuracy_matrix[i, j] != 0 and test_accuracy_matrix[i, j] != 0:
                continue

            config.B = B
            config.K = K
            icl_accuracy, iwl_accuracy, test_accuracy = run_experiment(config, max_epochs, alpha, epsilon, K, B)

            ic_accuracy_matrix[i, j] = icl_accuracy
            iw_accuracy_matrix[i, j] = iwl_accuracy
            test_accuracy_matrix[i, j] = test_accuracy

            # Save updated matrices after each iteration
            np.save('icl_accuracy_matrix.npy', ic_accuracy_matrix)
            np.save('iwl_accuracy_matrix.npy', iw_accuracy_matrix)
            np.save('test_accuracy_matrix.npy', test_accuracy_matrix)

    # plot results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title('IWL')
    sns.heatmap(iw_accuracy_matrix, annot=True, fmt='.2f', cmap='inferno', xticklabels=K_values, yticklabels=B_values, ax=ax[0])
    ax[1].set_title('ICL')
    sns.heatmap(ic_accuracy_matrix, annot=True, fmt='.2f', cmap='inferno', xticklabels=K_values, yticklabels=B_values, ax=ax[1])
    ax[2].set_title('Test')
    sns.heatmap(test_accuracy_matrix, annot=True, fmt='.2f', cmap='inferno', xticklabels=K_values, yticklabels=B_values, ax=ax[2])
    plt.savefig('accuracy_matrix.png')
