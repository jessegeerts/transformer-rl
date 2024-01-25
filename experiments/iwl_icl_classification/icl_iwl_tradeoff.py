from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.iwl_icl_classification.data import BurstyTrainingDataset
from experiments.iwl_icl_classification.model import Transformer
from experiments.rules_vs_exemplars.config import TransformerConfig
from experiments.rules_vs_exemplars.transformer_classification import CustomLRScheduler


def run_experiment(config, n_epochs, alpha, epsilon, K, B):
    if config.log_to_wandb:
        wandb.init(project="RulesExemplars", name='iwl-icl-tradeoff-B{}-K{}')

    # data preparation
    # ----------------------------------
    dataset = BurstyTrainingDataset(K=K, D=D, size=config.batch_size, alpha=alpha, epsilon=epsilon, B=B)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader_icl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader_iwl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model preparation
    # ----------------------------------
    model = Transformer(config=config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CustomLRScheduler(optimizer)
    criterion = nn.CrossEntropyLoss()

    # training loop
    # ----------------------------------
    for epoch in range(n_epochs):

        dataset.set_mode('train')
        model.train()

        for x, y in train_loader:  # note: this is kinda bullshit bc we can generate data on the fly
            optimizer.zero_grad()
            y_hat = model(x, y[:, :-1])
            loss = criterion(y_hat, y[:, -1])
            loss.backward()
            optimizer.step()
            scheduler.step()

            if config.log_to_wandb:
                # log to wandb
                wandb.log({'training_loss': loss.item()})
                wandb.log({'learning_rate': scheduler.calculate_lr(scheduler.global_step)})

        # eval loop
        model.eval()
        with torch.no_grad():
            # eval on ICL
            dataset.set_mode('eval_icl')
            for x, y in eval_loader_icl:
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
                y_hat = model(x,  y[:, :-1])
                loss = criterion(y_hat, y[:, -1])
                # calculate accuracy
                predicted_labels = torch.argmax(y_hat, dim=1)
                test_accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'test_loss': loss.item()})
                    wandb.log({'test_accuracy': test_accuracy.item()})
                break

    return icl_accuracy, iwl_accuracy, test_accuracy


if __name__ == '__main__':
    import wandb
    from tqdm import tqdm
    import numpy as np

    wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')

    # data parameters
    alpha = 0.
    L = 32  # number of labels
    D = 63  # stimulus dimension

    # Define network hyperparameters
    n_epochs = 10000
    P = 64  # number of possible positions
    h_dim = P + D
    mlp_dim = 128
    n_heads = 1  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.
    epsilon = 0.1

    config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=True, n_blocks=2, n_heads=n_heads, batch_size=128,
                               max_T=P, num_classes=L, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim,
                               drop_p=0.)

    # we are going to loop over values of B and K and monitor the ICL and IWL accuracies

    B_values = [0, 1, 2, 4]
    K_values = [2**7, 2**8, 2**9, 2**10, 2**11]

    ic_accuracy_matrix = np.zeros((len(B_values), len(K_values)))
    iw_accuracy_matrix = np.zeros((len(B_values), len(K_values)))
    test_accuracy_matrix = np.zeros((len(B_values), len(K_values)))

    for i, B in tqdm(enumerate(B_values)):
        for j, K in tqdm(enumerate(K_values), leave=False):
            config.B = B
            config.K = K
            icl_accuracy, iwl_accuracy, test_accuracy = run_experiment(config, n_epochs, alpha, epsilon, K, B)
            ic_accuracy_matrix[i, j] = icl_accuracy
            iw_accuracy_matrix[i, j] = iwl_accuracy
            test_accuracy_matrix[i, j] = test_accuracy

    # save results
    np.save('icl_accuracy_matrix.npy', ic_accuracy_matrix)
    np.save('iwl_accuracy_matrix.npy', iw_accuracy_matrix)
    np.save('test_accuracy_matrix.npy', test_accuracy_matrix)

    # plot results
    import matplotlib.pyplot as plt
    plt.imshow(ic_accuracy_matrix, vmin=0, vmax=1)
    plt.title('ICL accuracy')
    plt.show()
    plt.imshow(iw_accuracy_matrix, vmin=0, vmax=1)
    plt.title('IWL accuracy')
    plt.show()
    plt.imshow(test_accuracy_matrix, vmin=0, vmax=1)
    plt.title('Test accuracy')
    plt.show()




