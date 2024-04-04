from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.iwl_icl_classification.data import BurstyTrainingDataset
from experiments.iwl_icl_classification.model import Transformer
from experiments.rules_vs_exemplars.config import TransformerConfig


if __name__ == '__main__':
    import wandb
    from tqdm import tqdm

    # check if apple gpu is available
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = 'cpu'
    print(f'Using device: {device}')

    # Define hyperparameters
    n_epochs = 200000
    P = 64  # number of possible positions
    D = 63  # stimulus dimension
    K = 2 ** 9  # number of classes (not to be confused with number of labels. multiple classes can have the same label)
    L = 32  # number of labels
    h_dim = P + D
    mlp_dim = 128
    n_heads = 1  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.
    alpha = 0.  # zipf parameter
    B = 4  # burstiness
    within_class_var = .2
    Pb = .75
    Pc = .75

    config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=True, n_blocks=2, n_heads=n_heads, batch_size=128,
                               max_T=P, num_classes=L, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim,
                               drop_p=0., within_class_var=within_class_var, alpha=alpha)

    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb_name = f'TClf-B={B}, alpha={alpha}, K={K}, epsilon={config.within_class_var}, Pb={Pb}, Pc={Pc}'
        wandb.init(project="RulesExemplars", name=wandb_name)

    # data preparation
    # ----------------------------------
    dataset = BurstyTrainingDataset(K=K, D=D, size=config.batch_size, L=L, epsilon=config.within_class_var,
                                    alpha=config.alpha, Pb=Pb, Pc=Pc)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader_icl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader_iwl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model preparation
    # ----------------------------------
    model = Transformer(config=config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=.01)
    criterion = nn.CrossEntropyLoss()

    # training loop
    # ----------------------------------
    for epoch in tqdm(range(n_epochs)):

        dataset.set_mode('train')
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x, y[:, :-1])
            train_loss = criterion(y_hat, y[:, -1])
            train_loss.backward()
            optimizer.step()

            if config.log_to_wandb:
                # log to wandb
                wandb.log({'training_loss': train_loss.item()})

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
                ic_accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'icl_loss': loss.item()})
                    wandb.log({'icl_accuracy': ic_accuracy.item()})
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
                accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'iwl_loss': loss.item()})
                    wandb.log({'iwl_accuracy': accuracy.item()})
                break

            # eval on training distribution
            dataset.set_mode('train')
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x,  y[:, :-1])
                loss = criterion(y_hat, y[:, -1])
                # calculate accuracy
                predicted_labels = torch.argmax(y_hat, dim=1)
                accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'test_loss': loss.item()})
                    wandb.log({'test_accuracy': accuracy.item()})
                break

        print('Training loss: {}'.format(train_loss))
        print('IC accuracy: {}'.format(ic_accuracy))
