from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from experiments.iwl_icl_classification.data import BurstyTrainingDataset
# from experiments.iwl_icl_classification.model import Transformer
from experiments.rules_vs_exemplars.config import TransformerConfig


def embed_stimuli_and_labels(stimuli, labels):
    B, T, D = stimuli.shape  # batch size, sequence length, stimulus dimension
    embedded_labels = F.one_hot(labels, num_classes=D).float()
    seq_len = (T - 1) * 2 + 1
    start_pos = np.random.choice(P - seq_len + 1, size=B)  # randomly choose a starting position
    positions = [torch.arange(start, start + seq_len) for start in start_pos]
    pos_embeddings = torch.stack(
        [F.one_hot(pos, num_classes=P).float().to(stimuli.device) for pos in positions]).to(stimuli.device)

    # Create interleaved sequence with an extra stimulus at the end
    ctx_stimuli = stimuli[:, :-1, :]  # Exclude the last stimulus (query stimulus)
    h = torch.cat([ctx_stimuli, embedded_labels], dim=1)
    interleave_indices = torch.arange(h.shape[1]).view(-1, h.shape[1] // 2).t().reshape(-1)
    h = h[:, interleave_indices, :].view(B, -1, D)
    h = torch.cat([h, stimuli[:, -1, :].unsqueeze(1)], dim=1)  # Add the query stimulus at the end
    #h += pos_embeddings.unsqueeze(0)
    h = torch.cat([h, pos_embeddings], dim=-1)
    return h


class TransformerEncoderProjection(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderProjection, self).__init__()
        self.transformer = TransformerEncoder(TransformerEncoderLayer(d_model=config.h_dim,
                                                                      nhead=config.n_heads,
                                                                      dim_feedforward=config.widening_factor * config.h_dim,
                                                                      batch_first=True), num_layers=config.n_blocks)
        self.proj_head = nn.Linear(config.h_dim, config.num_classes)

    def forward(self, stimuli, labels):
        h = embed_stimuli_and_labels(stimuli, labels)
        mask = Transformer.generate_square_subsequent_mask(h.shape[1])
        # convert mask to boolean (-inf should go to false and 0 to True)
        mask = mask == 0
        h = self.transformer(h, is_causal=False)  # mask=mask)
        pred = self.proj_head(h)
        return pred[:, -1, :]  # Select the output corresponding to the last stimulus (query stimulus)



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
    n_epochs = 20000
    P = 64  # number of possible positions
    D = 63  # stimulus dimension
    K = 2 ** 9  # number of classes (not to be confused with number of labels. multiple classes can have the same label)
    L = 32  # number of labels
    h_dim = P + D
    mlp_dim = 128
    n_heads = 1  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.
    alpha = 0.  # zipf parameter
    burstiness = 4  # burstiness

    config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=True, n_blocks=2, n_heads=n_heads, batch_size=128,
                               max_T=P, num_classes=L, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim,
                               drop_p=0., within_class_var=.75, alpha=alpha)

    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb.init(project="RulesExemplars",
                   name=f'InbuiltTransformer-B={burstiness},alpha={alpha},K={K},epsilon={config.within_class_var}')

    # data preparation
    # ----------------------------------
    dataset = BurstyTrainingDataset(K=K, B=burstiness, D=D, size=config.batch_size, L=L,
                                    epsilon=config.within_class_var, alpha=config.alpha)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader_icl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader_iwl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # model preparation
    # ----------------------------------

    # model = Transformer(config=config).to(device)
    model = TransformerEncoderProjection(config).to(device)

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
            loss = criterion(y_hat, y[:, -1])
            loss.backward()
            optimizer.step()

            if config.log_to_wandb:
                # log to wandb
                wandb.log({'training_loss': loss.item()})

        # eval loop
        model.eval()
        with torch.no_grad():
            # eval on ICL
            dataset.set_mode('eval_icl')
            for x, y in eval_loader_icl:
                x, y = x.to(device), y.to(device)
                y_hat = model(x, y[:, :-1])
                loss = criterion(y_hat, y[:, -1])
                # calculate accuracy
                predicted_labels = torch.argmax(y_hat, dim=1)
                accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'icl_loss': loss.item()})
                    wandb.log({'icl_accuracy': accuracy.item()})
                break

            # eval on IWL
            dataset.set_mode('eval_iwl')
            for x, y in eval_loader_iwl:
                x, y = x.to(device), y.to(device)
                # eval on IWL
                y_hat = model(x, y[:, :-1])
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
                y_hat = model(x, y[:, :-1])
                loss = criterion(y_hat, y[:, -1])
                # calculate accuracy
                predicted_labels = torch.argmax(y_hat, dim=1)
                accuracy = (predicted_labels == y[:, -1]).float().mean()
                if config.log_to_wandb:
                    wandb.log({'test_loss': loss.item()})
                    wandb.log({'test_accuracy': accuracy.item()})
                break
