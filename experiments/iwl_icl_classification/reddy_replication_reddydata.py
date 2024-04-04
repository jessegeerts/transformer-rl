from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from experiments.rules_vs_exemplars.config import TransformerConfig
from experiments.iwl_icl_classification.model import Transformer as CustomTransformer
from reddy.datasets_v2 import generate_input_seqs, get_mus_label_class
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer


class CustomTransformerNoEmbed(CustomTransformer):
    """Variant of my custom transformer implementation without the separate embedding process (here the stimuli
    and labels have already been embedded).
    """
    def __init__(self, config):
        super().__init__(config=config)

    def forward(self, inputs):
        h = inputs
        for index, block in enumerate(self.blocks):
            h = block(h, index=index)  # Now you pass the index to each block's forward method

        if self.layer_norm:
            h = self.ln(h)
        pred = self.proj_head(h)

        # Select the output corresponding to the last stimulus (query stimulus)
        query_pred = pred[:, -1, :]

        return query_pred


class TorchTransformer(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=config.h_dim,
                                    nhead=config.n_heads,
                                    dim_feedforward=config.widening_factor * config.h_dim,
                                    batch_first=True,
                                    dropout=0.5), num_layers=config.n_blocks)
        self.proj_head = nn.Linear(config.h_dim, config.num_classes)

    def forward(self, inputs):
        h = inputs
        mask = Transformer.generate_square_subsequent_mask(h.shape[1])
        # convert mask to boolean (-inf should go to false and 0 to True)
        mask = mask == 0
        h = self.transformer(h, is_causal=True, mask=mask)
        pred = self.proj_head(h)
        return pred[:, -1, :]  # Select the output corresponding to the last stimulus (query stimulus)


if __name__ == '__main__':
    import wandb
    from tqdm import tqdm

    use_custom_transformer = True
    if use_custom_transformer:
        model_type = 'custom_tf'
    else:
        model_type = 'inbuilt_tf'

    # check if apple gpu is available
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    # Define hyperparameters
    n_iters = 200000
    PosDim = 65  # number of possible positions
    D = 63  # stimulus dimension
    K = 2 ** 9  # number of classes (not to be confused with number of labels. multiple classes can have the same label)
    L = 32  # number of labels
    h_dim = PosDim + D
    mlp_dim = 128
    n_heads = 1  # note: D being odd is a problem for n_heads > 1. the paper uses 1 head.
    alpha = 0.  # zipf parameter
    B = 4  # burstiness
    within_class_var = .2
    pB = 1.
    pC = .75
    N = 8
    Nmax = 32
    no_repeats = True
    S = 10000

    P = 1.0 / (np.arange(1, K + 1) ** alpha)
    P /= np.sum(P)

    config = TransformerConfig(token_dim=D, h_dim=h_dim, log_to_wandb=True, n_blocks=2, n_heads=n_heads, batch_size=128,
                               max_T=PosDim, num_classes=L, include_mlp=[False, True], layer_norm=False, mlp_dim=mlp_dim,
                               drop_p=0., within_class_var=within_class_var, alpha=alpha)

    if config.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb_name = f'TClf-ReddyData-{model_type}-B={B}, alpha={alpha}, K={K}, epsilon={config.within_class_var}, Pb={pB}, Pc={pC}'
        wandb.init(project="RulesExemplars", name=wandb_name)

    # model preparation
    # ----------------------------------
    if use_custom_transformer:
        model = CustomTransformerNoEmbed(config=config).to(device)
    else:
        model = TorchTransformer(config=config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=.01)
    criterion = nn.CrossEntropyLoss()

    # data preparation
    mus_label, mus_class, labels_class = get_mus_label_class(K, L, D)

    test_inputs, test_labels = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = within_class_var, P = P, B = B, p_B = pB, p_C = pC, no_repeats = no_repeats)
    test_inputs_ic, test_labels_ic = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = within_class_var, P = P, B = B, p_B = 1, p_C = 1, no_repeats = no_repeats)
    test_inputs_ic2, test_labels_ic2 = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = within_class_var, P = P, B = B, p_B = 1, p_C = 0, flip_labels = True, no_repeats = no_repeats)
    test_inputs_iw, test_labels_iw = generate_input_seqs(mus_label,mus_class,labels_class,S,N, Nmax,eps = within_class_var, P = P, B = 0, p_B = 0, p_C = 0, no_repeats = no_repeats)

    test_inputs_ic = torch.from_numpy(np.array(test_inputs_ic)).float().to(device)
    test_inputs_iw = torch.from_numpy(np.array(test_inputs_iw)).float().to(device)
    test_labels_ic = torch.from_numpy(np.array(test_labels_ic)).to(device)
    test_labels_iw = torch.from_numpy(np.array(test_labels_iw)).to(device)
    test_inputs = torch.from_numpy(np.array(test_inputs)).float().to(device)
    test_labels = torch.from_numpy(np.array(test_labels)).to(device)
    print_freq = 500  #5000

    # training loop
    # ----------------------------------
    for n in tqdm(range(n_iters)):

        model.train()

        inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label,mus_class,labels_class,
                                                                         config.batch_size,N, Nmax,
                                                                         eps = within_class_var, P=P,
                                                                         B = B, p_B = pB, p_C = pC,
                                                                         output_target_labels = True,
                                                                         no_repeats = no_repeats)
        inputs_batch = torch.from_numpy(inputs_batch).float().to(device)
        labels_batch = torch.from_numpy(np.array(labels_batch)).to(device)
        target_classes = torch.from_numpy(target_classes).to(device)

        optimizer.zero_grad()
        y_hat = model(inputs_batch)
        loss = criterion(y_hat, torch.argmax(labels_batch.float(), dim=-1))
        loss.backward()
        optimizer.step()

        # evaluate on ICL, IWL etc

        if n % print_freq == 0:
            print(f'loss: {loss}')
            if config.log_to_wandb:
                wandb.log({'train_loss': loss.item(), 'iter': n})
            model.eval()

            # evaluate on test set (same dist as training data)
            y_hat = model(test_inputs)
            test_loss = criterion(y_hat, torch.argmax(test_labels.float(), dim=-1))
            # calculate accuracy
            predicted_labels = torch.argmax(y_hat, dim=1)
            test_accuracy = (predicted_labels == torch.argmax(test_labels.float(), dim=-1)).float().mean()
            if config.log_to_wandb:
                wandb.log({'test_loss': test_loss.item(), 'iter': n})
                wandb.log({'test_accuracy': test_accuracy.item(), 'iter': n})

            # evaluate on ICL
            y_hat = model(test_inputs_ic)
            icl_loss = criterion(y_hat, torch.argmax(test_labels_ic.float(), dim=-1))
            # calculate accuracy
            predicted_labels = torch.argmax(y_hat, dim=1)
            icl_accuracy = (predicted_labels == torch.argmax(test_labels_ic.float(), dim=-1)).float().mean()
            if config.log_to_wandb:
                wandb.log({'icl_loss': icl_loss.item(), 'iter': n})
                wandb.log({'icl_accuracy': icl_accuracy.item(), 'iter': n})

            # evaluate on IWL
            y_hat = model(test_inputs_iw)
            iwl_loss = criterion(y_hat, torch.argmax(test_labels_iw.float(), dim=-1))
            # calculate accuracy
            predicted_labels = torch.argmax(y_hat, dim=1)
            iwl_accuracy = (predicted_labels == torch.argmax(test_labels_iw.float(), dim=-1)).float().mean()
            if config.log_to_wandb:
                wandb.log({'iwl_loss': iwl_loss.item(), 'iter': n})
                wandb.log({'iwl_accuracy': iwl_accuracy.item(), 'iter': n})
