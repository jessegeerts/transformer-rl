import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time

from reddy_replication_torch.config import config
from reddy_replication_torch.model import Transformer
from reddy_replication_torch.inbuilt_model import TorchTransformer
from reddy.datasets_v2 import *
from definitions import WANDB_KEY

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

## get parameters from config
# data parameters
custom_model = True
S = config.data.S
K = config.data.K  # number of classes
L = config.data.L  # number of labels
D = config.data.D  # dimension of inputs
alpha = config.data.alpha  # zipf exponent
eps = config.data.eps  # within-class variance
# sequence parameters
N = config.seq.N
B = config.seq.B
pB = config.seq.pB
pC = config.seq.pC
Nmax = 32  # this is fixed.

# determine the frequency of different classes
p_class = 1.0 / (np.arange(1, K + 1) ** alpha)
p_class /= np.sum(p_class)


if custom_model:
    mod_name = 'custom'
else:
    mod_name = 'inbuilt'

ln = config.model.apply_ln if mod_name == 'custom' else True

experiment_name = 'I{}_K{}_N{}_L{}_D{}_a{}_B{}_pB{}_pC{}_eps{}_lr{}_drop{}_{}_ln{}_wDecay{}'.format(
    config.train.niters,
    config.data.K,
    config.seq.N,
    config.data.L,
    config.data.D,
    config.data.alpha,
    config.seq.B,
    config.seq.pB,
    config.seq.pC,
    config.data.eps,
    config.train.learning_rate,
    config.model.drop_p,
    mod_name,
    ln,
    config.train.w_decay
)
config.model.out_dim = config.data.L
print(experiment_name)

if config.log_to_wandb:
    wandb.login(key=WANDB_KEY)
    wandb.init(project="IWvsIClearning", name=experiment_name)


# Loading datasets
mus_label, mus_class, labels_class = get_mus_label_class(K, L, D)
test_inputs, test_labels = generate_input_seqs(mus_label, mus_class, labels_class, S, N, Nmax, eps=eps, P=p_class, B=B,
                                               p_B=pB, p_C=pC, no_repeats=True)
test_inputs_ic, test_labels_ic = generate_input_seqs(mus_label, mus_class, labels_class, S, N, Nmax, eps=eps, P=p_class,
                                                     B=B, p_B=1, p_C=1, no_repeats=True)
test_inputs_ic2, test_labels_ic2 = generate_input_seqs(mus_label, mus_class, labels_class, S, N, Nmax, eps=eps,
                                                       P=p_class, B=B, p_B=1, p_C=0, flip_labels=True, no_repeats=True)
test_inputs_iw, test_labels_iw = generate_input_seqs(mus_label, mus_class, labels_class, S, N, Nmax, eps=eps, P=p_class,
                                                     B=0, p_B=0, p_C=0, no_repeats=True)

# cast to torch tensor
test_inputs_ic = torch.from_numpy(np.array(test_inputs_ic)).float().to(device)
test_inputs_iw = torch.from_numpy(np.array(test_inputs_iw)).float().to(device)
test_labels_ic = torch.from_numpy(np.array(test_labels_ic)).to(device)
test_labels_iw = torch.from_numpy(np.array(test_labels_iw)).to(device)
test_inputs = torch.from_numpy(np.array(test_inputs)).float().to(device)
test_labels = torch.from_numpy(np.array(test_labels)).to(device)

# initialize model, optimizer, loss fn
if custom_model:
    model = Transformer(config=config.model).to(device)   # my custom transformer encoder
else:
    model = TorchTransformer(config=config.model).to(device)  # pytorch transformer encoder
optimizer = optim.SGD(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.w_decay)
criterion = nn.CrossEntropyLoss()


def eval_loss_and_accuracy(mod, inputs, labels):
    y_hat = mod(inputs)
    loss = criterion(y_hat, torch.argmax(labels.float(), dim=-1))
    predicted_labels = torch.argmax(y_hat, dim=1)
    accuracy = (predicted_labels == torch.argmax(labels.float(), dim=-1)).float().mean()
    return loss, accuracy


# training loop
for n in range(config.train.niters):
    model.train()

    # load in a batch of data
    # dataload_start = time.time()
    inputs_batch, labels_batch, target_classes = generate_input_seqs(mus_label, mus_class, labels_class,
                                                                     config.train.batch_size, N, Nmax,
                                                                     eps=eps, P=p_class, B=B, p_B=pB, p_C=pC,
                                                                     output_target_labels=True, no_repeats=True)
    # cast to torch tensor (TODO: there's gotta be a better way to do this)
    inputs_batch = torch.from_numpy(inputs_batch).float().to(device)
    labels_batch = torch.from_numpy(np.array(labels_batch)).to(device)
    # dataload_end = time.time()
    # print(f'time to load data: {dataload_end-dataload_start}')

    optimizer.zero_grad()
    # forward_pass_start = time.time()
    y_hat = model(inputs_batch)
    # forward_pass_end = time.time()
    # print(f'time taken for forward pass: {forward_pass_end-forward_pass_start}')


    # optimizer_start = time.time()
    loss = criterion(y_hat, torch.argmax(labels_batch.float(), dim=-1))
    loss.backward()
    optimizer.step()
    # optimizer_end = time.time()
    # print(f'time taken for backward pass: {optimizer_end-optimizer_start}')

    # evaluate on ICL, IWL etc

    if n % config.logging_interval == 0:
        model.eval()
        with torch.no_grad():
            if config.log_to_wandb:
                wandb.log({'train_loss': loss.item(), 'iter': n})

            # evaluate on test set (same dist as training data)
            test_loss, test_accuracy = eval_loss_and_accuracy(model, test_inputs, test_labels)
            if config.log_to_wandb:
                wandb.log({'test_loss': test_loss.item(), 'iter': n})
                wandb.log({'test_accuracy': test_accuracy.item(), 'iter': n})

            # evaluate on ICL
            icl_loss, icl_accuracy = eval_loss_and_accuracy(model, test_inputs_ic, test_labels_ic)
            if config.log_to_wandb:
                wandb.log({'icl_loss': icl_loss.item(), 'iter': n})
                wandb.log({'icl_accuracy': icl_accuracy.item(), 'iter': n})

            # evaluate on IWL
            iwl_loss, iwl_accuracy = eval_loss_and_accuracy(model, test_inputs_iw, test_labels_iw)
            if config.log_to_wandb:
                wandb.log({'iwl_loss': iwl_loss.item(), 'iter': n})
                wandb.log({'iwl_accuracy': iwl_accuracy.item(), 'iter': n})

            print(f'iter {n}, loss: {loss}, ic_accuracy: {icl_accuracy}, iw_accuracy: {iwl_accuracy}')
