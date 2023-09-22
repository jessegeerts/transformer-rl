import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from utils import seed_everything
from experiments.temporal_backward_integration.exp_utils import train, eval
import wandb

seed_everything(10)

sweep_config = {
    'method': 'bayes', # can be 'grid', 'random' or 'bayes'
    'metric': {
        'name': 'eval_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'n_blocks': {
            'values': [1, 2, 3]
        },
        'h_dim': {
            'values': [32, 64, 128]
        },
        'n_heads': {
            'values': [1, 2, 4, 8]
        },
        'drop_p': {
            'values': [0.1, 0.2, 0.3]
        },
        'pos_embedding_type': {
            'values': ['sinusoidal', 'param', 'learned', 'onehot']
        },
        'epochs_phase1': {
            'values': [2000, 3000]
        },
        'epochs_phase2': {
            'values': [500, 1000]
        },
        'lr': {
            'values': [1e-3, 1e-2]
        },
    }
}

token_dim = 4
max_T = 100

sweep_id = wandb.sweep(sweep_config, project="SequenceExperiment")
wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')

def run_experiment():

    wandb.init()
    # Define stimuli using one-hot encoding
    S1 = torch.Tensor([1, 0, 0, 0])
    S2 = torch.Tensor([0, 1, 0, 0])
    R = torch.Tensor([0, 0, 0, 1])
    blank = torch.Tensor([0, 0, 0, 0])

    # Sequences
    phase1 = [S2, blank, blank, blank, blank, S1, blank, blank, blank]
    phase2 = [blank, blank, R, blank, blank, S1, blank, blank, blank]

    test_phase = [S2, blank, blank, blank, blank, blank, blank, blank, blank]

    # Define the model, loss, and optimizer
    model = Transformer(token_dim, wandb.config.n_blocks, wandb.config.h_dim, max_T, wandb.config.n_heads, wandb.config.drop_p)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)

    # Train on phase 1
    data_phase1 = torch.stack(phase1).unsqueeze(0)
    model = train(model, data_phase1, wandb.config.epochs_phase1, criterion, optimizer)
    # Test on phase 1
    predicted_seq1, phase1_eval_loss = eval(model, data_phase1.squeeze())
    predicted_seq_1 = np.array(predicted_seq1)

    images_1 = wandb.Image(
        predicted_seq_1,
        caption="Predicted phase 1 sequence"
    )
    wandb.log({"Phase 1 predictions": images_1})


    # train on phase 2
    data_phase2 = torch.stack(phase2).unsqueeze(0)
    model = train(model, data_phase2, wandb.config.epochs_phase2, criterion, optimizer, epochs_offset=wandb.config.epochs_phase1)
    # Test the model on phase 2
    predicted_seq2, phase2_eval_loss = eval(model, data_phase2.squeeze(), n_init_tokens=3)
    predicted_seq_2 = np.array(predicted_seq2)

    images_2 = wandb.Image(
        predicted_seq_2,
        caption="Predicted phase 2 sequence"
    )
    wandb.log({"Phase 2 predictions": images_2})


    eval_loss = phase1_eval_loss + phase2_eval_loss
    wandb.log({'eval_loss': eval_loss})

    # Test the model on test phase
    data_test = torch.stack(test_phase).unsqueeze(0)
    predicted_seq_test, eval_loss_test = eval(model, data_test.squeeze())

    predicted_seq_test = np.array(predicted_seq_test)

    reward_predicted = predicted_seq_test[1:5, 3].mean()

    wandb.log({'reward_predicted': reward_predicted})

    images_test = wandb.Image(
        predicted_seq_test,
        caption="Predicted test sequence"
    )

    wandb.log({"Test phase predictions": images_test})

    wandb.finish()


wandb.agent(sweep_id, function=run_experiment)
