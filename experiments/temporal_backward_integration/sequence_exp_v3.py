"""
TODO:
    - TODO FIRST: JUST TRAIN LONGER ON PHASE 1 until log loss is stable [DONE]
    - TODO: check if position encoding is correct [might have justrun the same in every condition] [DONE]
    - Implement control sequence with random offset in position encoding [DONE]
    - Implement control sequence with different stimuli [DONE]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
import numpy as np
import matplotlib.pyplot as plt
from utils import seed_everything
from experiments.temporal_backward_integration.exp_utils import train, eval
from experiments.temporal_backward_integration.config import TransformerConfig
import wandb

# Define hyperparameters
config = TransformerConfig(pos_embedding_type='learned',
                           n_blocks=1,
                           h_dim=32,
                           n_heads=1,
                           drop_p=0.1,
                           epochs_phase1=40000,
                           epochs_phase2=4000,
                           log_to_wandb=True,
                           lr=1e-4)

seed_everything(config.seed)

if config.log_to_wandb:
    wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')


# Define stimuli using one-hot encoding
S1 = torch.Tensor([1, 0, 0, 0])
S2 = torch.Tensor([0, 1, 0, 0])
S3 = torch.Tensor([0, 0, 1, 0])

R = torch.Tensor([0, 0, 0, 1])
blank = torch.Tensor([0, 0, 0, 0])


for experiment in ['BW', 'ConTI', 'ConBW']:

    if config.log_to_wandb:
        wandb.init(project="SequenceExp", name=experiment + '-' + config.pos_embedding_type)

    # Define the sequences
    test_phase = [S2, blank, blank, blank, blank, blank, blank, blank, blank]
    if experiment == 'BW':
        # Sequences
        phase1 = [S2, blank, blank, blank, blank, S1, blank, blank, blank]
        phase2 = [blank, blank, R, blank, blank, S1, blank, blank, blank]
    elif experiment == 'ConTI':
        # Sequences
        phase1 = [S2, blank, blank, blank, blank, S1, blank, blank, blank]
        phase2 = [blank, blank, R, blank, blank, S3, blank, blank, blank]
    elif experiment == 'ConBW':
        # Sequences
        phase1 = [S2, S1, blank, blank, blank, blank, blank, blank, blank]
        phase2 = [blank, blank, R, blank, blank, S1, blank, blank, blank]

    # Define the model, loss, and optimizer
    model = Transformer(config=config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train on phase 1
    # ----------------------------------
    data_phase1 = torch.stack(phase1).unsqueeze(0)
    model = train(model, data_phase1, config.epochs_phase1, criterion, optimizer, logging=config.log_to_wandb,
                  pos_embedding_type=config.pos_embedding_type)
    # Test on phase 1
    predicted_seq1, phase1_eval_loss = eval(model, data_phase1.squeeze(), pos_embedding_type=config.pos_embedding_type)
    predicted_seq_1 = np.array(predicted_seq1)

    images_1 = wandb.Image(
        predicted_seq_1,
        caption="Predicted phase 1 sequence"
    )
    if config.log_to_wandb:
        wandb.log({"Phase 1 predictions": images_1})

    # train on phase 2
    # ----------------------------------
    data_phase2 = torch.stack(phase2).unsqueeze(0)
    model = train(model, data_phase2, config.epochs_phase2, criterion, optimizer, logging=config.log_to_wandb,
                  epochs_offset=config.epochs_phase1, pos_embedding_type=config.pos_embedding_type)
    # Test the model on phase 2
    predicted_seq2, phase2_eval_loss = eval(model, data_phase2.squeeze(), n_init_tokens=3,
                                            pos_embedding_type=config.pos_embedding_type)
    predicted_seq_2 = np.array(predicted_seq2)

    images_2 = wandb.Image(
        predicted_seq_2,
        caption="Predicted phase 2 sequence"
    )

    eval_loss = phase1_eval_loss + phase2_eval_loss
    if config.log_to_wandb:
        wandb.log({"Phase 2 predictions": images_2,
                   "eval_loss": eval_loss})
    # Test the model on test phase
    # ----------------------------------
    data_test = torch.stack(test_phase).unsqueeze(0)
    predicted_seq_test, eval_loss_test = eval(model, data_test.squeeze(), pos_embedding_type=config.pos_embedding_type)

    predicted_seq_test = np.array(predicted_seq_test)

    reward_predicted = predicted_seq_test[1:5, 3].mean()

    if config.log_to_wandb:
        wandb.log({'reward_predicted': reward_predicted})
        images_test = wandb.Image(
            predicted_seq_test,
            caption="Predicted test sequence"
        )
        wandb.log({"Test phase predictions": images_test})
        wandb.finish()

