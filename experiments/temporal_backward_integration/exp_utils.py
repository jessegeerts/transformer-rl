import torch
from tqdm import tqdm
import wandb


def train(model, data, epochs, criterion, optimizer, pos_embedding_type='learned', logging=True, epochs_offset=0):
    """Training loop.

    :param model: Transformer model
    :param data: Training data
    :param epochs (int): Number of epochs
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param logging: Whether to log to wandb
    :param epochs_offset: Offset for logging
    :return: Trained model
    """
    model.train()

    if isinstance(data, list):
        data_list = data
    else:
        data_list = [data]

    for epoch in tqdm(range(epochs)):
        i_seq = torch.randint(0, len(data_list), (1,)).item()
        data = data_list[i_seq]
        input_data = data[:, :-1]
        target_data = data[:, 1:]
        total_loss = 0
        optimizer.zero_grad()

        start_time = torch.randint(0, 50, (1,)).item()
        output = model.forward_embedded(input_data, start_time, pos_embedding_type=pos_embedding_type)
        loss = criterion(output.squeeze(0), target_data.squeeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if logging:
            wandb.log({'train_loss': total_loss}, step=epoch + epochs_offset)
        if epoch % 500 == 0:
            tqdm.write(f'Epoch {epoch} loss: {total_loss}')
    return model


def eval(model, init_data, n_init_tokens=1, pos_embedding_type='learned'):
    model.eval()

    # Initialize the sequence with the first element of init_data
    init_seq = init_data[:n_init_tokens].unsqueeze(0)  # Shape should be [1, 1, 4]
    predicted_seq = [init_seq[:, i].squeeze().tolist() for i in range(init_seq.shape[1])]

    with torch.no_grad():
        timesteps = torch.arange(10)  # Extended by one for the blank token

        # Using only the first token for initialization
        current_seq = init_seq

        for _ in range(len(init_data)-init_seq.shape[1]):  # Predict up to the length of init_data (-1 because we already have the first token)
            # Append a blank token to the current sequence

            start_time = 0
            output = model.forward_embedded(current_seq, start_time, pos_embedding_type=pos_embedding_type)

            # Extract the prediction for the last token
            last_token_prediction = output[:, -1, :]
            predicted_seq.append(last_token_prediction.squeeze().tolist())

            # append the predicted token to the current sequence
            current_seq = torch.cat([current_seq, last_token_prediction.unsqueeze(1)], dim=1)

    eval_loss = torch.nn.functional.mse_loss(torch.Tensor(predicted_seq)[n_init_tokens:],
                                             init_data.squeeze()[n_init_tokens:])
    return predicted_seq, eval_loss
