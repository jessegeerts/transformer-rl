import matplotlib.pyplot as plt
import wandb
import torch


def log_attention_weights(attention_weights, step):
    num_layers, num_heads, _, _ = attention_weights.shape

    for i in range(num_layers):
        for j in range(num_heads):
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.detach().cpu().numpy()
            plt.imshow(attention_weights[i, j])
            plt.title(f'Layer {i+1}, Head {j+1}')
            wandb.log({f'attention_map/layer{i+1}_head{j+1}': plt}, step=step)

            plt.clf()
