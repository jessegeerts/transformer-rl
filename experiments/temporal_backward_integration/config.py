class TransformerConfig:
    def __init__(self, token_dim=4, n_blocks=3, h_dim=64, max_T=100, n_heads=2, drop_p=0.1,
                 pos_embedding_type='learned', epochs_phase1=2000, epochs_phase2=500, lr=1e-3, log_to_wandb=False):
        valid_pos_embedding_types = ['learned', 'param', 'onehot', 'sinusoidal']
        if pos_embedding_type not in valid_pos_embedding_types:
            raise ValueError(f'pos_embedding_type {pos_embedding_type} not recognized: must be in {valid_pos_embedding_types}.')
        if pos_embedding_type == 'onehot':
            raise NotImplementedError(f'onehot currently not supported.')
        self.token_dim = token_dim  # token_dim and h_dim are the same if using one-hot embedding directly
        self.n_blocks = n_blocks
        self.h_dim = h_dim
        self.max_T = max_T
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.pos_embedding_type = pos_embedding_type
        self.epochs_phase1 = epochs_phase1
        self.epochs_phase2 = epochs_phase2
        self.lr = lr
        self.log_to_wandb = log_to_wandb
        self.seed = 2

