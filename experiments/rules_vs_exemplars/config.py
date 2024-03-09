class TransformerConfig:
    def __init__(self, token_dim=4, n_blocks=3, h_dim=64, max_T=100, n_heads=2, drop_p=0.1,
                 pos_embedding_type='learned', epochs_phase1=2000, epochs_phase2=500, lr=1e-3, log_to_wandb=False,
                 num_classes=3, batch_size=4, layer_norm=True, include_mlp=None, mlp_dim=128,
                 loss_threshold=0.01, duration_threshold=100, within_class_var=.1,
                 example_encoding='embedding', widening_factor=4, alpha=1.):
        self.token_dim = token_dim  # token_dim and h_dim are the same if using one-hot embedding directly
        self.num_classes = num_classes
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
        self.batch_size = batch_size
        self.layer_norm = layer_norm
        self.include_mlp = include_mlp
        self.mlp_dim = mlp_dim
        self.loss_threshold = loss_threshold
        self.duration_threshold = duration_threshold
        self.within_class_var = within_class_var
        self.example_encoding = example_encoding
        self.widening_factor = widening_factor
        self.alpha = alpha  # zipf parameter

