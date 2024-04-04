"""TODO: note for this experiment, embedding the stimuli needs to be part of the model.
"""

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from experiments.iwl_icl_classification.data import BurstyTrainingDataset
from experiments.chan_replication.data_generators import SymbolicDatasetForSampling, SeqGenerator
from experiments.chan_replication import dataset_utils
from experiments.rules_vs_exemplars.config import TransformerConfig
from experiments.iwl_icl_classification.model import Transformer
from utils import dotdict
from experiments.rules_vs_exemplars.embedding import InputEmbedder
from torch.optim.lr_scheduler import LambdaLR


class TorchTransformerWithEmbedding(nn.Module):
    def __init__(self, config):
        super(TorchTransformerWithEmbedding, self).__init__()
        self.input_embedder = InputEmbedder(num_classes=config.num_classes,
                                            emb_dim=config.h_dim,
                                            token_dim=1,
                                            example_encoding='embedding',
                                            flatten_superpixels=False,
                                            example_dropout_prob=0.5,
                                            concatenate_labels=False,
                                            use_positional_encodings=True,
                                            positional_dropout_prob=0.1)

        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=config.h_dim,
                                    nhead=config.n_heads,
                                    dim_feedforward=config.widening_factor * config.h_dim,
                                    batch_first=True,
                                    dropout=0.5), num_layers=config.n_blocks)
        self.proj_head = nn.Linear(config.h_dim, config.num_classes)

    def forward(self, stimuli, labels):
        # h = embed_stimuli_and_labels(stimuli, labels)
        h = self.input_embedder(stimuli, labels)
        mask = Transformer.generate_square_subsequent_mask(h.shape[1])
        # convert mask to boolean (-inf should go to false and 0 to True)
        mask = mask == 0
        h = self.transformer(h, is_causal=True, mask=mask)
        pred = self.proj_head(h)
        return pred[:, -1, :]  # Select the output corresponding to the last stimulus (query stimulus)


def _get_ds_seqs(seq_type, seq_config, example_type='symbolic'):
    """Build a TF dataset of sequences for desired sequence type."""

    # Get sequence generator and corresponding config arguments.
    cfg = dotdict(seq_config)
    if seq_type == 'bursty':
        seq_generator = data_generator_factory.get_bursty_seq
        generator_args = (cfg.seq_len, cfg.bursty_shots, cfg.ways, cfg.p_bursty,
                          cfg.p_bursty_common, cfg.p_bursty_zipfian,
                          cfg.non_bursty_type, cfg.labeling_common,
                          cfg.labeling_rare, cfg.randomly_generate_rare,
                          cfg.grouped)
    elif seq_type == 'no_support_common':
        seq_generator = data_generator_factory.get_no_support_seq
        all_unique = False
        generator_args = ('common', cfg.seq_len, all_unique, cfg.labeling_common,
                          cfg.randomly_generate_rare)
    elif seq_type == 'no_support_rare':
        seq_generator = data_generator_factory.get_no_support_seq
        all_unique = False
        generator_args = ('rare', cfg.seq_len, all_unique, cfg.labeling_common,
                          cfg.randomly_generate_rare)
    elif seq_type == 'no_support_zipfian':
        seq_generator = data_generator_factory.get_no_support_seq
        all_unique = False
        generator_args = ('zipfian', cfg.seq_len, all_unique, cfg.labeling_common,
                          cfg.randomly_generate_rare)
    elif seq_type == 'fewshot_rare':
        seq_generator = data_generator_factory.get_fewshot_seq
        generator_args = ('rare', cfg.fs_shots, cfg.ways, 'unfixed',
                          cfg.randomly_generate_rare, cfg.grouped)
    elif seq_type == 'fewshot_common':
        seq_generator = data_generator_factory.get_fewshot_seq
        generator_args = ('common', cfg.fs_shots, cfg.ways, 'unfixed', False,
                          cfg.grouped)
    elif seq_type == 'fewshot_zipfian':
        seq_generator = data_generator_factory.get_fewshot_seq
        generator_args = ('zipfian', cfg.fs_shots, cfg.ways, 'unfixed',
                          cfg.randomly_generate_rare, cfg.grouped)
    elif seq_type == 'fewshot_holdout':
        seq_generator = data_generator_factory.get_fewshot_seq
        generator_args = ('holdout', cfg.fs_shots, cfg.ways, 'unfixed',
                          cfg.randomly_generate_rare, cfg.grouped)
    elif seq_type == 'mixed':
        seq_generator = data_generator_factory.get_mixed_seq
        generator_args = (cfg.fs_shots, cfg.ways, cfg.p_fewshot)
    else:
        raise ValueError('Invalid seq_type: %s' % seq_type)

    # Set the correct example shape and dtype.
    if example_type == 'omniglot':
        example_shape = (cfg.seq_len, 105, 105, 1)
        example_dtype = tf.dtypes.float32
    elif example_type == 'symbolic':
        example_shape = (cfg.seq_len,)
        example_dtype = tf.dtypes.int32
    else:
        raise ValueError('Invalid self.example_type: %s' % example_type)

    # Build the TF dataset from the generator.
    ds_seqs = tf.data.Dataset.from_generator(
        seq_generator,
        args=generator_args,
        output_signature={
            'example':
                tf.TensorSpec(
                    shape=example_shape, dtype=example_dtype),
            'label':
                tf.TensorSpec(shape=(cfg.seq_len,), dtype=tf.dtypes.int32),
            'is_rare':
                tf.TensorSpec(shape=(cfg.seq_len,), dtype=tf.dtypes.int32)
        })

    return ds_seqs


def _build_train_input(batch_size, seq_config, embed_config):
    """See base class."""
    global_batch_size = batch_size

    # Build TF dataset of sequences for desired sequence type.
    ds_seqs = _get_ds_seqs('bursty', seq_config)

    # Batch and prepare data for transformer.
    shuffle_buffer_size = 100
    ds = ds_seqs.batch(global_batch_size)
    ds = dataset_utils.prepare_seqs_for_transformer(
        ds,
        use_constant_labels=False,
        interleave_targets=(not embed_config.concatenate_labels),
        downsample=False
    )
    ds = ds.repeat().shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


def _linear_warmup_and_sqrt_decay(global_step, optim_config):
    """Linear warmup and then an inverse square root decay of learning rate."""
    max_lr = optim_config['max_lr']
    warmup_steps = int(optim_config['warmup_steps'])
    linear_ratio = max_lr / warmup_steps
    decay_ratio = np.power(warmup_steps * 1.0, 0.5) * max_lr
    return np.min(np.array([linear_ratio * global_step, decay_ratio * np.power(global_step, -0.5)]))


if __name__ == '__main__':
    import wandb

    log_to_wandb = True

    symbolic_config = dict(dataset_size=64)
    dataset_for_sampling = SymbolicDatasetForSampling(**symbolic_config)

    tfconfig = TransformerConfig(n_blocks=12, n_heads=8, num_classes=1623, log_to_wandb=True, h_dim=64)

    generator_config = dict(
        n_rare_classes=1600,  # 1623 - 20
        n_common_classes=10,
        n_holdout_classes=10,
        zipf_exponent=1.,
        use_zipf_for_common_rare=False,
        noise_scale=0.,
        preserve_ordering_every_n=None,
    )
    embed_config = dotdict(dict(
        num_classes=None,  # is set later, depending on data config
        emb_dim=64,
        example_encoding='linear',  # 'resnet'/'linear'/'embedding'
        flatten_superpixels=False,  # to flatten resnet outputs
        example_dropout_prob=0.0,
        concatenate_labels=False,
        use_positional_encodings=True,
        positional_dropout_prob=0.0,
    ))
    seq_config = dict(
        seq_len=9,  # NB: can get overridden for some seq types
        fs_shots=4,
        bursty_shots=3,
        ways=2,
        p_bursty=0.9,
        p_bursty_common=0.,
        p_bursty_zipfian=1.,
        p_fewshot=0.1,
        non_bursty_type='zipfian',
        labeling_common='ordered',
        labeling_rare='ordered',
        randomly_generate_rare=False,
        grouped=False,
    )
    optimizer_config = dict(
        name='adam',
        kwargs={},
        # Set up the learning rate schedule.
        max_lr=3e-4,
        warmup_steps=4000,
        clip_level=0.25,
    )
    training_config = dict(
        batch_size=4 * 8,
        learning_rate=1e-4,
        w_interim_predictions=0.,
    )
    model_config = dict()
    data_generator_factory = SeqGenerator(
        dataset_for_sampling,
        **generator_config,
    )

    # data preparation
    # ----------------------------------
    # train_input = _build_train_input(training_config['batch_size'], seq_config, embed_config)
    train_seqs = 'fewshot_rare'
    dataset = iter(tfds.as_numpy(_get_ds_seqs(train_seqs, seq_config).batch(training_config['batch_size'])))

    # model preparation
    # ----------------------------------
    # model = Transformer(config=config).to(device)
    model = TorchTransformerWithEmbedding(tfconfig)

    lr = _linear_warmup_and_sqrt_decay(1, optimizer_config)
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    # scheduler = LambdaLR(optimizer, lambda step: _linear_warmup_and_sqrt_decay(step+1, optimizer_config))
    criterion = nn.CrossEntropyLoss()

    if tfconfig.log_to_wandb:
        wandb.login(key='9f4a033fffce45cce1ee2d5f657d43634a1d2889')
        wandb.init(project="SymbolicDataset",
                   name=f'InbuiltTransformer-{train_seqs}-{tfconfig.n_blocks}blocks-{tfconfig.n_heads}heads-{tfconfig.num_classes}classes',)

    for i in range(500000):

        new_lr = _linear_warmup_and_sqrt_decay(i + 1, optimizer_config)
        # Manually update the optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        model.train()
        batch = next(dataset)
        example = torch.Tensor(batch['example']).long()
        label = torch.Tensor(batch['label']).long()
        pred = model(example, label)
        loss = criterion(pred, label[:, -1])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config['clip_level'])
        optimizer.step()
        # scheduler.step()

        if tfconfig.log_to_wandb:
            wandb.log({'loss': loss.item(), 'step': i})
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr'], 'step': i})

        if i % 1000 == 0:
            # now evaluate on holdout few-shot tasks
            # ----------------------------------
            eval_dataset = iter(tfds.as_numpy(_get_ds_seqs('fewshot_holdout', seq_config).batch(1)))
            model.eval()
            with torch.no_grad():
                accs = []
                for j in range(50):
                    batch = next(eval_dataset)
                    example = torch.Tensor(batch['example']).long()
                    label = torch.Tensor(batch['label']).long()
                    pred = model(example, label)
                    acc = (pred.argmax(-1) == label[:, -1]).float().mean().item()
                    accs.append(acc)

                print(f'Accuracy on holdout few-shot tasks: {np.mean(accs)}')
                if tfconfig.log_to_wandb:
                    wandb.log({f'holdout_fewshot_accuracy': np.mean(accs), 'step': i})