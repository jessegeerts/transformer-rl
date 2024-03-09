import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset

from experiments.rules_vs_exemplars.config import TransformerConfig
from experiments.rules_vs_exemplars.transformer_classification import TransformerClassifierV2
from experiments.chan_replication.data_generators import SeqGenerator, CompoundDataset, SymbolicDatasetForSampling


class MyIterableDataset(IterableDataset):
    def __init__(self, train_generator, holdout_generator):
        super(MyIterableDataset).__init__()
        self.train_generator = train_generator
        self.holdout_generator = holdout_generator
        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode

    def __iter__(self):
        if self.mode == 'train':
            for item in self.train_generator:
                yield item
        elif self.mode == 'holdout':
            for item in self.holdout_generator:
                yield item
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))


if __name__ == '__main__':
    from definitions import WANDB_KEY
    wandb.login(key=WANDB_KEY)

    # data params
    noise_scale = .1
    lr = 1e-3

    dataset_size = 2**10  # what does this even do? nothing at the moment
    n_common = int(dataset_size * .1)
    n_rare = int(dataset_size * .8)
    n_holdout = int(dataset_size * .1)

    # training params
    num_batches_per_epoch = 50  # it's an iterable dataset, so we need to specify the number of batches
    num_eval_batches = 3
    n_epochs = 200

    # first load the data
    data = CompoundDataset(n_total_classes=dataset_size)
    example_encoding = 'embedding' if data.example_type == 'symbolic' else 'linear'

    num_labels = len(list(data.data.keys()))
    seqgen = SeqGenerator(data, n_rare, n_common, n_holdout, noise_scale=noise_scale)

    # train_generator = seqgen.get_fewshot_seq('rare', 4, 3, labeling='unfixed', randomly_generate_rare=False)
    # for training, we get bursty sequences with examples drawn from the zipfian distribution
    # train_generator = seqgen.get_bursty_seq(13, 4, 3, p_bursty=1., p_bursty_zipfian=1.,
    #                                         labeling_rare='original',
    #                                         labeling_common='original',
    #                                         randomly_generate_rare=False)
    train_generator = seqgen.get_bursty_seq(9, 3, 2, p_bursty=1., p_bursty_zipfian=1.,
                                            labeling_rare='original',
                                            labeling_common='original',
                                            randomly_generate_rare=False)
    # generator = seqgen.get_no_support_seq('rare', 13, labeling='original')

    holdout_generator = seqgen.get_fewshot_seq('holdout', 4, 2, labeling='unfixed', randomly_generate_rare=False)

    dataset = MyIterableDataset(train_generator, holdout_generator)
    config = TransformerConfig(n_heads=1,
                               n_blocks=2,
                               num_classes=num_labels,
                               example_encoding=example_encoding,
                               batch_size=32,
                               log_to_wandb=True)

    if config.log_to_wandb:
        wandb.init(project="FewShotLearning", name='{}'.format(data.example_type))

    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    model = TransformerClassifierV2(config=config)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        dataset.set_mode('train')
        model.train()
        for i, batch in enumerate(dataloader):
            if i > num_batches_per_epoch:
                break
            examples = batch['example']
            if example_encoding == 'embedding':
                examples = examples.long()
            elif example_encoding == 'linear':
                examples = examples.float()
            labels = batch['label'].to(torch.int64)
            is_rare = batch['is_rare']

            # Forward pass
            optimizer.zero_grad()
            query_pred = model(examples, labels, is_training=True)

            # Calculate loss
            train_loss = criterion(query_pred, labels[:, -1])
            train_loss.backward()
            optimizer.step()

            if config.log_to_wandb:
                wandb.log({'training_loss': train_loss.item()})

            i += 1

        # evaluate on holdout classes at the end of each epoch
        model.eval()
        dataset.set_mode('holdout')
        for j, batch in enumerate(dataloader):
            examples = batch['example']
            if example_encoding == 'embedding':
                examples = examples.long()
            elif example_encoding == 'linear':
                examples = examples.float()
            labels = batch['label'].to(torch.int64)
            is_rare = batch['is_rare']

            # Forward pass
            query_pred = model(examples, labels, is_training=False)
            eval_loss = criterion(query_pred, labels[:, -1])
            # calculate accuracy (only considering the possible labels, 0 and 1)
            predicted_labels = torch.argmax(query_pred[:, :2], dim=1)
            icl_accuracy = (predicted_labels == labels[:, -1]).float().mean()
            if config.log_to_wandb:
                wandb.log({'icl_loss': eval_loss.item()})
                wandb.log({'icl_accuracy': icl_accuracy.item()})

            if j > num_eval_batches:
                break


        print(f'Epoch {epoch} done')
        print(f'Loss: {train_loss.item()}')
        print(f'Accuracy: {icl_accuracy.item()}')














