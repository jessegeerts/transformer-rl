from torch.utils.data import Dataset, DataLoader
import torch
from experiments.iwl_icl_classification.data import BurstyTrainingDataset




if __name__ == '__main__':

    # data preparation
    # ----------------------------------
    dataset = BurstyTrainingDataset(K=2 ** 4, D=2)


    # model definition
    # ----------------------------------
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        break
