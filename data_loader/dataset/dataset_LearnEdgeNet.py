import os
import fnmatch

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
        256*256 RGB images

        as input:
        modulo: [0, 1] float, as float32
        modulo_edge: [0, 1] float, as float32

        as target:
        fold_number_edge: binary, as float32
    """

    def __init__(self, data_dir='data', transform=None):
        self.modulo_dir = os.path.join(data_dir, 'modulo')
        self.modulo_edge_dir = os.path.join(data_dir, 'modulo_edge')
        self.fold_number_edge_dir = os.path.join(data_dir, 'fold_number_edge')

        self.names = fnmatch.filter(os.listdir(self.modulo_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        modulo = np.load(os.path.join(self.modulo_dir, self.names[index]))  # positive int, as float32
        modulo_edge = np.load(os.path.join(self.modulo_edge_dir, self.names[index]))  # [0, 1] float, as float32
        fold_number_edge = np.load(os.path.join(self.fold_number_edge_dir, self.names[index]))  # binary, as float32

        name = self.names[index].split('.')[0]
        assert modulo.ndim == 3  # for RGB image

        # (C, H, W)
        modulo = torch.tensor(np.transpose(modulo / np.max(modulo), (2, 0, 1)), dtype=torch.float32)
        modulo_edge = torch.tensor(np.transpose(modulo_edge, (2, 0, 1)), dtype=torch.float32)
        fold_number_edge = torch.tensor(np.transpose(fold_number_edge, (2, 0, 1)), dtype=torch.float32)

        if self.transform:
            modulo = self.transform(modulo)
            modulo_edge = self.transform(modulo_edge)
            fold_number_edge = self.transform(fold_number_edge)

        return {'modulo': modulo, 'modulo_edge': modulo_edge, 'fold_number_edge': fold_number_edge, 'name': name}


class InferDataset(Dataset):
    """
        256*256 RGB images

        modulo: positive int, as float32
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.names = fnmatch.filter(os.listdir(self.data_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # (H, W, C)
        modulo = np.load(os.path.join(self.data_dir, self.names[index]))  # positive int, as float32

        name = self.names[index].split('.')[0]
        assert modulo.ndim == 3  # for RGB image

        # (C, H, W)
        modulo = torch.tensor(np.transpose(modulo, (2, 0, 1)), dtype=torch.float32)

        if self.transform:
            modulo = self.transform(modulo)

        return {'modulo': modulo, 'name': name}

