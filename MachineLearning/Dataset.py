"""
The Dataset class manages loading and access to a dataset
in the CosmoAI model.
"""

import torch
import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset 
import pandas as pd

class CosmoDataset(Dataset):
    """CosmoAI  dataset."""

    def __init__(self, csvfile, imgdir="."):
        """
        Args:
            csvfile (string): Path to the csv file with annotations.
            imgdir (string): Directory with all the images.
        """
        self.frame = pd.read_csv(csvfile)
        self.imgdir = imgdir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fn = os.path.join(self.imgdir,
                                self.frame.iloc[idx, 1])
        image = io.imread(fn)
        targets = self.frame.iloc[idx, 4:]
        targets = np.array([targets])
        targets = targets.astype('float')

        return  image, targets 
