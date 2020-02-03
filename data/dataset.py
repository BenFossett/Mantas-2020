import torch
from torch.utils import data
import numpy as np
import json
from PIL import Image
import pickle


class MantaDataset(data.Dataset):
    def __init__(self, data_path):
        self.dataset = pickle.load(open(data_path, 'rb'))['mantas']

    def __getitem__(self, index):
        manta = self.dataset[index]
        image = manta['image']
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))

        resolution = manta['resolution']
        lighting = manta['lighting']
        pattern = manta['pattern']
        pose = manta['pose']
        targets = torch.tensor([resolution, lighting, pattern, pose])

        return image, targets

    def __len__(self):
        return len(self.dataset)
