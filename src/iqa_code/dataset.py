import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import json
from PIL import Image
import pickle

class MantaDataset(data.Dataset):
    def __init__(self, data_path, mode, train):
        self.dataset = pickle.load(open(data_path, 'rb'))['mantas']
        self.mode = mode

        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        manta = self.dataset[index]
        image = Image.fromarray(manta['image'])
        image_tensor = self.transforms(image)

        resolution = manta['resolution']
        environment = manta['environment']
        pattern = manta['pattern']
        pose = manta['pose']
        targets = torch.tensor([resolution, environment, pattern, pose])

        return image_tensor, targets

    def __len__(self):
        return len(self.dataset)
