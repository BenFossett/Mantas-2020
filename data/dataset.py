import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import json
from PIL import Image
import pickle

class MantaDataset(data.Dataset):
    def __init__(self, data_path):
        self.to_tensor = transforms.ToTensor()
        self.dataset = json.load(open(data_path))['mantas']

    def __getitem__(self, index):
        manta = self.dataset[index]
        image_path = "data/mantas_cropped/" + manta['image_id']
        image = Image.open(image_path)
        image_tensor = self.to_tensor(image)

        resolution = manta['resolution']
        lighting = manta['lighting']
        pattern = manta['pattern']
        pose = manta['pose']
        targets = torch.tensor([resolution, lighting, pattern, pose])

        return image_tensor, targets

    def __len__(self):
        return len(self.dataset)
