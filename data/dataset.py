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

        if self.mode in ["id-full", "id-qual"]:
            if train:
                self.transforms = transforms.Compose([
                    transforms.Resize(299),
                    transforms.RandomAffine(degrees=(-90, 90), translate=(0.15, 0.15), scale=(0.75, 1.25), shear=(-15, 15)),
                    transforms.RandomPerspective(distortion_scale=0.2),
                    transforms.ColorJitter(brightness=0.25, contrast=0.25, hue=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        manta = self.dataset[index]
        image = Image.fromarray(manta['image'])
        image_tensor = self.transforms(image)

        image_class = manta['image_class']
        class_index = manta['class_index']

        resolution = manta['resolution']
        environment = manta['environment']
        pattern = manta['pattern']
        pose = manta['pose']
        image_quality = torch.tensor([resolution, environment, pattern, pose])

        if self.mode == "iqa":
            targets = image_quality
        else:
            targets = class_index

        return image_tensor, targets

    def __len__(self):
        return len(self.dataset)
