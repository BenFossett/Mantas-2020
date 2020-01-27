import torch
from torch.utils import data
import numpy as np
import json
from PIL import Image
from scipy import misc


class MantaDataset(data.Dataset):
    def __init__(self, dataset_path):
        self.labels = json.load(open(data.json))

    def __getitem__(self, index):
        manta = self.labels[index]
        image_path = "mantas_cropped/" + manta['image_id']
        image = misc.imread(path)
        image = Image.fromarray(image)

        classID = manta['image_class']
        resolution = manta['resolution']
        lighting = manta['lighting']
        pattern = manta['pattern']
        pose = manta['pose']
        score = resolution + lighting + pattern + pose
        return image, classID, score

    def __len__(self):
        return len(self.dataset)
