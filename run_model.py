#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

from data.dataset import MantaDataset
from utils.images import imshow

import torch
import torch.backends.cudnn
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
import json
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser(
    description="Train a CNN for image quality assessment with manta ray images",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--checkpoint-path",
    default=Path("checkpoint_finetune.pkl"),
    type=Path,
    help="Provide a file to store checkpoints of the model parameters during training."
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def image_loader(image_name):
    to_tensor = transforms.ToTensor()
    image = Image.open(image_name)
    image = to_tensor(image)
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def main(args):
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 4),
        nn.Sigmoid())

    checkpoint = torch.load("checkpoint_finetune.pkl", map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = json.load(open("data/labels.json"))['mantas']
    new_dataset = {"mantas": []}

    classes = []
    for manta in dataset:
        if manta["image_class"] not in classes:
            classes.append(manta["image_class"])
    print(str(len(classes)) + "mantas")

    for manta in dataset:
        image_path = "data/mantas_cropped/" + manta['image_id']
        image_tensor = image_loader(image_path)
        image_class = manta['image_class']
        class_index = classes.index(image_class)

        prediction = model(image_tensor)

        new_dataset['mantas'].append({
            'image_id': manta['image_id'],
            'image_class': image_class,
            'class_index': class_index,
            'resolution': np.round(prediction[0][0].item(), 2),
            'environment': np.round(prediction[0][1].item(), 2),
            'pattern': np.round(prediction[0][2].item(), 2),
            'pose': np.round(prediction[0][3].item(), 2)
        })

    with open('data/manta_quality.json', 'w') as outfile:
        json.dump(new_dataset, outfile, indent=4)

if __name__ == "__main__":
    main(parser.parse_args())
