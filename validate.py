#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

from data.dataset import MantaDataset
from models.model import CNN
from utils.accuracies import compute_accuracy

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
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Train a CNN for image quality assessment with manta ray images",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--checkpoint-path",
    default=Path("checkpoint.pkl"),
    type=Path,
    help="Provide a file to store checkpoints of the model parameters during training."
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    transform = transforms.ToTensor()

    test_loader = torch.utils.data.DataLoader(
        MantaDataset('data/test_data.pkl'), batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    model = CNN(height=512, width=512, channels=3)
    criterion = nn.BCELoss()

    checkpoint = torch.load("checkpoint.pkl", map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])

    validator = Validator(model, test_loader, criterion, DEVICE)
    validator.validate()


class Validator:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

    def validate(self):
        results = {"labels": [], "logits": []}
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                batch = inputs.to(self.device)
                labels = targets.to(self.device)
                logits = self.model(batch)
                labels = labels.float()
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                results["logits"].extend(list(logits.cpu().numpy()))
                results["labels"].extend(list(labels.cpu().numpy()))

                for j in range(inputs.size()[0]):
                    fig = plt.figure()
                    image = inputs.cpu().data[j]
                    image = image.numpy().transpose((1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = std * image + mean
                    image = np.clip(image, 0, 1)
                    plt.imshow(image)
                    plt.savefig('test.png')
                    import sys; sys.exit(0)

        accuracy, label_accuracies = compute_accuracy(
            np.array(results["labels"]), np.array(results["logits"])
        )
        average_loss = total_loss / len(self.val_loader)

        labels = ["resolution", "lighting", "pattern", "pose"]
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        for i in range(0, len(labels)):
            print("accuracy for " + labels[i] + f": {label_accuracies[i] * 100:2.2f}")

if __name__ == "__main__":
    main(parser.parse_args())
