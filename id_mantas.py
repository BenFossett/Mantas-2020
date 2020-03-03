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
    default=Path("checkpoint_id_qual.pkl"),
    type=Path,
    help="Provide a file to store checkpoints of the model parameters during training."
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def topk_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    maxk = 10
    batch_size = len(labels)
    top_k_accs = np.zeros(maxk)

    for i, pred in enumerate(preds):
        _, ind = pred.topk(maxk)
        for k in range(1, maxk+1):
            k_inds = ind[0:k]
            if labels[i] in k_inds:
                top_k_accs[k-1] += 1

    print(top_k_accs)
    result = top_k_accs / batch_size
    return result

def main(args):
    transform = transforms.ToTensor()

    test_loader = torch.utils.data.DataLoader(
        MantaDataset('data/qual_test_data.pkl', mode="id-qual", train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=8,
        pin_memory=True
    )

    #model = CNN(height=512, width=512, channels=3)
    model = torchvision.models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs, 100)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,100)

    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
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
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_loader):
                batch = inputs.to(self.device)
                labels = targets.to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                results["preds"].extend(list(outputs))
                results["labels"].extend(list(labels.cpu().numpy()))

                #for j in range(inputs.size()[0]):
                #    image = inputs.cpu().data[j]
                #    imshow(image)
                #    import sys; sys.exit(0)

        accuracies = topk_accuracy(
            np.array(results["labels"]), results["preds"]
        )
        average_loss = total_loss / len(self.test_loader)

        print(f"validation loss: {average_loss:.5f}, top-1 accuracy: {accuracies[0] * 100:2.2f}, top-10 accuracy: {accuracies[9] * 100:2.2f}")

        np.savetxt("topk_qual.csv", accuracies, delimiter=",")

if __name__ == "__main__":
    main(parser.parse_args())
