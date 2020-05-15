#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

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
import pickle
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser(
    description="Train a CNN for identification with manta ray images",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--checkpoint-path",
    default=Path("src/trained_models/checkpoint_id.pkl"),
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

def image_loader(image):
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def main(args):
    model = torchvision.models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_ftrs, 100)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,100)

    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = pickle.load(open('src/dataset/test_data.pkl', 'rb'))['mantas']
    results = {"mantas": []}

    for manta in dataset:
        image_tensor = image_loader(manta['image'])
        image_class = manta['image_class']
        class_index = manta['class_index']

        m = nn.Softmax(dim=1)
        logits = m(model(image_tensor))[0]
        prediction = logits.argmax(dim=-1).cpu().numpy()
        confidence = logits[class_index].detach().numpy()
        _, ind = logits.topk(100)
        ind = ind.cpu().numpy()
        rank = np.where(ind == class_index)[0]

        sharpness = manta['sharpness']
        environment = manta['environment']
        pattern = manta['pattern']
        pose = manta['pose']

        results['mantas'].append({
            'image_id': manta['image_id'],
            'image_class': image_class,
            'class_index': class_index,
            'prediction': prediction.tolist(),
            'confidence': confidence.tolist(),
            'k-rank': int(rank[0]),
            'sharpness': sharpness,
            'environment': environment,
            'pattern': pattern,
            'pose': pose
        })

    with open('src/results/idcnn_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    main(parser.parse_args())
