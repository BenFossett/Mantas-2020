import torch
import numpy as np
from typing import Union

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], logits: Union[torch.tensor, np.ndarray]
) -> float:
    assert len(labels) == len(logits)

    sum = 0
    batch_size = len(labels)
    num_labels = len(labels[0])

    for i in range(0, batch_size):
        image_labels = labels[i]
        image_logits = logits[i]
        diff = image_labels - image_logits
        num_correct = len([j for j in diff if abs(j) < 0.5])
        image_accuracy = num_correct / num_labels
        sum += image_accuracy

    batch_accuracy = sum / batch_size
    return batch_accuracy
