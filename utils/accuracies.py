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
    label_sums = np.zeros(num_labels)
    label_accuracies = np.zeros(num_labels)

    for i in range(0, batch_size):
        image_labels = labels[i]
        image_logits = logits[i]
        diff = image_labels - image_logits
        total_correct = 0

        for j in range(0, num_labels):
            result = abs(diff[j]) <= 0.5
            total_correct += result
            label_sums[j] += result

        image_accuracy = total_correct / num_labels
        sum += image_accuracy

    batch_accuracy = sum / batch_size
    label_accuracies = np.divide(label_sums, batch_size)
    return batch_accuracy, label_accuracies
