import torch
import numpy as np
from typing import Union
import matplotlib.pyplot as plt

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

def create_histograms(logits):
    res_array = []
    env_array = []
    pat_array = []
    pos_array = []

    for i in range(0, len(logits)):
        res_array.append(logits[i][0])
        env_array.append(logits[i][1])
        pat_array.append(logits[i][2])
        pos_array.append(logits[i][3])

    fix, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(res_array, bins=4, range=(0.0, 1.0))
    axs[0, 0].set_title("Resolution")
    axs[0, 1].hist(env_array, bins=4, range=(0.0, 1.0))
    axs[0, 1].set_title("Environment")
    axs[1, 0].hist(pat_array, bins=4, range=(0.0, 1.0))
    axs[1, 0].set_title("Pattern")
    axs[1, 1].hist(pos_array, bins=4, range=(0.0, 1.0))
    axs[1, 1].set_title("Pose")
    plt.savefig("hists.png")
