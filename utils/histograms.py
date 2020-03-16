import numpy as np
import matplotlib.pyplot as plt

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
    axs[0, 0].hist(res_array, bins=2, range=(0.0, 1.0))
    axs[0, 0].set_title("Resolution")
    axs[0, 1].hist(env_array, bins=2, range=(0.0, 1.0))
    axs[0, 1].set_title("Environment")
    axs[1, 0].hist(pat_array, bins=2, range=(0.0, 1.0))
    axs[1, 0].set_title("Pattern")
    axs[1, 1].hist(pos_array, bins=2, range=(0.0, 1.0))
    axs[1, 1].set_title("Pose")
    plt.savefig("IQANet_Hist.png")
