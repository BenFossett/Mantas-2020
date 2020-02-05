import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    fig = plt.figure()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('test.png')
    #plt.show()
