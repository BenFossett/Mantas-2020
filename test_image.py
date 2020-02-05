import torch
from torch.utils import data
from utils.images import imshow
import numpy as np
import json
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from scipy import misc
import pickle
from torchvision import transforms

transform = transforms.ToTensor()
dataset = pickle.load(open('data/test_data.pkl', 'rb'))['mantas']
manta = dataset[0]
image_path = "data/mantas_cropped/" + manta['image_id']
image = Image.open(image_path)
plt.imshow(image)
plt.show()
image_tensor = transform(image)
imshow(image_tensor)
