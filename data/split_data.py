import json
import random
import pickle
from PIL import Image
import numpy as np

data = json.load(open('labels.json'))
mantas = data['mantas']
random.shuffle(mantas)

train_data = {'mantas': []}
test_data = {'mantas': []}

for i, manta in enumerate(mantas):
    image_path = "mantas_cropped/" + manta['image_id']
    image = np.asarray(Image.open(image_path))
    manta['image'] = image

    if i % 10 < 8:
        train_data['mantas'].append(manta)
    else:
        test_data['mantas'].append(manta)

print(len(train_data['mantas']))
print(len(test_data['mantas']))

output = open('train_data.pkl', 'wb')
pickle.dump(train_data, output)
output.close()

output = open('test_data.pkl', 'wb')
pickle.dump(test_data, output)
output.close()
