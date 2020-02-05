import json
import random
from PIL import Image
import numpy as np

data = json.load(open('labels.json'))
mantas = data['mantas']
random.shuffle(mantas)

train_data = {'mantas': []}
test_data = {'mantas': []}

for i, manta in enumerate(mantas):

    if i % 10 < 8:
        train_data['mantas'].append(manta)
    else:
        test_data['mantas'].append(manta)

print(len(train_data['mantas']))
print(len(test_data['mantas']))

with open('train_data.json', 'w') as outfile:
    json.dump(train_data, outfile, indent=4)

with open('test_data.json', 'w') as outfile:
    json.dump(test_data, outfile, indent=4)
