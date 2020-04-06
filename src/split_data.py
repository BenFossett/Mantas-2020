import json
import pickle
import random
from PIL import Image
import numpy as np
import argparse

def split_data(mantas, classes):
    train_data = {'mantas': []}
    test_data = {'mantas': []}

    for i, manta in enumerate(mantas):
        manta['image'] = np.asarray(Image.open('dataset/mantas_cropped/' + manta['image_id']))
        manta['class_index'] = classes.index(manta['image_class'])

        if i % 10 < 8:
            train_data['mantas'].append(manta)
        else:
            test_data['mantas'].append(manta)

    print(len(train_data['mantas']))
    print(len(test_data['mantas']))

    with open('dataset/train_data.pkl', 'wb') as outfile:
        pickle.dump(train_data, outfile)

    with open('manta_id/cnn/dataset/train_data.pkl', 'wb') as outfile:
        pickle.dump(train_data, outfile)

    with open('dataset/test_data.pkl', 'wb') as outfile:
        pickle.dump(test_data, outfile)

    with open('manta_id/cnn/dataset/test_data.pkl', 'wb') as outfile:
        pickle.dump(test_data, outfile)
        

def split_data_folders(mantas):
    train_count = 0
    test_count = 0
    for i, manta in enumerate(mantas):
        image = Image.open('dataset/mantas/' + manta['image_id'])

        if i % 10 < 8:
            image.save('dataset/train_folder/' + str(manta["image_id"]))
            train_count += 1
        else:
            image.save('dataset/test_folder/' + str(manta["image_id"]))
            test_count += 1

    print("Training images: " + str(train_count))
    print("Testing images: " + str(test_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='Where is the data that needs to be split?', default='dataset/quality_labels.json')
    args = parser.parse_args()

    data = json.load(open(args.data_path))

    classes = []
    for manta in data['mantas']:
        if manta["image_class"] not in classes:
            classes.append(manta["image_class"])
    print(str(len(classes)) + "mantas")

    #mantas = data['mantas']
    #split_data_folders(mantas)

    mantas = data['mantas']
    split_data(mantas, classes)
