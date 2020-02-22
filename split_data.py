import json
import pickle
import random
from PIL import Image
import numpy as np
import argparse

def split_data_iqa(mantas, classes):
    train_data = {'mantas': []}
    test_data = {'mantas': []}

    for i, manta in enumerate(mantas):
        manta['image'] = np.asarray(Image.open('data/mantas_cropped/' + manta['image_id']))
        manta['class_index'] = classes.index(manta['image_class'])

        if i % 10 < 8:
            train_data['mantas'].append(manta)
        else:
            test_data['mantas'].append(manta)

    print(len(train_data['mantas']))
    print(len(test_data['mantas']))

    with open('data/iqa_train_data.pkl', 'wb') as outfile:
        pickle.dump(train_data, outfile)

    with open('data/iqa_test_data.pkl', 'wb') as outfile:
        pickle.dump(test_data, outfile)


def split_data_quality(mantas, threshold):
    train_data = {'mantas': []}
    test_data = {'mantas': []}
    good_quality = {"mantas": []}

    for manta in mantas:
        score = manta["resolution"] + manta["environment"] + manta["pattern"] + manta["pose"]
        score = np.round(score / 4, 2)
        if score > threshold:
            good_quality["mantas"].append(manta)

    previous_class = "None"
    test_count = 0
    for manta in good_quality["mantas"]:
        if previous_class != manta["image_class"]:
            test_count = 0
        if test_count < 2:
            test_data['mantas'].append(manta)
            test_count += 1
        else:
            train_data['mantas'].append(manta)
        previous_class = manta["image_class"]

    print(len(train_data['mantas']))
    print(len(test_data['mantas']))

    with open('data/id_train_data.pkl', 'wb') as outfile:
        pickle.dump(train_data, outfile)

    with open('data/id_test_data.pkl', 'wb') as outfile:
        pickle.dump(test_data, outfile)


def split_data_full(mantas):
    train_data = {'mantas': []}
    test_data = {'mantas': []}

    for i, manta in enumerate(mantas):
        manta['image'] = np.asarray(Image.open('data/mantas_cropped/' + manta['image_id']))
        manta['class_index'] = classes.index(manta['image_class'])

        if i % 10 < 8:
            train_data['mantas'].append(manta)
        else:
            test_data['mantas'].append(manta)

    print(len(train_data['mantas']))
    print(len(test_data['mantas']))

    with open('data/id_train_data.json', 'w') as outfile:
        json.dump(train_data, outfile, indent=4)

    with open('data/id_test_data.json', 'w') as outfile:
        json.dump(test_data, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='What function is the data being split for?', default='iqa')
    args = parser.parse_args()

    dataset = json.load(open('data/labels.json'))

    classes = []
    for manta in dataset['mantas']:
        if manta["image_class"] not in classes:
            classes.append(manta["image_class"])
    print(str(len(classes)) + "mantas")

    if args.mode == "iqa":
        data = json.load(open('data/labels.json'))
        mantas = data['mantas']
        split_data_iqa(mantas, classes)
    elif args.mode == "id-full":
        data = json.load(open('data/manta_quality.json'))
        mantas = data['mantas']
        split_data_full(mantas, classes)
    elif args.mode == "id-qual":
        data = json.load(open('data/manta_quality.json'))
        mantas = data['mantas']
        split_data_quality(mantas, classes, 0.5)
