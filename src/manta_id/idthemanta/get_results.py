import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from PIL import Image
import sys, os

def get_manta(mantas, image_id):
    for manta in mantas:
        if manta['image_id'] == image_id:
            return manta

def main():
    results ={"mantas": []}
    quality_labels = json.load(open('dataset/manta_quality.json'))['mantas']

    for filename in os.listdir('results/raw results/'):
        item = json.load(open('results/raw results/' + filename))
        image_id = filename.split('.')[0] + '.jpg'
        labels = get_manta(quality_labels, image_id)

        image_class = item['testIndividualId']
        preds = item['matchedIndividuals']
        rank = np.where(np.asarray(preds) == image_class)[0]

        if len(rank) == 0:
            k_rank = 100
            confidence = 0
        else:
            k_rank = int(rank)
            confidence = item['individualScores'][k_rank]

        results['mantas'].append({
        'image_id': image_id,
        'image_class': item['testIndividualId'],
        'prediction': item['matchedIndividuals'][0],
        'confidence': confidence,
        'k-rank': k_rank,
        'resolution': labels['resolution'],
        'environment': labels['environment'],
        'pattern': labels['pattern'],
        'pose': labels['pose']
        })

    with open('results/test_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    main()
