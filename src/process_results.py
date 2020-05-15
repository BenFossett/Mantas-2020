import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--accs', help='Get per-label thresholded accuracies', action='store_true')
parser.add_argument('--topk', help='Top-K Accuracy (to be used with --accs argument)', type=int, default=1)
parser.add_argument('--hists', help='Get histograms for each label', action='store_true')
parser.add_argument('--vis', help='Visualise images with quality labels', action='store_true')
parser.add_argument('--best', help='Find best combination of quality labels', action='store_true')
parser.add_argument('--confidence', help='Generate confidence histograms', action='store_true')

def get_manta(image_id, mantas):
    for manta in mantas:
        if manta['image_id'] == image_id:
            return manta

def find_best_combo(data, topk):
    best_acc = 0
    best_sharp = 0
    best_env = 0
    best_patt = 0
    best_pose = 0

    for x in range(0, 10):
        curr_sharp = x * 0.1
        for y in range(0, 10):
            curr_env = y * 0.1
            for i in range(0, 10):
                curr_patt = i * 0.1
                for j in range(0, 10):
                    curr_pose = j * 0.1
                    num_mantas = 0
                    num_correct = 0

                    for manta in data:
                        sharpness = manta['sharpness']
                        environment = manta['environment']
                        pattern = manta['pattern']
                        pose = manta['pose']
                        rank = manta['k-rank']

                        if (sharpness >= curr_sharp) and (environment >= curr_env) and (pattern >= curr_patt) and (pose >= curr_pose):
                            num_mantas += 1

                            if rank < topk:
                                num_correct += 1

                    accuracy = num_correct / num_mantas
                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_sharp = curr_sharp
                        best_env = curr_env
                        best_patt = curr_patt
                        best_pose = curr_pose
                        best_num = num_mantas
    return (best_acc, best_sharp, best_env, best_patt, best_pose, best_num)

def accuracy_threshold(data, threshold, label, topk):
    num_mantas = 0
    num_correct = 0

    for manta in data:
        score = manta[label]
        if score >= threshold:
            rank = manta['k-rank']
            num_mantas += 1
            if rank < topk:
                num_correct += 1
    accuracy = num_correct / num_mantas
    return accuracy

def main(args):
    gt_labels = json.load(open('dataset/quality_labels.json'))['mantas']
    idcnn_results = json.load(open('results/idcnn_results.json'))['mantas']
    itm_results = json.load(open('results/itm_results.json'))['mantas']
    mm_results = json.load(open('results/mm_results.json'))['mantas']

    if args.accs:
        sha_accs = np.zeros(10)
        env_accs = np.zeros(10)
        patt_accs = np.zeros(10)
        pose_accs = np.zeros(10)

        threshold = 0.0
        for i in range(0, 10):
            sha_accs[i] = accuracy_threshold(mm_results, threshold, "sharpness", args.topk)
            env_accs[i] = accuracy_threshold(mm_results, threshold, "environment", args.topk)
            patt_accs[i] = accuracy_threshold(mm_results, threshold, "pattern", args.topk)
            pose_accs[i] = accuracy_threshold(mm_results, threshold, "pose", args.topk)
            threshold += 0.1

        accs = np.zeros((4, 10))
        accs[0] = sha_accs
        accs[1] = env_accs
        accs[2] = patt_accs
        accs[3] = pose_accs
        print(accs)
        np.savetxt("results/mm_per_label_accuracies_rank=" + str(args.topk) + ".csv", accs, delimiter=",")

    if args.hists:
        sha_array = []
        env_array = []
        pat_array = []
        pos_array = []
        for i in range(0, len(mm_results)):
            manta = itm_results[i]
            sha_array.append(manta['sharpness'])
            env_array.append(manta['environment'])
            pat_array.append(manta['pattern'])
            pos_array.append(manta['pose'])

        fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
        axs[0, 0].hist(res_array, bins=2)
        axs[0, 0].set_title("Sharpness")
        axs[0, 1].hist(env_array, bins=2)
        axs[0, 1].set_title("Environment")
        axs[1, 0].hist(pat_array, bins=2)
        axs[1, 0].set_title("Pattern")
        axs[1, 1].hist(pos_array, bins=2)
        axs[1, 1].set_title("Pose")
        plt.savefig("hists_output.png")

    if args.vis:
        for i in range(0, len(mm_results)):
            manta = itm_results[i]
            #prediction = manta['prediction']
            #class_index = manta['class_index']
            image_class = manta['image_class']
            image_id = manta['image_id']
            resolution = manta['resolution']
            environment = manta['environment']
            pattern = manta['pattern']
            pose = manta['pose']

            fig = plt.figure(figsize=(12, 6))
            fig.add_subplot(121)
            #title = "Prediction: " + str(prediction) + ", Actual: " + str(class_index)
            title = "Manta ID: " + str(image_class)
            plt.title(title)
            image = Image.open('dataset/mantas_cropped/' + image_id)
            plt.imshow(image)
            plt.axis('off')

            fig.add_subplot(122)
            plt.title('Image Quality')
            plt.bar(np.arange(4), [resolution, environment, pattern, pose])
            plt.ylim(0, 1)
            plt.xticks(np.arange(4), ["sha", "env", "patt", "pose"])

            if manta['k-rank'] == 0:
                folder = 'results/preds_success/'
            else:
                folder = 'results/preds_failure/'

            plt.savefig(folder + str(manta['image_class']) + "_" + str(i) + ".png")



if __name__ == "__main__":
    main(parser.parse_args())
