import os
import csv
import colorsys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import voc_ap

colors = [
    'red', 'aqua', 'greenyellow', 'hotpink', 'gray',
    'indigo', 'aquamarine', 'gold', 'lightblue', 'lightcoral',
    'firebrick', 'black', 'indianred', 'blue', 'blueviolet',
    'brown', 'darkslategray', 'cadetblue', 'goldenrod', 'chocolate',
    'darkgoldenrod', 'hotpink', 'deepskyblue', 'darkseagreen',
    'darkslateblue', 'darkslategray',
    'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue', 'firebrick',
    'forestgreen', 'ghostwhite',
    'green', ]

mAP = {"airplane": 0.93, "tenniscourt": 0.91, "ship": 0.91, "basketballcourt": 0.91,
       "windmill": 0.83, "storagetank": 0.8, "baseballfield": 0.79, "chimney": 0.78,
       "groundtrackfield": 0.77, "golffield": 0.71, "airport": 0.71, "stadium": 0.66,
       "harbor": 0.64, "Expressway-toll-station": 0.63, "overpass": 0.6,
       "Expressway-Service-area": 0.6, "vehicle": 0.59, "trainstation": 0.59,
       "dam": 0.59, "bridge": 0.46}


# mAP = {"aircraft": 0.9906, "oiltank": 0.9585, "playground": 0.9903, "overpass": 0.8457, }


def deal_csv(files_list, AP_dir):
    save_info = []

    for file in tqdm(files_list):
        csv_path = os.path.join(AP_dir, file)
        class_ = file.split(".")[0]
        class_ = class_ + " " + str(mAP[class_])
        with open(csv_path, "rt") as csvfile:
            column0 = []
            column1 = []
            reader = csv.reader(csvfile)
            for row in reader:
                column0.append(float(row[0]))
                column1.append(float(row[1]))

            save_info.append([class_, column0, column1])

    return save_info


if __name__ == '__main__':

    classes_path = '../dataset/DIOR_VOC/labels.txt'
    AP_dir = '../map_out/DIOR/results/CSV'

    files_list = os.listdir(AP_dir)
    AP_info = deal_csv(files_list, AP_dir)
    # ---------------------------------------------------#
    #   画框设置不同的颜色
    # ---------------------------------------------------#
    # class_names, num_classes = get_classes(classes_path)
    # hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    plt.figure(figsize=(12, 8), dpi=200)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    func = lambda x, pos: "" if np.isclose(x, 0) else x

    for index, info in enumerate(AP_info):
        # print(index, info)
        plt.plot(info[1], info[2], label=info[0], color=colors[index])
        # ap, mrec, mprec = voc_ap(info[1][:], info[2][:])
        # area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
        # area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
        # plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
        # plt.plot([0.1, 0.2, 0.3, 0.5, 0.8], [0.8, 0.4, 0.2, 0.2, 0.1], label=info[0], color=colors[index])

        # break

    # plt.text(6, 97, "AP", fontdict=15)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.grid(linestyle='-.')
    plt.grid(True)
    # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    axes = plt.gca()
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    # axes.spines['right'].set_visible(False)
    # plt.grid(True, linestyle='--')
    # plt.legend(loc=9)
    plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.77))
    plt.tick_params(labelsize=16)
    # plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(func))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter('%1.1f'))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(func))

    fig = plt.gcf()
    fig.savefig("res.png")

    # plt.show()
