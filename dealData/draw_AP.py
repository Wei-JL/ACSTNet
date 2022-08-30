import os
import csv
import colorsys

import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import get_classes

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige',
    'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown',
    'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'darkgoldenrod', 'darkgray',
    'darkgreen', 'darkkhaki', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen''fuchsia', 'gainsboro', 'ghostwhite', 'gold',
    'goldenrod',
    'gray',
    'green',
    'greenyellow',
    'honeydew',
    'hotpink',
    'indianred',
    'indigo',
    'ivory',
    'khaki',
    'lavender',
    'lavenderblush',
    'lawngreen',
    'lemonchiffon',
    'lightblue',
    'lightcoral', ]


def deal_csv(files_list, AP_dir):
    save_info = []

    for file in tqdm(files_list):
        csv_path = os.path.join(AP_dir, file)
        class_ = file.split(".")[0]
        with open(csv_path, "rt") as csvfile:
            column0 = []
            column1 = []
            reader = csv.reader(csvfile)
            for row in reader:
                column0.append(row[0])
                column1.append(row[1])

            # column0 = [row[0] for row in reader]
            # column1 = [row[1] for row in reader]
            save_info.append([class_, column0, column1])

    return save_info


if __name__ == '__main__':

    classes_path = '../dataset/DIOR_VOC/labels.txt'
    AP_dir = '../map_out/results/CSV'

    files_list = os.listdir(AP_dir)
    AP_info = deal_csv(files_list, AP_dir)
    # ---------------------------------------------------#
    #   画框设置不同的颜色
    # ---------------------------------------------------#
    # class_names, num_classes = get_classes(classes_path)
    # hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    plt.figure(figsize=(16, 12))
    for index, info in enumerate(AP_info):
        # print(index, info)
        plt.plot(info[1], info[2], label=info[0], color=colors[index])

    plt.grid(True, linestyle='--')
    plt.legend(loc=9)
    plt.text(6, 97,)
    plt.show()