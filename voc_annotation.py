import os
import random
import xml.etree.ElementTree as ET
import time

from tqdm import tqdm

from utils.utils import get_classes

# --------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOC/ImageSets里面的txt以及训练用的
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的train.txt、val.txt、test.txt、all_data.txt
#   annotation_mode为2代表获得训练用的train_xywh.txt、val_xywh.txt
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 0
# -------------------------------------------------------------------#
#   数据集标签信息
# -------------------------------------------------------------------#
classes_path = 'dataset/FruitVOCData/labels.txt'
# --------------------------------------------------------------------------------------------------------------------------------#
#   all_data_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#
all_data_percent = 1
train_percent = 0.85
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
VOC_path = 'dataset/FruitVOCData'
VOC_sets = [('train_xywh', 'train'), ('val_xywh', 'val')]
classes, _ = get_classes(classes_path)


def convert_annotation(image_id, list_file):
    in_file = open(os.path.join(VOC_path, 'Annotations/%s.xml' % image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    for img_name in root.iter('filename'):
        list_file.write(img_name.text)
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    main_path = os.path.join(VOC_path, "ImageSets/Main")
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    if annotation_mode == 0:
        print("Generate txt in ImageSets.")
        xmlfilepath = os.path.join(VOC_path, 'Annotations')
        saveBasePath = os.path.join(VOC_path, 'ImageSets/Main')
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in tqdm(temp_xml):
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        _list = range(num)
        tv = int(num * all_data_percent)
        tr = int(tv * train_percent)
        # 打乱数据集顺序，随机性
        all_data = random.sample(_list, tv)
        train = random.sample(all_data, tr)

        print("train and val size", tv)
        print("train size", tr)
        fall_data = open(os.path.join(saveBasePath, 'all_data.txt'), 'w')
        ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
        ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
        fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

        for i in tqdm(_list):
            name = total_xml[i][:-4] + '\n'
            if i in all_data:
                fall_data.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        fall_data.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("Generate txt in ImageSets done.")

    annotation_mode = 2

    if annotation_mode == 2:
        print("Generate train_xywh.txt and val_xywh.txt for train.")
        for train_val, image_set in tqdm(VOC_sets):
            image_ids = open(os.path.join(VOC_path, 'ImageSets/Main/%s.txt' % (image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s/%s.txt' % (VOC_path, train_val), 'w', encoding='utf-8')
            for image_id in image_ids:
                # list_file.write('%s/JPEGImages/' % (os.path.abspath(VOC_path)))  # 绝对路径
                list_file.write('%s/JPEGImages/' % (VOC_path))  # 相对路径路径

                convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate train_xywh.txt and val_xywh.txt for train done.")
