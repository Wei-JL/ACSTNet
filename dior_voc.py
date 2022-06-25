import os
import random
import xml.etree.ElementTree as ET

from tqdm import tqdm

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
xmlDir = "dataset/DIOR_VOC/Annotations"
classes_path = 'dataset/DIOR_VOC/labels.txt'
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
set_cls = set()
xmlDir = "dataset/DIOR_VOC/Annotations"
cls_txt = "dataset/DIOR_VOC/labels.txt"
VOC_path = 'dataset/DIOR_VOC'
VOC_sets = [('train_xywh', 'train'), ('val_xywh', 'val'), ('test_xywh', 'test')]


def getClsTxt(xmlDir, cls_txt):
    """
    xmlDir  ：xml地址
    cls_txt : 输出cls文件地址
    """
    for name in tqdm(os.listdir(xmlDir)):
        xmlFile = os.path.join(xmlDir, name)
        with open(xmlFile, "r+", encoding='utf-8') as fp:
            tree = ET.parse(fp)
            root = tree.getroot()
            for obj in root.iter('object'):
                cls = obj.find('name').text
                # print(type(cls))
                set_cls.add(cls)

    with open(cls_txt, "w+") as ft:
        for i in set_cls:
            ft.write(i + "\n")


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


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
    getClsTxt(xmlDir, classes_path)
    classes, _ = get_classes(classes_path)
    random.seed(0)
    main_path = os.path.join(VOC_path, "ImageSets/Main")
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    print("Generate train_xywh.txt and val_xywh.txt and test_xywh.txt for train.")
    for train_val, image_set in tqdm(VOC_sets):
        if image_set in ("train", "val"):
            image_ids = open(os.path.join(VOC_path, 'ImageSets/Main/%s.txt' % (image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s/%s.txt' % (VOC_path, train_val), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/JPEGImages-trainval/' % (os.path.abspath(VOC_path)))
                convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
            # print("Generate train_xywh.txt and val_xywh.txt for train done.")
        elif image_set == "test":
            image_ids = open(os.path.join(VOC_path, 'ImageSets/Main/%s.txt' % (image_set)),
                             encoding='utf-8').read().strip().split()
            list_file = open('%s/%s.txt' % (VOC_path, train_val), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/JPEGImages-test/' % (os.path.abspath(VOC_path)))
                convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
    print("Generate train_xywh.txt and val_xywh.txt and test_xywh.txt for train done.")
