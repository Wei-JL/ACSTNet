import os
import xml.etree.ElementTree as ET

from tqdm import tqdm


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


if __name__ == '__main__':
    set_cls = set()
    xmlDir = "dataset/RSOD-Dataset/Annotations"
    cls_txt = "dataset/RSOD-Dataset/labels.txt"

    getClsTxt(xmlDir, cls_txt)
