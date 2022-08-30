import os

Ann = "../dataset/RSOD-Dataset/Annotations"
Img = "../dataset/RSOD-Dataset/JPEGImages"
with open("rsod.txt", "r+") as fp:
    for info in fp.readlines():
        info = info.replace("\n", "")
        info_xml = os.path.join(Ann, info+".xml")
        info_img = os.path.join(Img, info + ".jpg")
        os.remove(info_xml)
        os.remove(info_img)