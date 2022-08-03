import os
import re
import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.utils import get_classes, cvtColor


def count_label(txt_line, class_names, save_img_dir):
    """
    txt_line : 图片路径 + box信息
    """
    count = 0
    box_list = []
    for str_data in txt_line.split(' '):
        if count == 0:
            count += 1
            img_path = str_data
        else:
            data_list = re.findall(r"\d+", str_data)
            data_list = list(map(int, data_list[:]))
            box_list.append(data_list)

    print(img_path, box_list)

    image = Image.open(img_path)
    # ---------------------------------------------------#
    #   获得输入图片的高和宽
    # ---------------------------------------------------#
    image_shape = np.array(np.shape(image)[0:2])
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)

    # ---------------------------------------------------#
    #   画框设置不同的颜色
    # ---------------------------------------------------#
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    font = ImageFont.truetype(font='model_data/simhei.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // np.mean(image_shape), 1))

    # _class = set()
    # for cls in range(len(box_list)):
    #     _class.add(box_list[cls][-1])
    # for c, info in zip(_class, box_list):
    for info in box_list:
        c = info[-1]
        predicted_class = str(class_names[int(info[-1])])
        left, top, right, bottom = info[:4]

        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(predicted_class, font)
        label = predicted_class.encode('utf-8')

        print(label, top, left, bottom, right)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        # text_origin = np.array([left, top])
        c = c % 20
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    os.makedirs(save_img_dir, exist_ok=True)
    img_name = os.path.basename(img_path)
    out_img_path = os.path.join(save_img_dir, img_name)
    image.save(out_img_path)
    print("已保存到:{}".format(out_img_path))

    # img_path = txt_line.split(' ')[0]
    # box_info = txt_line.split(img_path)


if __name__ == '__main__':
    # ---------------------------------------------------#
    #   获得种类和先验框的数量
    # ---------------------------------------------------#
    # txt_path = "TestIMG/DIOR_IMG/test_xywh.txt"
    txt_path = "TestIMG/RSOD_IMG/test_xywh.txt"
    with open(txt_path, "r+", encoding='UTF-8') as fp:
        for txt_line in fp.readlines():
            if len(txt_line) > 3:
                # 大于三个字符
                txt_line = txt_line.replace("\n", "")
                # classes_path = "dataset/DIOR_VOC/labels.txt"
                # save_img_dir = "TestIMG/DIOR_IMG/resultGT"
                classes_path = "dataset/RSOD-Dataset/labels.txt"
                save_img_dir = "TestIMG/RSOD_IMG/resultGT"
                class_names, num_classes = get_classes(classes_path)
                count_label(txt_line, class_names, save_img_dir)
