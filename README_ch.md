## ACSTNet: An improved YOLO X method for small object detection with pixel-level attention and parallel Swin Transformer ##

---

## 目录
1. [所需环境 Environment](#所需环境)

2. [实现的内容 Achievement](#实现的内容)

3. [文件下载 Download](#文件下载)

4. [训练步骤 How2train](#训练步骤)

5. [预测步骤 How2predict](#预测步骤)

6. [评估步骤 How2eval](#评估步骤)



## 1.所需环境

    torch version: 1.11.0+cu113
    cuda version: 11.3
    cudnn version: 8200
    torchversion -V: 0.12.0+cu113

Python环境导入:

```bash
pip install -r requirements.txt
```



## 2.实现的内容
- [x] 主干特征提取网络：使用了CSPNet与Swin Transform网络结构。  
- [x] 颈部层：增加(ECSNFAM多注意力结构
- [x] 分类回归层：Decoupled Head，在YoloX中，Yolo Head被分为了分类回归两部分，最后预测的时候才整合在一起。
- [x] 训练用到的小技巧：Mosaic数据增强、IOU和GIOU、学习率余弦退火衰减。
- [x] Anchor Free：不使用先验框
- [x] SimOTA：为不同大小的目标动态匹配正样本。



## 3.文件下载

训练所需的权值可在Google Drive中下载。  
链接: https://drive.google.com/drive/folders/1cF7GUiqjay0WJ-lElzG-MrHZHpY3kuQu?usp=sharing    



## 4.训练步骤

### 训练RSOD数据集
1. 数据集的准备   
    **本文使用VOC格式进行训练，将文件放在：ACSTNet/dataset/RSOD-Dataset 路径下**

2. 修改训练所需要的参数      

3. 开始网络训练   

  在默认的参数情况下

  ```python
  python train.py
  ```


4. 训练结果预测   
    训练结果预测需要用到两个文件，分别是yolo.py和predict.py。我们首先需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改。   
    **model_path指向训练好的权值文件，在logs文件夹里。   
    classes_path指向检测类别所对应的txt。**   
    完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。

  ```python
  python predict.py
  ```

​     

## 6.评估步骤 
### 评估RSOD的测试集
1. 本文使用VOC格式进行评估。RSOD已经划分好了测试集，它的路径在:dataset/RSOD-Dataset/test_xywh.txt.
2. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。



## Reference
https://github.com/Megvii-BaseDetection/YOLOX
