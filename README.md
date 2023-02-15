## ACSTNet: An improved YOLO X method for small object detection with pixel-level attention and parallel Swin Transformer ##

---

## 目录
1. [Environment](#所需环境)

2. [Achievement](#实现的内容)

3. [Download file](#文件下载)

4. [How2train](#训练步骤)

5. [How2predict](#预测步骤)

6. [How2eval](#评估步骤)



## 1.Environment

    torch version: 1.11.0+cu113
    cuda version: 11.3
    cudnn version: 8200
    torchversion -V: 0.12.0+cu113

Python environment import:

```bash
pip install -r requirements.txt
```



## 2.Achievement
- [x] Backbone feature extraction network: CSPNet with Swin Transform network structure is used.  
- [x] Neck layer: increase (ECSNFAM multi-attention structure.
- [x] Categorical regression layer: Decoupled Head, in YoloX, Yolo Head is divided into two parts of categorical regression, and only integrated together in the final prediction.
- [x] Tips used for training: Mosaic data enhancement, IOU and GIOU, learning rate cosine annealing decay.
- [x] Anchor Free
- [x] SimOTA

## 3.**Weights Download**

The weights required for training can be downloaded from Google Drive.  
Link: https://drive.google.com/drive/folders/1cF7GUiqjay0WJ-lElzG-MrHZHpY3kuQu?usp=sharing    

## 4.Train

### Training RSOD dataset.

1. **Preparation of the data set **
   First download the RSOD dataset, then use the VOC format for training and place the file under the path: ACSTNet/dataset/RSOD-Dataset

2. **Modify the parameters needed for training   **

3. **Start training **

  ```python
python train.py
  ```



## 5.Predict
**Training result prediction**

We need to use two files to predict the training results, yolo.py and predict.py. We first need to go to yolo.py and modify model_path and classes_path, these two parameters must be modified. model_path points to the trained weights file, which is in the logs folder.   

Once you have completed the changes you can run predict.py for testing.

  ```python
  python predict.py
  ```

​     

## 6.Eval 
### 评估RSOD的测试集
1. This paper uses the VOC format for evaluation. RSOD has divided the test set and its path is at:dataset/RSOD-Dataset/test_xywh.txt.
2. Modify model_path as well as classes_path inside yolo.py. **model_path points to the trained weights file, in the logs folder. classes_path points to the txt corresponding to the detected category.** 
3. The evaluation results can be obtained by running get_map.py, and the evaluation results will be saved in the map_out folder.

```python
python get_map.py
```



## Reference
https://github.com/Megvii-BaseDetection/YOLOX
