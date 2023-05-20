## YOLOV4：You Only Look Once
Object Detection Model - Modify backbone network replacement MRay - implementation in pytorch
---

## 所需环境
torch==1.2.0

## 训练步骤  
### a、训练Sixray数据集
1. Data set preparation
** This paper uses VOC format for training. Before training, 
it is necessary to download the corresponding data set and put it in the root directory after decompression.
**  

2. Processing of data sets
Modify annotation_mode=2 in voc_annotation.py and run voc_annotation.py to generate 2021_train.txt and 2021_val.txt in the root directory.   

3. Start network training
The default parameter of train.py is used to train the VOC data set. Training can be started by running train.py directly.   

4. Training result prediction
Two files, OLO.py and predict.py, are needed for the prediction of training results. 
We first need to go to OLO.py and change model_path and classes_path. These two parameters must be changed.   
**model_path points to the trained weight file, in the logs folder.
classes_path points to the txt corresponding to the detection category.**   
Once the changes are complete, you can run predict.py for testing. After running, enter the picture path to detect.   

### b. Train your own data set
1. Data set preparation  
**This paper uses VOC format for training, and needs to make a good data set before training,**    
Before training, put the label file in the Annotation in the VOC2021 folder under the VOCdevkit folder.
Place the image file in JPEGImages under VOC2021 folder under VOCdevkit folder before training.   

2. Processing of data sets
After placing the data set, we need to use voc_annotation.py to get 2021_train.txt and 2021_val.txt for training.
Modify the parameters in voc_annotation.py. The first training can only modify classes_path, which is used to point to the txt corresponding to the detection category.
When training your own data set, you can create a cls_classes.txt to write the categories you need to distinguish.
The contents of the model_data/cls_classes.txt file are:      
```python
cat
dog
...
```
Modify the classes_path in voc_annotation.py so that it corresponds to cls_classes.txt and run voc_annotation.py.  

3. Start network training  
**There are many training parameters in train.py. You can read the notes carefully after downloading the library. 
The most important part is still classes_path in train.py.**  
**classes_path is used to point to the txt corresponding to the class to be detected. 
This txt is the same as the txt in voc_annotation.py! Training yourself data sets must be modified!**  
After modifying classes_path, you can run train.py to start training. 
After training multiple epoCs, the weights will be generated in the logs folder.  

4. Training result prediction
Two files, OLO.py and predict.py, are needed for the prediction of training results. 
Modify model_path and classes_path in YOLO.py.  
**model_path points to the trained weight file, in the logs folder.
classes_path points to the txt corresponding to the detection category.**  
Once the changes are complete, you can run predict.py for testing. After running, enter the picture path to detect.

## Prediction step
### a、Use pre-training weights
1. After downloading the library, decompress it, download yolo_weights.pth on Baidu web disk, put in model_data, run predict.py, and enter  
```python
img/street.jpg
```
2. The Settings in predict.py can be used for fps testing and video detection.  
### b、Use your own trained weights
1. Follow the training steps.  
2. In the olo.py file, modify model_path and classes_path in the following parts to make them correspond to the trained files; 
**model_path corresponds to the weight file under the logs folder. classes_path is the class that model_path corresponds to**.  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   Make sure to modify model_path and classes_path when using your trained model to predict!
    #   model_path points to the weights file in the folder logs, and classes_path points to txt in model_data.
    #   If shape mismatch occurs, also note that the model_path and classes_path parameters are changed during training.
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolo_weights.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path indicates the txt file corresponding to the Prior box, which is not modified.
    #   anchors_mask used to help code find the corresponding prior box, generally unchanged.
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   The size of the input image, which must be a multiple of 32.
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   Only prediction boxes with scores greater than confidence will be retained
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   Nms_iou size used for non-maximum suppression
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   This variable controls whether letterbox_image is used to resize the input image without loss of fidelity.
    #   After several tests, it was found that turning off letterbox_image to resize directly works better
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    #   Whether to use Cuda
    #   No GPU can be set to False
    #-------------------------------#
    "cuda"              : True,
}
```
3. Run predict.py and enter  
```python
img/street.jpg
```
4. The Settings in predict.py can be used for fps testing and video detection.  

## Evaluation procedure 
### a、Evaluate SIXRAY's test set
1. This paper uses VOC format for assessment. Now that the test set is partitioned, there is no need to use voc_annotation.py to generate txt in the ImageSets folder.
2. Modify model_path and classes_path in OLO.py. **model_path points to the trained weight file, in the logs folder. classes_path points to the txt corresponding to the detection category.**  
3. Run get_map.py to get the evaluation results, which are stored in the map_out folder.

### b、Evaluate your own data set
1. This paper uses VOC format for assessment.  
2. If the voc_annotation.py file has been run before training, the code automatically divides the data set into a training set, a validation set, and a test set. If you want to change the ratio of the test set, you can change trainval_percent under the voc_annotation.py file. trainval_percent specifies the ratio (training + verification) to test. By default (training + verification): test = 9:1. train_percent Specifies the ratio of training set to verification set (training set + verification set). By default, training set: verification set = 9:1.
3. After the test set is divided by voc_annotation.py, go to the get_map.py file to modify classes_path. classes_path is used to point to the txt corresponding to the detection category, which is the same as the txt in training. The data set that evaluates yourself must be modified.
4. Modify model_path and classes_path in OLO.py. **model_path points to the trained weight file, in the logs folder. classes_path points to the txt corresponding to the detection category.**  
5. Run get_map.py to get the evaluation results, which are stored in the map_out folder.

