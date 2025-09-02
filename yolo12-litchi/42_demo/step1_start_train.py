#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from ultralytics import YOLO

# 获取当前路径的根路径
import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录路径
current_dir = os.path.dirname(current_file_path)
# 获取当前文件所在目录的上级目录路径
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "ultralytics/cfg/datasets/A_DATA.yaml")
# ---------------------------------- 训练超参数配置  ------------------------------------------------------
# DATA_CONFIG_PATH = r'F:\Upppppdate\35-\yolo11-elec_device\ultralytics\cfg\datasets\A_my_data.yaml' # 数据集配置文件路径
DATA_CONFIG_PATH = data_dir
EPOCHS = 200# 模型训练的轮数
IMAGE_SIZE = 1080# 图像输入的大小
DEVICE = [0]       # 设备配置
WORKERS = 0       # 多线程配置
BATCH = 8       # 数据集批次大小
CACHE = False        # 缓存
AMP = True       # 是否开启自动混合精度训练


# ---------------------------------- 训练超参数配置  ------------------------------------------------------


model = YOLO("yolo12s.yaml").load("yolo12s.pt")
results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo12n_pretrained_6", epochs=EPOCHS,
                      imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=False, amp=AMP,
                      conf=0.1,iou=0.7,max_det=1000
                    )
time.sleep(10)

# model = YOLO("yolo12s.yaml").load("yolo12s.pt")# load a pretrained model (recommended for training)
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo12s_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
#
#
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
#
# model = YOLO("yolov10n.yaml").load("yolov10n.pt")# load a pretrained model (recommended for training)
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolov10n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
#
#
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")# load a pretrained model (recommended for training)
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolov8n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
#
# model = YOLO("yolov5n.yaml").load("yolov5n.pt")# load a pretrained model (recommended for training)
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolov5n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

#
# 目标检测
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# model = YOLO("yolo12n-c2fuib.yaml")
# results = model.train(data=DATA_CONFIG_PATH, project="./runsmodel = YOLO("yolo12n.yaml").load("yolo12n.pt")
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo12n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用/yolo12n-c2fuib", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用


# model = YOLO("yolo12n.yaml").load("yolo12n.pt")
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo12n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# model = YOLO("yolo12n-Coord.yaml")
# model = YOLO("yolo12n-SEAttention.yaml")
# model = YOLO("yolo12-x-cbam.yaml")
# model = YOLO("yolo12n-GhostConv.yaml")
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo12n_coord_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# 分割
# model = YOLO("yolo12n-seg.yaml")
# results = model.train(data=os.path.join(parent_dir, "ultralytics/cfg/datasets/coco128-seg.yaml"), project="./runs/yolo12n_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用


# 关键点检测
# model = YOLO("yolo12n-pose.yaml")
# results = model.train(data=os.path.join(parent_dir, "ultralytics/cfg/datasets/tiger-pose.yaml"), project="./runs/yolo12n-pose_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
#

# 旋转目标检测
# model = YOLO("yolo12n-obb.yaml")
# results = model.train(data=os.path.join(parent_dir, "ultralytics/cfg/datasets/dota8.yaml"), project="./runs/yolo12n-obb_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# 旋转目标检测
# model = YOLO("yolo12n-cls.yaml")
# results = model.train(data=r"H:\raspi\0000-38-visdrone-detect-yolo12\yolo_data\mnist160", project="./runs/yolo12n-cls", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

#
# model = YOLO("yolo11s.yaml").load("yolo11s.pt")
# results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11s_pretrained", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
# time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用




""" 检测模型训练 
model = YOLO("yolo11-SeAttention.yaml")
results = model.train(data=DATA_CONFIG_PATH, project="./runs/test", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用



# 各类消融实验
model = YOLO("yolo11n.yaml") # load a pretrained model (recommended for training)
results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11n_original", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

# --------------------------------------------------模型精度提升实验----------------------------------
model = YOLO("yolo11n-cbam.yaml") # load a pretrained model (recommended for training)
results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11n_cbam", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用


model = YOLO("yolo11n-CoordAttention.yaml") # load a pretrained model (recommended for training)
results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11n_CoordAttention", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

#---------------------------  模型轻量化实验 -------------------------------------------------------
model = YOLO("yolo11n-GhostConv.yaml") # load a pretrained model (recommended for training)
results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11n_GhostConv", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用

model = YOLO("yolo11n-DWConv.yaml") # load a pretrained model (recommended for training)
results = model.train(data=DATA_CONFIG_PATH, project="./runs/yolo11n_DWConv", epochs=EPOCHS, imgsz=IMAGE_SIZE, device=DEVICE, workers=WORKERS, batch=BATCH, cache=CACHE, amp=AMP)  # CPU 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
"""


"""
训练可选参数
Argument	    Default	    Description
model	        None	    Specifies the model file for training. Accepts a path to either a .pt pretrained model or a .yaml configuration file. Essential for defining the model structure or initializing weights.
data	        None	    Path to the dataset configuration file (e.g., coco8.yaml). This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.
epochs	        100	        Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.
time	        None	    Maximum training time in hours. If set, this overrides the epochs argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.
patience	    100	        Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.
batch	        16	        Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
imgsz	        640	        Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.
save	        True	    Enables saving of training checkpoints and final model weights. Useful for resuming training or model deployment.
save_period	    -1	        Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.
cache	        False	    Enables caching of dataset images in memory (True/ram), on disk (disk), or disables it (False). Improves training speed by reducing disk I/O at the cost of increased memory usage.
device	        None	    Specifies the computational device(s) for training: a single GPU (device=0), multiple GPUs (device=0,1), CPU (device=cpu), or MPS for Apple silicon (device=mps).
workers	        8	        Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.
project	        None	    Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.
name	        None	    Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.
exist_ok	    False	    If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.
pretrained	    True	    Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.
optimizer	    'auto'	    Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.
verbose	        False	    Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.
seed	        0	        Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.
deterministic	True	    Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.
single_cls	    False	    Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.
rect	        False	    Enables rectangular training, optimizing batch composition for minimal padding. Can improve efficiency and speed but may affect model accuracy.
cos_lr	        False	    Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.
close_mosaic	10	        Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.
resume	        False	    Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.
amp	True	    Enables     Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.
fraction	    1.0	        Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.
profile 	    False	    Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.
freeze	        None	    Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.
lr0	0.01	    Initial     learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
lrf	0.01	    Final       learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
momentum	    0.937	    Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.
weight_decay	0.0005	    L2 regularization term, penalizing large weights to prevent overfitting.
warmup_epochs	3.0	        Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.
warmup_momentum	0.8	        Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.
warmup_bias_lr	0.1	        Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.
box	            7.5	        Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.
cls	            0.5	        Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.
dfl    	        1.5   	    Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.
pose	        12.0	    Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.
kobj	        2.0	        Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.
label_smoothing	0.0	        Applies label smoothing, softening hard labels to a mix of the target label and a uniform distribution over labels, can improve generalization.
nbs	            64	        Nominal batch size for normalization of loss.
overlap_mask	True	    Determines whether segmentation masks should overlap during training, applicable in instance segmentation tasks.
mask_ratio	    4	        Downsample ratio for segmentation masks, affecting the resolution of masks used during training.
dropout	        0.0 	    Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.
val	            True	    Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.
plots	        False	    Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
"""