#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
专门针对放大图片害虫检测的增强训练脚本
解决识别不出来的问题
"""
import time
from ultralytics import YOLO
import os

# 获取当前路径的根路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "ultralytics/cfg/datasets/A_DATA.yaml")

# ---------------------------------- 针对放大目标的训练超参数配置  ------------------------------------------------------
DATA_CONFIG_PATH = data_dir
EPOCHS = 200  # 增加训练轮数
IMAGE_SIZE = 1536  # 进一步增大输入尺寸，提升大目标检测能力
DEVICE = [0]       # 设备配置（使用第0块GPU）
WORKERS = 0       # 多线程配置（0避免Windows系统线程冲突）
BATCH = 2         # 减小批次大小以适应更大的输入尺寸
CACHE = False     # 缓存（False避免占用过多内存）
AMP = True        # 开启自动混合精度训练（加速训练且节省显存）

# 专门针对放大目标的增强策略
AUG_DEGREES = 0.0
AUG_TRANSLATE = 0.15      # 增加平移范围
AUG_SCALE = 1.2           # 进一步扩大缩放范围，包含更强的放大
AUG_SHEAR = 0.0
AUG_PERSPECTIVE = 0.0
AUG_FLIPLR = 0.5
AUG_FLIPUD = 0.0
AUG_MOSAIC = 0.9          # 保持强马赛克增强
AUG_MIXUP = 0.15          # 增加mixup强度
AUG_COPY_PASTE = 0.5      # 增强复制粘贴，提升大目标密度多样性
CLOSE_MOSAIC_LAST = 15    # 末期关闭mosaic稳定收敛

# 学习率策略优化
LR0 = 0.01               # 初始学习率
LRF = 0.01               # 最终学习率
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 5.0      # 增加预热轮数

# 损失函数权重调整（针对大目标优化）
BOX_LOSS_WEIGHT = 10.0   # 增加框回归损失权重
CLS_LOSS_WEIGHT = 1.0    # 分类损失权重
DFL_LOSS_WEIGHT = 2.0    # 分布焦点损失权重

print("=" * 60)
print("开始训练针对放大目标的害虫检测模型")
print(f"数据集路径: {DATA_CONFIG_PATH}")
print(f"输入尺寸: {IMAGE_SIZE}")
print(f"训练轮数: {EPOCHS}")
print(f"批次大小: {BATCH}")
print(f"缩放范围: {AUG_SCALE}")
print("=" * 60)

# 使用修改后的分割模型
model = YOLO("yolo12n-seg.yaml").load("yolo12n.pt")

# 启动增强训练
results = model.train(
    data=DATA_CONFIG_PATH,
    project="D:\校内\新建文件夹\训练",
    name="enhanced_large_targets",  # 给这次训练起个专门的名字
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    device=DEVICE,
    workers=WORKERS,
    batch=BATCH,
    cache=CACHE,
    amp=AMP,
    
    # 关键增强参数
    degrees=AUG_DEGREES,
    translate=AUG_TRANSLATE,
    scale=AUG_SCALE,
    shear=AUG_SHEAR,
    perspective=AUG_PERSPECTIVE,
    fliplr=AUG_FLIPLR,
    flipud=AUG_FLIPUD,
    mosaic=AUG_MOSAIC,
    mixup=AUG_MIXUP,
    copy_paste=AUG_COPY_PASTE,
    close_mosaic=CLOSE_MOSAIC_LAST,
    
    # 学习率策略
    lr0=LR0,
    lrf=LRF,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    warmup_epochs=WARMUP_EPOCHS,
    
    # 损失函数权重
    box=BOX_LOSS_WEIGHT,
    cls=CLS_LOSS_WEIGHT,
    dfl=DFL_LOSS_WEIGHT,
    
    # 其他优化参数
    patience=50,          # 早停耐心值
    save_period=10,       # 每10轮保存一次
    plots=True,           # 生成训练图表
    val=True,             # 启用验证
    verbose=True,         # 详细输出
)

print("训练完成！")
print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
time.sleep(10)
