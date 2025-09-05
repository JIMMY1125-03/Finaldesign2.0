#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
检查数据集质量和标注的脚本
帮助诊断为什么识别不出来
"""
import os
import cv2
import numpy as np
from pathlib import Path
import yaml

def check_dataset_quality():
    """检查数据集质量和标注"""
    
    # 读取数据集配置
    config_path = r"D:\校内\新建文件夹\Finaldesign2.0\yolo12-litchi\ultralytics\cfg\datasets\A_DATA.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"无法读取配置文件: {e}")
        return
    
    dataset_path = config['path']
    train_images = os.path.join(dataset_path, config['train'])
    train_labels = os.path.join(dataset_path, config['segments_train'])
    
    print("=" * 60)
    print("数据集质量检查报告")
    print("=" * 60)
    print(f"数据集路径: {dataset_path}")
    print(f"训练图片路径: {train_images}")
    print(f"训练标签路径: {train_labels}")
    print(f"类别数量: {config['nc']}")
    print(f"类别名称: {config['names']}")
    print()
    
    # 检查图片和标签文件
    if not os.path.exists(train_images):
        print(f"❌ 训练图片路径不存在: {train_images}")
        return
    
    if not os.path.exists(train_labels):
        print(f"❌ 训练标签路径不存在: {train_labels}")
        return
    
    # 统计文件数量
    image_files = list(Path(train_images).glob("*.jpg")) + list(Path(train_images).glob("*.png"))
    label_files = list(Path(train_labels).glob("*.txt"))
    
    print(f"📊 文件统计:")
    print(f"   图片文件数量: {len(image_files)}")
    print(f"   标签文件数量: {len(label_files)}")
    
    if len(image_files) == 0:
        print("❌ 没有找到图片文件！")
        return
    
    if len(label_files) == 0:
        print("❌ 没有找到标签文件！")
        return
    
    # 检查图片尺寸分布
    print(f"\n📏 图片尺寸分析:")
    sizes = []
    for img_file in image_files[:50]:  # 只检查前50张
        try:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w = img.shape[:2]
                sizes.append((w, h))
        except:
            continue
    
    if sizes:
        sizes = np.array(sizes)
        print(f"   平均尺寸: {sizes.mean(axis=0).astype(int)}")
        print(f"   最小尺寸: {sizes.min(axis=0)}")
        print(f"   最大尺寸: {sizes.max(axis=0)}")
        print(f"   尺寸标准差: {sizes.std(axis=0).astype(int)}")
    
    # 检查标签质量
    print(f"\n🏷️ 标签质量分析:")
    total_objects = 0
    class_counts = {}
    bbox_sizes = []
    
    for label_file in label_files[:100]:  # 只检查前100个标签
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # 解析分割点（简化处理）
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) >= 6:  # 至少3个点
                        total_objects += 1
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        
                        # 计算边界框大小
                        x_coords = coords[::2]
                        y_coords = coords[1::2]
                        if len(x_coords) >= 2 and len(y_coords) >= 2:
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            bbox_sizes.append((width, height))
        except Exception as e:
            print(f"   处理标签文件 {label_file} 时出错: {e}")
    
    print(f"   总目标数量: {total_objects}")
    print(f"   各类别目标数量:")
    for class_id, count in sorted(class_counts.items()):
        class_name = config['names'][class_id] if class_id < len(config['names']) else f"未知类别{class_id}"
        print(f"     {class_name} (ID:{class_id}): {count}个")
    
    if bbox_sizes:
        bbox_sizes = np.array(bbox_sizes)
        print(f"   目标尺寸统计:")
        print(f"     平均尺寸: {bbox_sizes.mean(axis=0)}")
        print(f"     最小尺寸: {bbox_sizes.min(axis=0)}")
        print(f"     最大尺寸: {bbox_sizes.max(axis=0)}")
        
        # 分析大目标比例
        large_targets = np.sum((bbox_sizes > 0.3).any(axis=1))
        print(f"     大目标数量 (>0.3): {large_targets} ({large_targets/len(bbox_sizes)*100:.1f}%)")
    
    # 检查是否有空标签文件
    empty_labels = 0
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    empty_labels += 1
        except:
            continue
    
    if empty_labels > 0:
        print(f"⚠️  发现 {empty_labels} 个空标签文件")
    
    # 建议
    print(f"\n💡 优化建议:")
    if len(image_files) < 100:
        print("   - 数据集较小，建议增加更多训练样本")
    if bbox_sizes and bbox_sizes.mean() < 0.1:
        print("   - 目标尺寸偏小，建议增加更多大目标样本")
    if large_targets/len(bbox_sizes) < 0.2:
        print("   - 大目标样本不足，建议增加放大图片的标注")
    print("   - 确保训练时使用 imgsz=1536 或更大")
    print("   - 使用增强训练脚本: step1_start_train_enhanced.py")

if __name__ == "__main__":
    check_dataset_quality()
