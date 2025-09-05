#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
紧急苍蝇检测脚本
使用极低阈值检测，将苍蝇归类到最相似的类别
"""
import cv2
import os
from ultralytics import YOLO
import numpy as np

def emergency_fly_detection():
    """紧急苍蝇检测"""
    
    print("=" * 60)
    print("紧急苍蝇检测 - 使用极低阈值")
    print("=" * 60)
    
    # 模型路径
    model_path = r"D:\校内\新建文件夹\Finaldesign2.0\yolo12-litchi\42_demo\runs\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 获取图片路径
    image_path = input("请输入你的苍蝇图片路径: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return
    
    print(f"✅ 图片文件存在: {image_path}")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 数据集类别
    class_names = ['麻蝇', '蛾类', '黑斑果蝇', '粪蝇', '蚤蝇']
    print(f"📋 模型类别: {class_names}")
    print()
    
    # 极低阈值配置
    configs = [
        {"conf": 0.001, "iou": 0.1, "imgsz": 1920, "name": "超低阈值+大尺寸"},
        {"conf": 0.005, "iou": 0.2, "imgsz": 1536, "name": "极低阈值+中尺寸"},
        {"conf": 0.01, "iou": 0.3, "imgsz": 1280, "name": "低阈值+标准尺寸"},
        {"conf": 0.05, "iou": 0.4, "imgsz": 1280, "augment": True, "name": "低阈值+TTA"},
    ]
    
    print("🔍 开始检测...")
    print()
    
    best_result = None
    max_detections = 0
    
    for i, config in enumerate(configs):
        print(f"测试 {i+1}: {config['name']}")
        print(f"   置信度: {config['conf']}, IOU: {config['iou']}, 尺寸: {config['imgsz']}")
        
        try:
            results = model(
                image_path,
                conf=config['conf'],
                iou=config['iou'],
                imgsz=config['imgsz'],
                augment=config.get('augment', False),
                save=False,
                verbose=False
            )
            
            result = results[0]
            detections = len(result.boxes) if result.boxes is not None else 0
            
            print(f"   结果: {detections} 个目标")
            
            if detections > 0:
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for j, (cls, conf) in enumerate(zip(classes, confidences)):
                    class_name = class_names[int(cls)]
                    print(f"     {j+1}. {class_name}: {conf:.4f}")
                
                if detections > max_detections:
                    max_detections = detections
                    best_result = config
                    
        except Exception as e:
            print(f"   错误: {e}")
        
        print()
    
    if best_result:
        print("🏆 最佳检测结果:")
        print(f"   配置: {best_result['name']}")
        print(f"   检测到 {max_detections} 个目标")
        print()
        print("💡 建议:")
        print("   虽然检测到了目标，但可能归类不准确")
        print("   因为训练数据中没有'苍蝇'类别")
        print("   建议重新训练模型，添加'苍蝇'类别")
        
        # 保存结果
        try:
            results = model(
                image_path,
                conf=best_result['conf'],
                iou=best_result['iou'],
                imgsz=best_result['imgsz'],
                augment=best_result.get('augment', False),
                save=True,
                project="emergency_fly_detection",
                name="best_result"
            )
            print(f"   结果已保存到: emergency_fly_detection/best_result/")
        except:
            pass
    else:
        print("❌ 所有配置都未能检测到目标")
        print()
        print("🔧 进一步解决方案:")
        print("   1. 重新训练模型，添加'苍蝇'类别")
        print("   2. 使用通用目标检测模型")
        print("   3. 将苍蝇图片标注到最相似的类别（如'麻蝇'）")

if __name__ == "__main__":
    emergency_fly_detection()
