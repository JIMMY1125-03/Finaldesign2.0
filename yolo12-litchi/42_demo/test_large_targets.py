#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
测试放大目标检测的脚本
使用多种参数组合来找到最佳设置
"""
import cv2
import os
from ultralytics import YOLO
import numpy as np


def test_detection_with_multiple_settings():
    """使用多种设置测试检测效果"""

    # 模型路径
    model_path = r"D:\校内\新建文件夹\Finaldesign2.0\yolo12-litchi\42_demo\runs\best.pt"

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或检查路径")
        return

    # 测试图片路径（请替换为你的测试图片）
    test_image = input("请输入测试图片路径: ").strip()
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        return

    print("=" * 60)
    print("放大目标检测测试")
    print("=" * 60)

    # 加载模型
    model = YOLO(model_path)

    # 多种测试配置
    test_configs = [
        {
            "name": "默认设置",
            "conf": 0.25,
            "iou": 0.7,
            "imgsz": 640
        },
        {
            "name": "低置信度",
            "conf": 0.1,
            "iou": 0.5,
            "imgsz": 1280
        },
        {
            "name": "大尺寸输入",
            "conf": 0.15,
            "iou": 0.6,
            "imgsz": 1536
        },
        {
            "name": "超大尺寸+TTA",
            "conf": 0.1,
            "iou": 0.5,
            "imgsz": 1920,
            "augment": True
        },
        {
            "name": "极低阈值",
            "conf": 0.05,
            "iou": 0.4,
            "imgsz": 1280
        }
    ]

    results_summary = []

    for i, config in enumerate(test_configs):
        print(f"\n🔍 测试配置 {i + 1}: {config['name']}")
        print(f"   置信度: {config['conf']}")
        print(f"   IOU阈值: {config['iou']}")
        print(f"   输入尺寸: {config['imgsz']}")
        if 'augment' in config:
            print(f"   TTA增强: {config['augment']}")

        try:
            # 执行检测
            results = model(
                test_image,
                conf=config['conf'],
                iou=config['iou'],
                imgsz=config['imgsz'],
                augment=config.get('augment', False),
                save=False,
                verbose=False
            )

            # 分析结果
            result = results[0]
            detections = len(result.boxes) if result.boxes is not None else 0

            print(f"   ✅ 检测到 {detections} 个目标")

            if detections > 0:
                # 显示检测到的类别
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_names = result.names

                print(f"   检测详情:")
                for j, (cls, conf) in enumerate(zip(classes, confidences)):
                    class_name = class_names[int(cls)]
                    print(f"     {j + 1}. {class_name}: {conf:.3f}")

            results_summary.append({
                'config': config['name'],
                'detections': detections,
                'conf': config['conf'],
                'iou': config['iou'],
                'imgsz': config['imgsz']
            })

        except Exception as e:
            print(f"   ❌ 检测失败: {e}")
            results_summary.append({
                'config': config['name'],
                'detections': 0,
                'error': str(e)
            })

    # 总结报告
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)

    best_config = None
    max_detections = 0

    for result in results_summary:
        if 'error' not in result:
            print(f"{result['config']}: {result['detections']} 个目标")
            if result['detections'] > max_detections:
                max_detections = result['detections']
                best_config = result
        else:
            print(f"{result['config']}: 失败 - {result['error']}")

    if best_config:
        print(f"\n🏆 最佳配置: {best_config['config']}")
        print(f"   检测到 {best_config['detections']} 个目标")
        print(f"   推荐参数:")
        print(f"     conf = {best_config['conf']}")
        print(f"     iou = {best_config['iou']}")
        print(f"     imgsz = {best_config['imgsz']}")

        # 保存最佳结果
        save_path = f"best_result_{best_config['config'].replace(' ', '_')}.jpg"
        try:
            results = model(
                test_image,
                conf=best_config['conf'],
                iou=best_config['iou'],
                imgsz=best_config['imgsz'],
                save=True,
                project="test_results",
                name="best_config"
            )
            print(f"   结果已保存到: test_results/best_config/{save_path}")
        except:
            pass
    else:
        print("\n❌ 所有配置都未能检测到目标")
        print("建议:")
        print("1. 检查图片中是否确实有目标")
        print("2. 重新训练模型")
        print("3. 增加更多大目标样本到训练集")


if __name__ == "__main__":
    test_detection_with_multiple_settings()
