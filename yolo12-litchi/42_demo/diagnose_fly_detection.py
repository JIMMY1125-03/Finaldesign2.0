#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
诊断苍蝇检测问题的脚本
分析为什么识别不出来
"""
import cv2
import os
from ultralytics import YOLO
import numpy as np

def diagnose_fly_detection():
    """诊断苍蝇检测问题"""
    
    print("=" * 60)
    print("苍蝇检测问题诊断")
    print("=" * 60)
    
    # 检查模型路径
    model_path = r"D:\校内\新建文件夹\Finaldesign2.0\yolo12-litchi\42_demo\runs\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型")
        return
    
    # 数据集类别信息
    dataset_classes = ['麻蝇', '蛾类', '黑斑果蝇', '粪蝇', '蚤蝇']
    print(f"📋 训练数据集类别: {dataset_classes}")
    print()
    
    # 分析你上传的图片
    print("🔍 图片分析:")
    print("   你上传的图片特征:")
    print("   - 大复眼，橙红色")
    print("   - 胸部有黑白条纹")
    print("   - 透明翅膀，有深色翅脉")
    print("   - 这是典型的家蝇/丽蝇特征")
    print()
    
    print("❌ 问题诊断:")
    print("   1. 类别不匹配：你的图片是普通苍蝇，但训练数据中没有'苍蝇'类别")
    print("   2. 可能的原因：")
    print("      - 训练数据中只有特定的蝇类（麻蝇、黑斑果蝇等）")
    print("      - 模型没有见过这种类型的苍蝇")
    print("      - 需要重新标注或增加'苍蝇'类别")
    print()
    
    # 测试当前模型
    print("🧪 测试当前模型:")
    model = YOLO(model_path)
    
    # 创建测试图片路径
    test_image_path = input("请输入你的苍蝇图片路径（或按回车跳过）: ").strip()
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"   测试图片: {test_image_path}")
        
        # 使用多种参数测试
        test_configs = [
            {"name": "极低阈值", "conf": 0.01, "iou": 0.3, "imgsz": 1280},
            {"name": "超低阈值", "conf": 0.005, "iou": 0.2, "imgsz": 1536},
            {"name": "最大尺寸", "conf": 0.001, "iou": 0.1, "imgsz": 1920}
        ]
        
        for config in test_configs:
            print(f"   测试 {config['name']}: conf={config['conf']}, imgsz={config['imgsz']}")
            try:
                results = model(
                    test_image_path,
                    conf=config['conf'],
                    iou=config['iou'],
                    imgsz=config['imgsz'],
                    verbose=False
                )
                
                result = results[0]
                detections = len(result.boxes) if result.boxes is not None else 0
                print(f"     结果: {detections} 个目标")
                
                if detections > 0:
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    for i, (cls, conf) in enumerate(zip(classes, confidences)):
                        class_name = dataset_classes[int(cls)]
                        print(f"       {i+1}. {class_name}: {conf:.4f}")
            except Exception as e:
                print(f"     错误: {e}")
    
    print()
    print("💡 解决方案:")
    print("   方案1: 重新标注数据集")
    print("      - 在数据集中添加'苍蝇'类别")
    print("      - 标注更多普通苍蝇的图片")
    print("      - 重新训练模型")
    print()
    print("   方案2: 使用通用检测模型")
    print("      - 使用预训练的通用目标检测模型")
    print("      - 或者使用昆虫检测的预训练模型")
    print()
    print("   方案3: 调整现有模型")
    print("      - 将苍蝇归类到最相似的类别（如'麻蝇'）")
    print("      - 增加更多苍蝇样本到现有类别")
    print()
    print("   方案4: 创建新的检测脚本")
    print("      - 使用更低的置信度阈值")
    print("      - 使用更大的输入尺寸")
    print("      - 使用TTA增强")

def create_fly_detection_script():
    """创建专门的苍蝇检测脚本"""
    
    script_content = '''#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
专门的苍蝇检测脚本
使用极低阈值和多种参数组合
"""
import cv2
import os
from ultralytics import YOLO
import numpy as np

def detect_fly_with_low_threshold(image_path, model_path):
    """使用极低阈值检测苍蝇"""
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        return
    
    model = YOLO(model_path)
    
    # 极低阈值配置
    configs = [
        {"conf": 0.001, "iou": 0.1, "imgsz": 1920, "name": "超低阈值+大尺寸"},
        {"conf": 0.005, "iou": 0.2, "imgsz": 1536, "name": "极低阈值+中尺寸"},
        {"conf": 0.01, "iou": 0.3, "imgsz": 1280, "name": "低阈值+标准尺寸"},
        {"conf": 0.05, "iou": 0.4, "imgsz": 1280, "augment": True, "name": "低阈值+TTA"},
    ]
    
    print("=" * 60)
    print("苍蝇检测测试")
    print("=" * 60)
    
    best_result = None
    max_detections = 0
    
    for config in configs:
        print(f"\\n🔍 测试: {config['name']}")
        print(f"   置信度: {config['conf']}")
        print(f"   IOU: {config['iou']}")
        print(f"   尺寸: {config['imgsz']}")
        
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
                class_names = ['麻蝇', '蛾类', '黑斑果蝇', '粪蝇', '蚤蝇']
                
                for i, (cls, conf) in enumerate(zip(classes, confidences)):
                    class_name = class_names[int(cls)]
                    print(f"     {i+1}. {class_name}: {conf:.4f}")
                
                if detections > max_detections:
                    max_detections = detections
                    best_result = config
                    
        except Exception as e:
            print(f"   错误: {e}")
    
    if best_result:
        print(f"\\n🏆 最佳配置: {best_result['name']}")
        print(f"   检测到 {max_detections} 个目标")
        
        # 保存最佳结果
        try:
            results = model(
                image_path,
                conf=best_result['conf'],
                iou=best_result['iou'],
                imgsz=best_result['imgsz'],
                augment=best_result.get('augment', False),
                save=True,
                project="fly_detection_results",
                name="best_config"
            )
            print(f"   结果已保存到: fly_detection_results/best_config/")
        except:
            pass
    else:
        print("\\n❌ 所有配置都未能检测到目标")
        print("建议:")
        print("1. 检查图片中是否确实有目标")
        print("2. 重新训练模型，添加'苍蝇'类别")
        print("3. 使用通用目标检测模型")

if __name__ == "__main__":
    image_path = input("请输入苍蝇图片路径: ").strip()
    model_path = r"D:\\校内\\新建文件夹\\Finaldesign2.0\\yolo12-litchi\\42_demo\\runs\\best.pt"
    detect_fly_with_low_threshold(image_path, model_path)
'''
    
    with open("42_demo/detect_fly_special.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 已创建专门的苍蝇检测脚本: 42_demo/detect_fly_special.py")

if __name__ == "__main__":
    diagnose_fly_detection()
    create_fly_detection_script()
