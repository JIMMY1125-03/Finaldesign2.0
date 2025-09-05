#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
解决苍蝇检测问题的脚本
"""
import os
from ultralytics import YOLO

print("=" * 60)
print("苍蝇检测问题诊断")
print("=" * 60)

# 检查模型
model_path = r"D:\校内\新建文件夹\Finaldesign2.0\yolo12-litchi\42_demo\runs\best.pt"

if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    exit()

print("✅ 模型文件存在")

# 数据集类别
classes = ['麻蝇', '蛾类', '黑斑果蝇', '粪蝇', '蚤蝇']
print(f"📋 训练数据集类别: {classes}")
print()

print("🔍 问题分析:")
print("   你上传的图片是普通苍蝇，但训练数据中没有'苍蝇'类别")
print("   只有: 麻蝇、蛾类、黑斑果蝇、粪蝇、蚤蝇")
print()

print("❌ 这就是为什么识别不出来的原因！")
print()

print("💡 解决方案:")
print("   1. 重新标注数据集，添加'苍蝇'类别")
print("   2. 将苍蝇归类到最相似的类别（如'麻蝇'）")
print("   3. 使用极低阈值尝试检测")
print("   4. 重新训练模型")
print()

# 测试模型
try:
    model = YOLO(model_path)
    print("🧪 测试模型加载...")
    print(f"   模型类别数: {model.model[-1].nc}")
    print(f"   模型类别名: {model.names}")
    print()
except Exception as e:
    print(f"   模型加载错误: {e}")
    print()

print("📝 建议操作:")
print("   1. 运行: python 42_demo/detect_fly_special.py")
print("   2. 或者重新训练模型，添加'苍蝇'类别")
print()

print("🚀 立即解决方案:")
print("   使用极低阈值检测，将苍蝇归类到最相似的类别")
print("   运行以下命令测试:")
print("   python 42_demo/detect_fly_special.py")
