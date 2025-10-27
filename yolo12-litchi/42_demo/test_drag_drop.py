#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# 测试拖拽上传功能
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from step3_start_window import MainWindow, DragDropImageLabel
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    
    print("✅ 导入成功！")
    
    # 创建应用程序实例
    app = QApplication([])
    
    # 测试DragDropImageLabel类
    drag_label = DragDropImageLabel()
    print("✅ DragDropImageLabel创建成功！")
    
    # 测试MainWindow
    window = MainWindow()
    print("✅ MainWindow创建成功！")
    
    # 检查拖拽功能是否启用
    if window.acceptDrops():
        print("✅ 窗口拖拽功能已启用")
    else:
        print("❌ 窗口拖拽功能未启用")
    
    # 检查左侧图片是否为拖拽标签
    if isinstance(window.left_img, DragDropImageLabel):
        print("✅ 左侧图片支持拖拽上传")
    else:
        print("❌ 左侧图片不支持拖拽上传")
    
    # 检查是否有处理拖拽的方法
    if hasattr(window, 'handle_dropped_file'):
        print("✅ 拖拽处理方法存在")
    else:
        print("❌ 拖拽处理方法不存在")
    
    print("✅ 拖拽上传功能测试完成！")
    print("\n🎯 功能说明：")
    print("1. 可以直接拖拽图片文件到左侧图片区域")
    print("2. 可以拖拽图片文件到窗口任意位置")
    print("3. 支持的格式：jpg, jpeg, png, tif, tiff, bmp, gif")
    print("4. 支持快捷键 Ctrl+O 打开文件对话框")
    print("5. 文件大小限制：50MB以内")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()