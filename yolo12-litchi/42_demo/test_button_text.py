#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
测试按钮文字显示的简单脚本
"""

import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class TestButtonWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("按钮文字显示测试")
        self.setGeometry(100, 100, 400, 300)
        
        layout = QVBoxLayout(self)
        
        # 测试标签
        label = QLabel("按钮文字显示测试")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont('Microsoft YaHei UI', 14, QFont.Bold))
        layout.addWidget(label)
        
        # 测试不同样式的按钮
        
        # 1. 基本按钮
        btn1 = QPushButton("基本按钮")
        btn1.setFont(QFont('Microsoft YaHei UI', 12, QFont.Bold))
        layout.addWidget(btn1)
        
        # 2. 带样式的按钮
        btn2 = QPushButton("📁 选择图片")
        btn2.setFont(QFont('Microsoft YaHei UI', 12, QFont.Bold))
        btn2.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2196F3, stop:1 #1976D2);
                color: white !important;
                border: none;
                border-radius: 12px;
                padding: 12px 18px;
                font-weight: bold !important;
                font-size: 12px !important;
                font-family: 'Microsoft YaHei UI', 'SimHei', 'Arial' !important;
                text-align: center;
            }
        """)
        layout.addWidget(btn2)
        
        # 3. 工具栏样式按钮
        btn3 = QPushButton("缩小")
        btn3.setFont(QFont('Microsoft YaHei UI', 12, QFont.Bold))
        btn3.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #007BFF, stop:1 #0056B3);
                color: white !important;
                border: 1px solid #0056B3;
                border-radius: 4px;
                font-weight: bold !important;
                font-size: 12px !important;
                font-family: 'Microsoft YaHei UI', 'SimHei', 'Arial' !important;
                padding: 3px 6px;
                min-width: 55px;
                min-height: 28px;
                text-align: center;
            }
        """)
        layout.addWidget(btn3)
        
        # 4. 快捷问题按钮
        btn4 = QPushButton("🔍 分析检测结果")
        btn4.setFont(QFont('Microsoft YaHei UI', 9, QFont.Bold))
        btn4.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E8F5E8, stop:1 #C8E6C9);
                color: #2E7D32 !important;
                border: 1px solid #4CAF50;
                border-radius: 14px;
                padding: 4px 8px;
                font-weight: bold !important;
                font-size: 9px !important;
                font-family: 'Microsoft YaHei UI', 'SimHei', 'Arial' !important;
                text-align: center;
            }
        """)
        layout.addWidget(btn4)
        
        # 连接点击事件
        btn1.clicked.connect(lambda: print("基本按钮被点击"))
        btn2.clicked.connect(lambda: print("选择图片按钮被点击"))
        btn3.clicked.connect(lambda: print("缩小按钮被点击"))
        btn4.clicked.connect(lambda: print("分析检测结果按钮被点击"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    window = TestButtonWindow()
    window.show()
    
    print("🔍 按钮文字显示测试窗口已打开")
    print("📋 请检查以下内容：")
    print("   1. 所有按钮是否显示文字")
    print("   2. 文字是否清晰可见")
    print("   3. 字体大小是否合适")
    print("   4. 颜色对比度是否足够")
    
    sys.exit(app.exec())