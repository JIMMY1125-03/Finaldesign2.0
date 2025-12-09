#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy  # 用于图像复制
import os  # 用于系统路径查找
import shutil  # 用于复制
from PySide6.QtGui import *  # GUI组件
from PySide6.QtCore import *  # 字体、边距等系统变量
from PySide6.QtWidgets import *  # 窗口等小组件
import sys  # 系统库
import cv2  # opencv图像处理
import torch  # 深度学习框架
import os.path as osp  # 路径查找
import time  # 时间计算
from ultralytics import YOLO  # yolo核心算法
from ultralytics.utils.torch_utils import select_device
import numpy as np
import re
import json
import html
from typing import Optional
import datetime
import sqlite3
from collections import defaultdict, Counter
# 添加动画和主题相关导入
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QSequentialAnimationGroup, QTimer
from PySide6.QtWidgets import QGraphicsOpacityEffect, QGraphicsDropShadowEffect
# 添加API调用相关导入
import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role

# 设置API密钥
dashscope.api_key = 'sk-0cb0cdd81cc74719b30bd6458338a340'

# 支持拖拽的图片标签类
class DragDropImageLabel(QLabel):
    # 定义信号，当文件被拖拽时发出
    fileDropped = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)  # 启用拖拽接收
        self.setMinimumSize(200, 200)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #007BFF;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FF, stop:1 #E3F2FD);
                color: #007BFF;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel:hover {
                border: 2px dashed #0056B3;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:1 #BBDEFB);
                color: #0056B3;
            }
        """)
        self.setText("📁 拖拽图片到这里\n或点击上传按钮")
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
    
    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否包含图片文件
            urls = event.mimeData().urls()
            if urls and self.is_image_file(urls[0].toLocalFile()):
                event.acceptProposedAction()
                # 改变样式提示用户可以放置
                self.setStyleSheet("""
                    QLabel {
                        border: 3px solid #28A745;
                        border-radius: 10px;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #E8F5E8, stop:1 #C8E6C9);
                        color: #28A745;
                        font-size: 14px;
                        font-weight: bold;
                    }
                """)
                self.setText("✅ 释放以上传图片")
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        # 恢复原始样式
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #007BFF;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FF, stop:1 #E3F2FD);
                color: #007BFF;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel:hover {
                border: 2px dashed #0056B3;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:1 #BBDEFB);
                color: #0056B3;
            }
        """)
        self.setText("📁 拖拽图片到这里\n或点击上传按钮")
    
    def dropEvent(self, event):
        """拖拽放置事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self.is_image_file(file_path):
                    # 发出文件拖拽信号
                    self.fileDropped.emit(file_path)
                    event.acceptProposedAction()
                    
                    # 显示成功样式
                    self.setStyleSheet("""
                        QLabel {
                            border: 3px solid #28A745;
                            border-radius: 10px;
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #E8F5E8, stop:1 #C8E6C9);
                            color: #28A745;
                            font-size: 14px;
                            font-weight: bold;
                        }
                    """)
                    self.setText("✅ 图片上传成功！")
                    
                    # 2秒后恢复原始样式
                    QTimer.singleShot(2000, self.reset_style)
                else:
                    # 显示错误样式
                    self.setStyleSheet("""
                        QLabel {
                            border: 3px solid #DC3545;
                            border-radius: 10px;
                            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #F8D7DA, stop:1 #F5C6CB);
                            color: #DC3545;
                            font-size: 14px;
                            font-weight: bold;
                        }
                    """)
                    self.setText("❌ 请上传图片文件\n支持格式：jpg, png, jpeg, tif")
                    
                    # 2秒后恢复原始样式
                    QTimer.singleShot(2000, self.reset_style)
        else:
            event.ignore()
    
    def is_image_file(self, file_path):
        """检查文件是否为支持的图片格式"""
        if not file_path:
            return False
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif']
        file_extension = os.path.splitext(file_path.lower())[1]
        return file_extension in supported_formats
    
    def reset_style(self):
        """重置为原始样式"""
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #007BFF;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FF, stop:1 #E3F2FD);
                color: #007BFF;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel:hover {
                border: 2px dashed #0056B3;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:1 #BBDEFB);
                color: #0056B3;
            }
        """)
        self.setText("📁 拖拽图片到这里\n或点击上传按钮")
    
    def set_image(self, pixmap):
        """设置图片并调整样式"""
        self.setPixmap(pixmap)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #28A745;
                border-radius: 10px;
                background: white;
            }
        """)
        self.setText("")  # 清除文本，只显示图片

# 主题管理类
class ThemeManager:
    def __init__(self):
        self.current_theme = "light"  # 默认浅色主题
        self.themes = {
            "light": {
                "primary_bg": "#F8F9FA",
                "secondary_bg": "#FFFFFF", 
                "accent_color": "#007BFF",
                "text_color": "#2C3E50",
                "border_color": "#E9ECEF",
                "hover_color": "#0056B3",
                "success_color": "#28A745",
                "warning_color": "#FFC107",
                "danger_color": "#DC3545",
                "info_color": "#17A2B8",
                "shadow_color": "rgba(0, 0, 0, 0.1)"
            },
            "dark": {
                "primary_bg": "#1E1E1E",
                "secondary_bg": "#2D2D2D",
                "accent_color": "#0D7377",
                "text_color": "#E9EEF6",
                "border_color": "#404040",
                "hover_color": "#14A085",
                "success_color": "#198754",
                "warning_color": "#FFC107",
                "danger_color": "#DC3545",
                "info_color": "#0DCAF0",
                "shadow_color": "rgba(255, 255, 255, 0.1)"
            }
        }
    
    def get_theme(self, theme_name=None):
        if theme_name is None:
            theme_name = self.current_theme
        return self.themes.get(theme_name, self.themes["light"])
    
    def switch_theme(self):
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        return self.current_theme
    
    def get_button_style(self, color_type="accent"):
        theme = self.get_theme()
        base_color = theme.get(f"{color_type}_color", theme["accent_color"])
        hover_color = theme["hover_color"]
        
        return f"""
            QPushButton {{
                background: {base_color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
                margin: 5px;
            }}
            QPushButton:hover {{
                background: {hover_color};
                transform: translateY(-2px);
            }}
            QPushButton:pressed {{
                background: {theme["border_color"]};
                transform: translateY(0px);
            }}
        """
    
    def get_label_style(self, style_type="primary"):
        theme = self.get_theme()
        if style_type == "primary":
            return f"""
                QLabel {{
                    color: {theme["text_color"]};
                    background: {theme["secondary_bg"]};
                    border: 2px solid {theme["accent_color"]};
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px;
                    font-weight: bold;
                }}
            """
        elif style_type == "info":
            return f"""
                QLabel {{
                    color: {theme["text_color"]};
                    background: {theme["primary_bg"]};
                    border: 2px solid {theme["info_color"]};
                    border-radius: 8px;
                    padding: 8px;
                    margin: 5px;
                    font-weight: bold;
                }}
            """
    
    def get_input_style(self):
        theme = self.get_theme()
        return f"""
            QLineEdit {{
                background: {theme["secondary_bg"]};
                border: 2px solid {theme["border_color"]};
                border-radius: 20px;
                padding: 12px 20px;
                font-size: 14px;
                margin: 5px;
                color: {theme["text_color"]};
            }}
            QLineEdit:focus {{
                border: 2px solid {theme["accent_color"]};
                background: {theme["primary_bg"]};
            }}
            QLineEdit:hover {{
                border: 2px solid {theme["accent_color"]};
            }}
        """
    
    def get_textedit_style(self):
        theme = self.get_theme()
        return f"""
            QTextEdit {{
                background: {theme["secondary_bg"]};
                border: 2px solid {theme["border_color"]};
                border-radius: 12px;
                padding: 15px;
                font-size: 13px;
                line-height: 1.4;
                selection-background-color: {theme["accent_color"]};
                color: {theme["text_color"]};
            }}
            QTextEdit:focus {{
                border: 2px solid {theme["accent_color"]};
                box-shadow: 0 0 10px {theme["shadow_color"]};
            }}
        """
    
    def get_splitter_style(self, color_type="accent"):
        theme = self.get_theme()
        base_color = theme.get(f"{color_type}_color", theme["accent_color"])
        hover_color = theme["hover_color"]
        
        return f"""
            QSplitter::handle {{
                background: {base_color};
                border: 2px solid {base_color};
                border-radius: 5px;
                margin: 2px;
            }}
            QSplitter::handle:hover {{
                background: {hover_color};
                border: 2px solid {hover_color};
                box-shadow: 0 0 8px {theme["shadow_color"]};
            }}
            QSplitter::handle:pressed {{
                background: {theme["border_color"]};
                border: 2px solid {theme["border_color"]};
            }}
        """
    
    def get_main_window_style(self):
        theme = self.get_theme()
        return f"""
            QTabWidget::pane {{
                border: 1px solid {theme["border_color"]};
                background: {theme["primary_bg"]};
                border-radius: 8px;
            }}
            QTabWidget::tab-bar {{
                alignment: center;
            }}
            QTabBar::tab {{
                background: {theme["secondary_bg"]};
                color: {theme["text_color"]};
                padding: 12px 24px;
                margin: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {theme["border_color"]};
            }}
            QTabBar::tab:selected {{
                background: {theme["accent_color"]};
                color: white;
                border: 2px solid {theme["accent_color"]};
            }}
            QTabBar::tab:hover {{
                background: {theme["hover_color"]};
                color: white;
                border: 2px solid {theme["hover_color"]};
            }}
            QWidget {{
                background: {theme["primary_bg"]};
                color: {theme["text_color"]};
            }}
        """

# 动画管理类
class AnimationManager:
    def __init__(self):
        self.animations = []
    
    def fade_in_widget(self, widget, duration=500):
        """淡入动画"""
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        self.animations.append(animation)
        animation.start()
        return animation
    
    def fade_out_widget(self, widget, duration=300):
        """淡出动画"""
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        self.animations.append(animation)
        animation.start()
        return animation
    
    def slide_in_widget(self, widget, direction="left", duration=400):
        """滑入动画"""
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        
        start_rect = widget.geometry()
        end_rect = widget.geometry()
        
        if direction == "left":
            start_rect.moveLeft(-widget.width())
        elif direction == "right":
            start_rect.moveLeft(widget.parent().width())
        elif direction == "top":
            start_rect.moveTop(-widget.height())
        elif direction == "bottom":
            start_rect.moveTop(widget.parent().height())
        
        animation.setStartValue(start_rect)
        animation.setEndValue(end_rect)
        
        self.animations.append(animation)
        animation.start()
        return animation
    
    def bounce_widget(self, widget, duration=600):
        """弹跳动画"""
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.OutBounce)
        
        start_rect = widget.geometry()
        end_rect = widget.geometry()
        
        # 稍微放大然后恢复
        mid_rect = QRect(start_rect)
        mid_rect.adjust(-10, -10, 10, 10)
        
        animation.setStartValue(start_rect)
        animation.setKeyValueAt(0.5, mid_rect)
        animation.setEndValue(end_rect)
        
        self.animations.append(animation)
        animation.start()
        return animation
    
    def pulse_widget(self, widget, duration=1000, repeat=3):
        """脉冲动画"""
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setKeyValueAt(0.5, 0.3)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.InOutSine)
        animation.setLoopCount(repeat)
        
        self.animations.append(animation)
        animation.start()
        return animation
    
    def shake_widget(self, widget, duration=500):
        """摇摆动画"""
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.InOutSine)
        
        original_rect = widget.geometry()
        
        # 创建摇摆关键帧
        animation.setStartValue(original_rect)
        
        shake_rect1 = QRect(original_rect)
        shake_rect1.moveLeft(original_rect.left() + 10)
        animation.setKeyValueAt(0.2, shake_rect1)
        
        shake_rect2 = QRect(original_rect)
        shake_rect2.moveLeft(original_rect.left() - 10)
        animation.setKeyValueAt(0.4, shake_rect2)
        
        shake_rect3 = QRect(original_rect)
        shake_rect3.moveLeft(original_rect.left() + 5)
        animation.setKeyValueAt(0.6, shake_rect3)
        
        shake_rect4 = QRect(original_rect)
        shake_rect4.moveLeft(original_rect.left() - 5)
        animation.setKeyValueAt(0.8, shake_rect4)
        
        animation.setEndValue(original_rect)
        
        self.animations.append(animation)
        animation.start()
        return animation
    
    def add_shadow_effect(self, widget, color=QColor(0, 0, 0, 80)):
        """添加阴影效果"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(color)
        shadow.setOffset(3, 3)
        widget.setGraphicsEffect(shadow)
        return shadow

# 常用的字符串常量
WINDOW_TITLE = "农业害虫检测系统"  # 系统上方标题
WELCOME_SENTENCE = "欢迎使用基于yolov12的农业害虫检测系统"  # 欢迎的句子
ICON_IMAGE = "images/UI/lufei.png"  # 系统logo界面
IMAGE_LEFT_INIT = "images/UI/up.jpeg"  # 图片检测界面初始化左侧图像
IMAGE_RIGHT_INIT = "images/UI/right.jpeg"  # 图片检测界面初始化右侧图像
ZHU_IMAGE_PATH = "images/UI/lizhi.jpg"
PREDICT_PROJECT_DIR = "record/yolo_outputs"  # 固定保存目录（ultralytics 内部保存）
PREDICT_RUN_NAME = "fixed"  # 固定子目录名
USERNAME = "123456"
PASSWORD = "123456"
LOGIN_TITLE = "😁 欢迎使用农业害虫检测系统"
USERS_DB_PATH = os.path.join(os.path.dirname(__file__), "users.json")


# 添加ChatBot类
class ChatBot:
    def __init__(self):
        self.conversation_history = []  # 存储对话历史
        self.detection_results = None  # 存储检测结果
        # 可以添加系统提示来设定AI的性格和能力
        self.system_prompt = "你是一个专业的农业病虫害防治专家，专门负责农业害虫的识别和治理建议。请用中文回答用户的问题。"
        self.conversation_history.append({'role': Role.SYSTEM, 'content': self.system_prompt})

    def set_detection_results(self, results):
        """设置检测结果"""
        print(f"=== ChatBot接收检测结果 ===")
        print(f"接收到的results: {results}")
        print(f"结果类型: {type(results)}")
        print(f"结果数量: {len(results) if results else 0}")
        print("============================")
        
        self.detection_results = results

    def get_detection_context(self):
        """获取检测结果的上下文信息"""
        if not self.detection_results:
            return "目前还没有进行图片检测，请先上传图片并进行检测。"

        print(f"=== 生成检测上下文 ===")
        print(f"detection_results: {self.detection_results}")
        
        context = "根据刚才的图片检测结果：\n"
        for class_name, count in self.detection_results.items():
            context += f"- 检测到 {count} 个 {class_name}\n"
            print(f"添加到上下文: {class_name}: {count}")
        
        print(f"最终上下文: {context}")
        print("=====================")
        return context

    def chat(self, user_input, include_detection=True):
        """处理用户输入并返回AI回复"""
        # 1. 将用户输入加入历史
        self.conversation_history.append({'role': Role.USER, 'content': user_input})

        try:
            # 2. 如果有检测结果且用户询问治理相关的问题，将检测结果作为上下文
            if include_detection and self.detection_results and any(
                    keyword in user_input for keyword in ['治理', '防治', '处理', '消灭', '杀灭', '控制', '预防']):
                # 在用户输入前添加检测结果上下文
                detection_context = self.get_detection_context()
                classes = ", ".join(self.detection_results.keys())
                format_rules = (
                    "请输出JSON，键为 categories，值是数组。数组项结构固定：\n"
                    "{\"name\": 类别名, \"count\": 数量, \"physical\": [要点...], \"biological\": [要点...], \"chemical\": [要点...], \"other\": [要点...] }\n"
                    "严格只输出一段合法JSON，不要任何解释、前后缀或Markdown代码块。"
                )
                enhanced_input = (
                    f"{detection_context}\n\n已识别类别：{classes}\n\n{format_rules}\n\n"
                    f"用户问题：{user_input}\n"
                )

                # 更新对话历史中的用户输入
                self.conversation_history[-1]['content'] = enhanced_input

                # 同时更新系统提示，让AI知道这是病虫害治理咨询
                system_prompt = (
                    "你是一个专业的农业病虫害防治专家，专门负责农业害虫的识别和治理建议。"
                    "根据检测结果输出结构化JSON，包含每个类别的物理/生物/化学/其他要点列表。"
                )
                self.conversation_history[0]['content'] = system_prompt

            # 3. 调用API，传入整个对话历史
            response = Generation.call(
                model='qwen-turbo',
                messages=self.conversation_history,
                result_format='message',
                temperature=0.8
            )

            if response.status_code == 200:
                # 4. 获取AI回复并加入历史
                ai_reply = response.output.choices[0].message.content
                self.conversation_history.append({'role': Role.ASSISTANT, 'content': ai_reply})

                # 5. (可选) 防止历史过长，可以设置一个最大轮数
                if len(self.conversation_history) > 10:  # 保留最近10轮对话（包括系统提示）
                    # 保留系统提示和最近的对话
                    self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-9:]

                return ai_reply
            else:
                return f"抱歉，我遇到了一个错误：{response.message}"

        except Exception as e:
            return f"调用出错：{e}"

    def print_conversation(self):
        """打印当前对话历史（调试用）"""
        for msg in self.conversation_history:
            if msg['role'] == Role.SYSTEM:
                continue  # 跳过系统提示
            speaker = "您" if msg['role'] == Role.USER else "AI"
            print(f"{speaker}: {msg['content']}")


class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)  # 系统界面标题
        self.resize(1400, 900)  # 增加窗口大小以容纳AI聊天框
        self.setWindowIcon(QIcon(ICON_IMAGE))  # 系统logo图像
        self.output_size = 480  # 上传的图像和视频在系统界面上显示的大小
        self.img2predict = ""  # 要进行预测的图像路径
        # 用来进行设置的参数
        self.conf_thres = 0.2  # 置信度的阈值
        self.iou_thres = 0.5  # NMS操作的时候 IOU过滤的阈值
        self.imgsz = 1280  # 推理图像尺寸（较大尺寸有助于识别放大目标）
        self.use_tta = False  # 是否启用测试时增强（TTA）
        self.save_txt = False
        self.save_conf = False
        self.save_crop = False

        # 初始化主题和动画管理器
        self.theme_manager = ThemeManager()
        self.animation_manager = AnimationManager()
        
        # 初始化ChatBot
        self.chatbot = ChatBot()
        # 存储最新的检测结果
        self.latest_detection_results = {}

        # self.model_path = "runs/detect/yolo11-n/weights/best.pt"  # todo 指明模型加载的位置的设备
        self.model_path = r"D:\JM\毕业设计\model-cache\best.pt"  # todo 指明模型加载的位置的设备
        self.model = self.model_load(weights=self.model_path)

        # 设置现代化样式
        self.setup_modern_style()

        self.initUI()  # 初始化图形化界面
        
        # 应用初始主题
        self.apply_theme()
        
        # 刷新配置界面显示
        QTimer.singleShot(100, self.refresh_config_values)
        
        # 添加启动动画
        self.show_startup_animation()

    def setup_modern_style(self):
        """设置现代化样式"""
        # 基础样式将通过主题管理器动态设置
        pass
    
    def apply_theme(self):
        """应用当前主题"""
        # 应用主窗口样式
        self.setStyleSheet(self.theme_manager.get_main_window_style())
        
        # 更新所有子组件的样式
        self.update_all_styles()
    
    def update_all_styles(self):
        """更新所有组件的样式"""
        # 更新所有按钮样式
        buttons = self.findChildren(QPushButton)
        for button in buttons:
            if "上传" in button.text() or "检测" in button.text():
                button.setStyleSheet(self.theme_manager.get_button_style("accent"))
            elif "发送" in button.text():
                button.setStyleSheet(self.theme_manager.get_button_style("success"))
            elif "清空" in button.text():
                button.setStyleSheet(self.theme_manager.get_button_style("danger"))
            elif "查看" in button.text():
                button.setStyleSheet(self.theme_manager.get_button_style("info"))
            elif "主题" in button.text() or "模式" in button.text():
                button.setStyleSheet(self.theme_manager.get_button_style("warning"))
            else:
                button.setStyleSheet(self.theme_manager.get_button_style("accent"))
        
        # 更新所有标签样式
        labels = self.findChildren(QLabel)
        theme = self.theme_manager.get_theme()
        
        for label in labels:
            if "功能" in label.text() or "助手" in label.text() or "配置" in label.text():
                label.setStyleSheet(self.theme_manager.get_label_style("primary"))
            elif "检测结果" in label.text():
                label.setStyleSheet(self.theme_manager.get_label_style("info"))
            elif any(keyword in label.text() for keyword in [
                "系统图像显示大小", "推理图像尺寸", "检测模型置信度阈值", "检测模型IOU阈值",
                "推理时是否保存txt文件", "启用测试时增强", "推理时是否保存置信度", "推理时是否保存切片文件"
            ]):
                # 配置界面的标签使用主题颜色
                label.setStyleSheet(f"""
                    QLabel {{
                        color: {theme["text_color"]};
                        font-weight: bold;
                        font-size: 13px;
                        background: transparent;
                        border: none;
                        padding: 5px;
                    }}
                """)
        
        # 更新输入框样式
        line_edits = self.findChildren(QLineEdit)
        for line_edit in line_edits:
            # 检查是否是配置界面的输入框，如果是则保持特殊样式
            if hasattr(self, 'config_output_size_value') and line_edit in [
                self.config_output_size_value, 
                self.config_imgsz_value, 
                self.config_conf_thres_value, 
                self.config_iou_thres_value
            ]:
                # 配置界面输入框使用固定样式确保可见性
                config_input_style = """
                    QLineEdit {
                        background: white;
                        border: 2px solid #E9ECEF;
                        border-radius: 6px;
                        padding: 8px;
                        font-size: 12px;
                        color: black !important;
                        font-weight: bold;
                    }
                    QLineEdit:focus {
                        border: 2px solid #007BFF;
                        background: #F8F9FF;
                        color: black !important;
                    }
                    QLineEdit:hover {
                        border: 2px solid #007BFF;
                        color: black !important;
                    }
                """
                line_edit.setStyleSheet(config_input_style)
            else:
                # 其他输入框使用主题样式
                line_edit.setStyleSheet(self.theme_manager.get_input_style())
        
        # 更新配置界面的单选按钮样式
        radio_buttons = self.findChildren(QRadioButton)
        for radio_button in radio_buttons:
            if hasattr(self, 'config_save_txt_value') and radio_button in [
                self.config_save_txt_value, self.config_tta_value, 
                self.config_save_conf_value, self.config_save_crop_value
            ]:
                # 配置界面单选按钮使用主题适配样式
                theme = self.theme_manager.get_theme()
                radio_style = f"""
                    QRadioButton {{
                        font-size: 12px;
                        color: {theme["text_color"]};
                        font-weight: bold;
                        background: transparent;
                    }}
                    QRadioButton::indicator {{
                        width: 16px;
                        height: 16px;
                        border-radius: 8px;
                        border: 2px solid {theme["accent_color"]};
                        background: {theme["secondary_bg"]};
                    }}
                    QRadioButton::indicator:checked {{
                        background: {theme["accent_color"]};
                    }}
                    QRadioButton::indicator:hover {{
                        border: 2px solid {theme["hover_color"]};
                    }}
                """
                radio_button.setStyleSheet(radio_style)
        
        # 更新文本编辑器样式
        text_edits = self.findChildren(QTextEdit)
        for text_edit in text_edits:
            text_edit.setStyleSheet(self.theme_manager.get_textedit_style())
        
        # 更新分割器样式
        if hasattr(self, 'main_splitter'):
            self.main_splitter.setStyleSheet(self.theme_manager.get_splitter_style("accent"))
        if hasattr(self, 'left_splitter'):
            self.left_splitter.setStyleSheet(self.theme_manager.get_splitter_style("accent"))
        if hasattr(self, 'chat_splitter'):
            self.chat_splitter.setStyleSheet(self.theme_manager.get_splitter_style("success"))
        if hasattr(self, 'img_splitter'):
            self.img_splitter.setStyleSheet(self.theme_manager.get_splitter_style("warning"))

    # 模型初始化
    @torch.no_grad()
    def model_load(self, weights=""):
        """
        模型加载
        """
        # 权重存在性检查与回退
        fallback = os.path.join(os.path.dirname(__file__), "yolo11n.pt")
        candidate = weights if weights and osp.exists(weights) else fallback
        if not osp.exists(candidate):
            raise FileNotFoundError(f"模型权重不存在：{weights}，且未找到回退权重：{fallback}")
        model_loaded = YOLO(candidate)
        return model_loaded

    def initUI(self):
        """
        图形化界面初始化
        """
        # ********************* 图片识别界面（集成AI聊天） *****************************
        font_title = QFont('Microsoft YaHei UI', 16, QFont.Bold)
        font_main = QFont('Microsoft YaHei UI', 12)
        img_detection_widget = QWidget()

        # 使用QSplitter创建可调整大小的分割器
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(10)  # 设置分割条宽度
        main_splitter.setChildrenCollapsible(False)  # 防止子窗口被完全折叠
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #007BFF, stop:1 #0056B3);
                border: 2px solid #0056B3;
                border-radius: 5px;
                margin: 2px;
            }
            QSplitter::handle:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0056B3, stop:1 #004085);
                border: 2px solid #004085;
                box-shadow: 0 0 8px rgba(0, 123, 255, 0.5);
            }
            QSplitter::handle:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #004085, stop:1 #003366);
                border: 2px solid #003366;
            }
        """)

        # 左侧：图片检测区域 - 使用垂直分割器
        left_detection_widget = QWidget()
        left_detection_widget.setMinimumWidth(400)  # 设置最小宽度
        left_main_layout = QVBoxLayout()
        
        # 创建左侧垂直分割器
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setHandleWidth(8)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2196F3, stop:1 #1976D2);
                border: 2px solid #1976D2;
                border-radius: 4px;
                margin: 2px;
            }
            QSplitter::handle:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1976D2, stop:1 #1565C0);
                border: 2px solid #1565C0;
                box-shadow: 0 0 6px rgba(33, 150, 243, 0.5);
            }
            QSplitter::handle:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1565C0, stop:1 #0D47A1);
            }
        """)

        # 上半部分：标题和图片显示
        img_display_widget = QWidget()
        img_display_layout = QVBoxLayout()
        
        img_detection_title = QLabel("📸 图片识别功能")
        img_detection_title.setFont(font_title)
        img_detection_title.setAlignment(Qt.AlignCenter)
        img_detection_title.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E3F2FD, stop:1 #BBDEFB);
                border: 2px solid #2196F3;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                font-weight: bold;
            }
        """)

        # 图片显示区域 - 使用水平分割器让两张图片可以独立调整
        img_splitter = QSplitter(Qt.Horizontal)
        img_splitter.setHandleWidth(6)
        img_splitter.setChildrenCollapsible(False)
        img_splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF9800, stop:1 #F57C00);
                border: 2px solid #F57C00;
                border-radius: 3px;
                margin: 2px;
            }
            QSplitter::handle:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F57C00, stop:1 #EF6C00);
                border: 2px solid #EF6C00;
                box-shadow: 0 0 4px rgba(255, 152, 0, 0.5);
            }
            QSplitter::handle:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EF6C00, stop:1 #E65100);
            }
        """)
        
        # 左图片容器
        left_img_container = QWidget()
        left_img_container.setMinimumWidth(200)
        left_img_layout = QVBoxLayout()
        left_img_label = QLabel("📤 原始图片 (支持拖拽上传)")
        left_img_label.setAlignment(Qt.AlignCenter)
        left_img_label.setStyleSheet("""
            font-weight: bold; 
            color: #2C3E50; 
            margin: 5px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #E3F2FD, stop:1 #BBDEFB);
            border: 1px solid #2196F3;
            border-radius: 5px;
            padding: 5px;
        """)
        # 使用支持拖拽的图片标签
        self.left_img = DragDropImageLabel()
        self.left_img.fileDropped.connect(self.handle_dropped_file)  # 连接拖拽信号
        # 初始时显示默认图片
        if os.path.exists(IMAGE_LEFT_INIT):
            pixmap = QPixmap(IMAGE_LEFT_INIT)
            self.left_img.set_image(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.left_img.setScaledContents(True)  # 让图片自适应大小
        left_img_layout.addWidget(left_img_label)
        left_img_layout.addWidget(self.left_img)
        left_img_container.setLayout(left_img_layout)
        
        # 右图片容器
        right_img_container = QWidget()
        right_img_container.setMinimumWidth(200)
        right_img_layout = QVBoxLayout()
        right_img_label = QLabel("检测结果")
        right_img_label.setAlignment(Qt.AlignCenter)
        right_img_label.setStyleSheet("font-weight: bold; color: #2C3E50; margin: 5px;")
        self.right_img = QLabel()
        self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
        self.right_img.setAlignment(Qt.AlignCenter)
        self.right_img.setScaledContents(True)  # 让图片自适应大小
        right_img_layout.addWidget(right_img_label)
        right_img_layout.addWidget(self.right_img)
        right_img_container.setLayout(right_img_layout)
        
        img_splitter.addWidget(left_img_container)
        img_splitter.addWidget(right_img_container)
        img_splitter.setSizes([250, 250])  # 设置初始大小
        
        img_display_layout.addWidget(img_detection_title)
        img_display_layout.addWidget(img_splitter)
        img_display_widget.setLayout(img_display_layout)

        # 下半部分：检测结果和控制按钮
        img_control_widget = QWidget()
        img_control_layout = QVBoxLayout()
        
        self.img_num_label = QLabel("📊 当前检测结果：待检测")
        self.img_num_label.setFont(font_main)
        self.img_num_label.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFF3E0, stop:1 #FFE0B2);
                border: 2px solid #FF9800;
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                font-weight: bold;
                font-size: 14px;
                min-height: 60px;
                box-shadow: 0 4px 15px rgba(255, 152, 0, 0.2);
            }
        """)

        # 按钮区域
        button_container = QWidget()
        button_container_layout = QHBoxLayout()
        
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(lambda: self.upload_img_with_animation(up_img_button))
        det_img_button.clicked.connect(lambda: self.detect_img_with_animation(det_img_button))
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        
        # 添加工具提示和快捷键
        up_img_button.setToolTip("上传图片进行检测 (Ctrl+O)\n支持格式：JPG, PNG, JPEG, TIF, BMP, GIF")
        det_img_button.setToolTip("开始AI检测分析\n将识别图片中的农业害虫")
        
        # 为按钮添加图标
        up_img_button.setText("📁 上传图片")
        det_img_button.setText("🔍 开始检测")
        
        # 现代化按钮样式
        modern_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #007BFF, stop:1 #0056B3);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px 30px;
                font-weight: bold;
                font-size: 14px;
                margin: 8px;
                min-height: 20px;
                box-shadow: 0 4px 15px rgba(0, 123, 255, 0.3);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0056B3, stop:1 #004085);
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0, 123, 255, 0.4);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #004085, stop:1 #003366);
                transform: translateY(-1px);
                box-shadow: 0 2px 10px rgba(0, 123, 255, 0.2);
            }
        """

        up_img_button.setStyleSheet(modern_button_style)
        det_img_button.setStyleSheet(modern_button_style)
        
        button_container_layout.addWidget(up_img_button)
        button_container_layout.addWidget(det_img_button)
        button_container.setLayout(button_container_layout)

        img_control_layout.addWidget(self.img_num_label)
        img_control_layout.addWidget(button_container)
        img_control_widget.setLayout(img_control_layout)
        
        # 将图片显示和控制区域添加到左侧分割器
        left_splitter.addWidget(img_display_widget)
        left_splitter.addWidget(img_control_widget)
        left_splitter.setSizes([400, 150])  # 设置初始大小比例
        
        # 保存分割器引用
        self.left_splitter = left_splitter
        self.img_splitter = img_splitter
        
        left_main_layout.addWidget(left_splitter)
        left_detection_widget.setLayout(left_main_layout)

        # 右侧：AI聊天区域 - 使用垂直分割器进一步细分
        right_chat_widget = QWidget()
        right_chat_widget.setMinimumWidth(400)  # 设置最小宽度
        right_main_layout = QVBoxLayout()
        
        # 创建垂直分割器用于聊天区域的上下分割
        chat_splitter = QSplitter(Qt.Vertical)
        chat_splitter.setHandleWidth(8)
        chat_splitter.setChildrenCollapsible(False)
        chat_splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #28A745, stop:1 #1E7E34);
                border: 2px solid #1E7E34;
                border-radius: 4px;
                margin: 2px;
            }
            QSplitter::handle:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1E7E34, stop:1 #155724);
                border: 2px solid #155724;
                box-shadow: 0 0 6px rgba(40, 167, 69, 0.5);
            }
            QSplitter::handle:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #155724, stop:1 #0D4A1A);
            }
        """)

        # 上半部分：聊天显示和标题
        chat_display_widget = QWidget()
        chat_display_layout = QVBoxLayout()
        
        chat_title = QLabel("🤖 AI智能助手")
        chat_title.setFont(font_title)
        chat_title.setAlignment(Qt.AlignCenter)
        chat_title.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E8F5E8, stop:1 #C8E6C9);
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                font-weight: bold;
            }
        """)

        # 聊天显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(200)  # 减小最小高度，让分割器更灵活
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFFFF, stop:1 #F8F9FA);
                border: 2px solid #E9ECEF;
                border-radius: 12px;
                padding: 15px;
                font-size: 13px;
                line-height: 1.4;
                selection-background-color: #007BFF;
                color: black;
            }
            QTextEdit:focus {
                border: 2px solid #007BFF;
                box-shadow: 0 0 10px rgba(0, 123, 255, 0.3);
            }
        """)
        
        # 为聊天显示区域添加CSS动画支持
        self.chat_display.document().setDefaultStyleSheet("""
            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0; }
            }
        """)
        
        chat_display_layout.addWidget(chat_title)
        chat_display_layout.addWidget(self.chat_display)
        chat_display_widget.setLayout(chat_display_layout)

        # 下半部分：输入区域和控制按钮
        chat_control_widget = QWidget()
        chat_control_layout = QVBoxLayout()
        
        # 输入区域
        input_widget = QWidget()
        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("请输入您的问题...")
        self.chat_input.returnPressed.connect(self.send_message)
        self.chat_input.setStyleSheet("""
            QLineEdit {
                background: white;
                border: 2px solid #E9ECEF;
                border-radius: 20px;
                padding: 12px 20px;
                font-size: 14px;
                margin: 5px;
                color: black;
            }
            QLineEdit:focus {
                border: 2px solid #007BFF;
                background: #F8F9FF;
            }
            QLineEdit:hover {
                border: 2px solid #007BFF;
            }
        """)

        send_button = QPushButton("发送")
        send_button.clicked.connect(lambda: self.send_message_with_animation(send_button))
        send_button.setFont(font_main)
        send_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #28A745, stop:1 #1E7E34);
                color: white;
                border: none;
                border-radius: 20px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
                margin: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1E7E34, stop:1 #155724);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #155724, stop:1 #0D4A1A);
            }
        """)
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_button)
        input_widget.setLayout(input_layout)

        # 按钮区域 - 使用水平布局让按钮并排显示
        button_widget = QWidget()
        button_layout = QHBoxLayout()
        
        # 查看检测结果按钮
        view_results_button = QPushButton("查看检测结果")
        view_results_button.clicked.connect(self.show_detection_results)
        view_results_button.setFont(font_main)
        view_results_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #17A2B8, stop:1 #138496);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                margin: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #138496, stop:1 #0F6674);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0F6674, stop:1 #0B4A52);
            }
        """)

        # 清空聊天记录按钮
        clear_button = QPushButton("清空聊天")
        clear_button.clicked.connect(self.clear_chat)
        clear_button.setFont(font_main)
        clear_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #DC3545, stop:1 #C82333);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
                margin: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #C82333, stop:1 #A71E2A);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #A71E2A, stop:1 #8B1A1F);
            }
        """)
        
        # 布局控制按钮
        reset_layout_button = QPushButton("重置布局")
        reset_layout_button.clicked.connect(self.reset_layout)
        reset_layout_button.setFont(font_main)
        
        save_layout_button = QPushButton("保存布局")
        save_layout_button.clicked.connect(self.save_layout)
        save_layout_button.setFont(font_main)
        
        load_layout_button = QPushButton("恢复布局")
        load_layout_button.clicked.connect(self.load_layout)
        load_layout_button.setFont(font_main)
        
        # 主题切换按钮
        self.theme_button = QPushButton("🌙 深色模式")
        self.theme_button.clicked.connect(self.toggle_theme)
        self.theme_button.setFont(font_main)
        
        layout_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6F42C1, stop:1 #5A2D91);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: 11px;
                margin: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5A2D91, stop:1 #4C1F78);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4C1F78, stop:1 #3E1A5F);
            }
        """
        
        reset_layout_button.setStyleSheet(layout_button_style)
        save_layout_button.setStyleSheet(layout_button_style)
        load_layout_button.setStyleSheet(layout_button_style)

        # 第一行按钮
        button_layout.addWidget(view_results_button)
        button_layout.addWidget(clear_button)
        
        # 创建第二行布局按钮
        layout_button_widget = QWidget()
        layout_button_layout = QHBoxLayout()
        layout_button_layout.addWidget(reset_layout_button)
        layout_button_layout.addWidget(save_layout_button)
        layout_button_layout.addWidget(load_layout_button)
        layout_button_widget.setLayout(layout_button_layout)
        
        # 创建第三行主题按钮
        theme_button_widget = QWidget()
        theme_button_layout = QHBoxLayout()
        theme_button_layout.addWidget(self.theme_button)
        theme_button_widget.setLayout(theme_button_layout)
        button_widget.setLayout(button_layout)

        chat_control_layout.addWidget(input_widget)
        chat_control_layout.addWidget(button_widget)
        chat_control_layout.addWidget(layout_button_widget)
        chat_control_layout.addWidget(theme_button_widget)
        chat_control_widget.setLayout(chat_control_layout)
        
        # 将聊天显示和控制区域添加到垂直分割器
        chat_splitter.addWidget(chat_display_widget)
        chat_splitter.addWidget(chat_control_widget)
        chat_splitter.setSizes([300, 150])  # 设置初始大小比例
        
        # 将分割器添加到右侧主布局
        right_main_layout.addWidget(chat_splitter)
        right_chat_widget.setLayout(right_main_layout)

        # 将左右两个区域添加到主分割器
        main_splitter.addWidget(left_detection_widget)
        main_splitter.addWidget(right_chat_widget)
        main_splitter.setSizes([600, 600])  # 设置初始大小比例
        
        # 保存分割器引用以便重置布局
        self.main_splitter = main_splitter
        self.chat_splitter = chat_splitter
        
        # 尝试加载保存的布局设置
        self.load_layout_on_startup()
        
        # 加载主题偏好
        self.load_theme_preference()
        
        # 添加悬停效果
        QTimer.singleShot(100, self.add_hover_effects)
        
        # 启用整个窗口的拖拽功能
        self.enable_drag_drop_for_window()
        
        # 将主分割器添加到检测界面布局
        img_detection_layout = QVBoxLayout()
        img_detection_layout.addWidget(main_splitter)
        img_detection_widget.setLayout(img_detection_layout)

        # ********************* 模型切换界面 *****************************
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel("🌟 " + WELCOME_SENTENCE)
        about_title.setFont(QFont('Microsoft YaHei UI', 18, QFont.Bold))
        about_title.setAlignment(Qt.AlignCenter)
        about_title.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F3E5F5, stop:1 #E1BEE7);
                border: 3px solid #9C27B0;
                border-radius: 15px;
                padding: 15px;
                margin: 10px;
                font-weight: bold;
            }
        """)
        about_img = QLabel()
        about_img.setPixmap(QPixmap(ZHU_IMAGE_PATH))
        self.model_label = QLabel("🔧 当前模型：{}".format(self.model_path))
        self.model_label.setFont(font_main)
        self.model_label.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E0F2F1, stop:1 #B2DFDB);
                border: 2px solid #009688;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
                font-weight: bold;
            }
        """)
        change_model_button = QPushButton("切换模型")
        change_model_button.setFont(font_main)
        change_model_button.setStyleSheet(modern_button_style)

        record_button = QPushButton("查看历史记录")
        record_button.clicked.connect(self.check_record)
        record_button.setFont(font_main)
        record_button.setStyleSheet(modern_button_style)
        change_model_button.clicked.connect(self.change_model)
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("                      ")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addWidget(self.model_label)
        about_layout.addStretch()
        about_layout.addWidget(change_model_button)
        about_layout.addWidget(record_button)
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        self.left_img.setAlignment(Qt.AlignCenter)

        # ********************* 配置切换界面 ****************************
        config_widget = QWidget()

        config_grid_widget = QWidget()
        config_grid_layout = QGridLayout()

        # 1. 先定义 config_save_txt_value（后续要被其他组件复制样式）
        config_save_txt_label = QLabel("📄 推理时是否保存txt文件")
        # 标签样式将通过主题管理器动态设置
        self.config_save_txt_value = QRadioButton("True")
        self.config_save_txt_value.setChecked(False)
        self.config_save_txt_value.setAutoExclusive(False)
        # 单选按钮样式将通过主题管理器动态设置
        radio_button_style = """
            QRadioButton {
                font-size: 12px;
                color: #2C3E50;
                font-weight: bold;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid #007BFF;
                background: white;
            }
            QRadioButton::indicator:checked {
                background: #007BFF;
            }
        """
        self.config_save_txt_value.setStyleSheet(radio_button_style)
        config_grid_layout.addWidget(config_save_txt_label, 0, 0)  # 注意：调整网格布局的行号（避免重叠）
        config_grid_layout.addWidget(self.config_save_txt_value, 0, 1)

        # 2. 再定义 config_tta_value（此时可以安全复制 config_save_txt_value 的样式）
        config_tta_label = QLabel("🧪 启用测试时增强(TTA)")
        # 标签样式将通过主题管理器动态设置
        self.config_tta_value = QRadioButton("True")
        self.config_tta_value.setChecked(self.use_tta)
        self.config_tta_value.setAutoExclusive(False)
        # 现在 self.config_save_txt_value 已存在，可以正常复制样式
        self.config_tta_value.setStyleSheet(self.config_save_txt_value.styleSheet())
        config_grid_layout.addWidget(config_tta_label, 1, 0)  # 行号+1，避免与上一个组件重叠
        config_grid_layout.addWidget(self.config_tta_value, 1, 1)

        # 3. 其他原有组件（按原顺序保留，注意调整网格行号避免重叠）
        # 系统图像显示大小
        config_output_size_label = QLabel("🖼️ 系统图像显示大小")
        # 不设置固定颜色，让主题管理器控制
        self.config_output_size_value = QLineEdit("")
        self.config_output_size_value.setText(str(self.output_size))
        # 使用主题管理器的输入框样式，确保在任何主题下都可见
        config_input_style = """
            QLineEdit {
                background: white;
                border: 2px solid #E9ECEF;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
                color: black !important;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 2px solid #007BFF;
                background: #F8F9FF;
                color: black !important;
            }
            QLineEdit:hover {
                border: 2px solid #007BFF;
                color: black !important;
            }
        """
        self.config_output_size_value.setStyleSheet(config_input_style)
        config_grid_layout.addWidget(config_output_size_label, 2, 0)  # 行号继续递增
        config_grid_layout.addWidget(self.config_output_size_value, 2, 1)

        # 推理图像尺寸
        config_imgsz_label = QLabel("🧩 推理图像尺寸(imgsz)")
        # 不设置固定颜色，让主题管理器控制
        self.config_imgsz_value = QLineEdit("")
        self.config_imgsz_value.setText(str(self.imgsz))
        self.config_imgsz_value.setStyleSheet(config_input_style)
        config_grid_layout.addWidget(config_imgsz_label, 3, 0)
        config_grid_layout.addWidget(self.config_imgsz_value, 3, 1)

        # 检测模型置信度阈值
        config_conf_thres_label = QLabel("🎯 检测模型置信度阈值")
        # 不设置固定颜色，让主题管理器控制
        self.config_conf_thres_value = QLineEdit("")
        self.config_conf_thres_value.setText(str(self.conf_thres))
        self.config_conf_thres_value.setStyleSheet(config_input_style)
        config_grid_layout.addWidget(config_conf_thres_label, 4, 0)
        config_grid_layout.addWidget(self.config_conf_thres_value, 4, 1)

        # 检测模型IOU阈值
        config_iou_thres_label = QLabel("📏 检测模型IOU阈值")
        # 不设置固定颜色，让主题管理器控制
        self.config_iou_thres_value = QLineEdit("")
        self.config_iou_thres_value.setText(str(self.iou_thres))
        self.config_iou_thres_value.setStyleSheet(config_input_style)
        config_grid_layout.addWidget(config_iou_thres_label, 5, 0)
        config_grid_layout.addWidget(self.config_iou_thres_value, 5, 1)

        # 推理时是否保存置信度（原有组件，行号继续递增）
        config_save_conf_label = QLabel("📊 推理时是否保存置信度")
        # 不设置固定颜色，让主题管理器控制
        self.config_save_conf_value = QRadioButton("True")
        self.config_save_conf_value.setChecked(False)
        self.config_save_conf_value.setAutoExclusive(False)
        self.config_save_conf_value.setStyleSheet(self.config_save_txt_value.styleSheet())
        config_grid_layout.addWidget(config_save_conf_label, 6, 0)
        config_grid_layout.addWidget(self.config_save_conf_value, 6, 1)

        # 推理时是否保存切片文件（原有组件）
        config_save_crop_label = QLabel("✂️ 推理时是否保存切片文件")
        # 不设置固定颜色，让主题管理器控制
        self.config_save_crop_value = QRadioButton("True")
        self.config_save_crop_value.setChecked(False)
        self.config_save_crop_value.setAutoExclusive(False)
        self.config_save_crop_value.setStyleSheet(self.config_save_txt_value.styleSheet())
        config_grid_layout.addWidget(config_save_crop_label, 7, 0)
        config_grid_layout.addWidget(self.config_save_crop_value, 7, 1)

        # 后续原有代码（无需修改）
        config_grid_widget.setLayout(config_grid_layout)
        config_grid_widget.setFont(font_main)

        save_config_button = QPushButton("保存配置信息")
        # ... （保存按钮及其他布局代码保持不变）
        save_config_button.setFont(font_main)
        save_config_button.clicked.connect(self.save_config_change)
        save_config_button.setStyleSheet(modern_button_style)
        config_layout = QVBoxLayout()
        config_vid_title = QLabel("⚙️ 配置信息修改")
        config_icon_label = QLabel()
        config_icon_label.setPixmap(QPixmap("images/UI/config.png"))
        config_icon_label.setAlignment(Qt.AlignCenter)
        config_vid_title.setAlignment(Qt.AlignCenter)
        config_vid_title.setFont(font_title)
        config_vid_title.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFF8E1, stop:1 #FFECB3);
                border: 2px solid #FFC107;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
                font-weight: bold;
            }
        """)
        config_layout.addWidget(config_vid_title)
        config_layout.addWidget(config_icon_label)
        config_layout.addWidget(config_grid_widget)
        config_layout.addStretch()
        config_layout.addWidget(save_config_button)
        config_widget.setLayout(config_layout)

        self.addTab(about_widget, '主页')
        self.addTab(img_detection_widget, '图片检测+AI助手')
        self.addTab(config_widget, '配置信息')
        self.setTabIcon(0, QIcon(ICON_IMAGE))
        self.setTabIcon(1, QIcon(ICON_IMAGE))
        self.setTabIcon(2, QIcon(ICON_IMAGE))
        
        # 添加状态栏提示
        self.create_status_bar()

        # ********************* todo 布局修改和颜色变换等相关插件 *****************************

    def show_message(self, icon, title, text, buttons=QMessageBox.Ok):
        """统一的消息框，根据当前主题设置颜色。"""
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        
        # 根据当前主题设置消息框样式
        theme = self.theme_manager.get_theme()
        msg_style = f"""
            QMessageBox {{
                background: {theme["secondary_bg"]};
                color: {theme["text_color"]};
                border: 2px solid {theme["border_color"]};
                border-radius: 10px;
            }}
            QLabel {{ 
                color: {theme["text_color"]}; 
                background: transparent;
                font-size: 14px;
                padding: 10px;
            }}
            QPushButton {{ 
                background: {theme["accent_color"]};
                color: white;
                border: 2px solid {theme["accent_color"]};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background: {theme["hover_color"]};
                border: 2px solid {theme["hover_color"]};
            }}
            QPushButton:pressed {{
                background: {theme["border_color"]};
                border: 2px solid {theme["border_color"]};
            }}
        """
        msg.setStyleSheet(msg_style)
        return msg.exec()

    def _simplify_ai_text(self, text: str) -> str:
        """将大模型回复从Markdown/长段文本压缩为简洁要点。
        规则：去粗体/标题/表格/分隔线，将列表前缀统一为"• ", 合并多余空行，并限制行数。
        """
        if not text:
            return ""
        s = text
        # 统一换行
        s = s.replace('\r\n', '\n').replace('\r', '\n')
        # 去除表格与分隔线
        s = re.sub(r"^\s*\|.*\|\s*$", "", s, flags=re.MULTILINE)
        s = re.sub(r"^-{3,}\s*$", "", s, flags=re.MULTILINE)
        s = re.sub(r"^—+\s*$", "", s, flags=re.MULTILINE)
        # 去除标题符号和多余标点
        s = re.sub(r"^\s*#{1,6}\s*", "", s, flags=re.MULTILINE)
        # 去除粗体/斜体/行内代码
        s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
        s = re.sub(r"\*(.*?)\*", r"\1", s)
        s = re.sub(r"`([^`]*)`", r"\1", s)
        # 统一列表符号
        s = re.sub(r"^\s*[-*+]\s+", "• ", s, flags=re.MULTILINE)
        s = re.sub(r"^\s*\d+\.[)\s]+", "• ", s, flags=re.MULTILINE)
        # 合并空行
        s = re.sub(r"\n{3,}", "\n\n", s)
        # 去掉首尾空白
        s = s.strip()
        # 分段展示：按空行保留全部段落
        lines = [ln.rstrip() for ln in s.split('\n')]
        return "\n".join(lines)

    def _append_preserved(self, text: str, with_prefix: Optional[str] = None):
        """将文本按原换行渲染到 QTextEdit 中，避免换行被吞。"""
        safe = html.escape(text or "")
        body = safe.replace("\n", "<br>")
        if with_prefix:
            prefix = html.escape(with_prefix)
            html_block = f"<div style='white-space: normal;'><b>{prefix}:</b><br>{body}</div>"
        else:
            html_block = f"<div style='white-space: normal;'>{body}</div>"
        self.chat_display.append(html_block)

    def upload_img(self):
        """通过文件对话框上传图像"""
        fileName, fileType = QFileDialog.getOpenFileName(
            self, 
            '选择图片文件', 
            '', 
            '图片文件 (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.gif);;所有文件 (*.*)'
        )
        
        if fileName:
            # 使用统一的拖拽处理方法
            self.handle_dropped_file(fileName)

    def change_model(self):
        """切换模型，重新对self.model进行赋值"""
        # 用于pt格式模型的结果，这个模型必须是经过这里的代码训练出来的
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.pt')
        if fileName:
            # 如果用户选择了对应的pt文件，根据用户选择的pt文件重新对模型进行初始化
            self.model_path = fileName
            self.model = self.model_load(weights=self.model_path)
            self.show_message(QMessageBox.Information, "成功", "模型切换成功！")
            self.model_label.setText("当前模型：{}".format(self.model_path))

    # 图片检测
    def detect_img(self):
        """检测单张的图像文件"""
        # 1) 基础校验：是否已选择图片
        if not self.img2predict or not osp.exists(self.img2predict):
            self.show_message(QMessageBox.Warning, "提示", "请先上传一张图片再进行检测。")
            return

        # 2) 检测开始前的UI动画效果
        self.img_num_label.setText("🔍 正在检测中...")
        self.show_success_animation(self.img_num_label)  # 添加脉冲动画
        
        # 3) 确保输出目录存在（首次运行可能未创建，避免写文件失败）
        os.makedirs("images/tmp", exist_ok=True)
        os.makedirs("record/img", exist_ok=True)
        os.makedirs(PREDICT_PROJECT_DIR, exist_ok=True)

        output_size = self.output_size

        try:
            results = self.model(
                self.img2predict,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                augment=self.use_tta,
                save_txt=self.save_txt,
                save_conf=self.save_conf,
                save_crop=self.save_crop,
                project=PREDICT_PROJECT_DIR,
                name=PREDICT_RUN_NAME,
                exist_ok=True,
                save=True,
            )  # 读取图像并执行检测的逻辑
        except Exception as e:
            msg = str(e)
            # 针对 "Plain typing.Self is not valid as type argument" 的一次性自动修复重试
            if "Self is not valid as type argument" in msg or "typing.Self" in msg:
                try:
                    # 重新加载模型并重试一次
                    self.model = self.model_load(weights=self.model_path)
                    results = self.model(
                        self.img2predict,
                        conf=self.conf_thres,
                        iou=self.iou_thres,
                        imgsz=self.imgsz,
                        augment=self.use_tta,
                        save_txt=self.save_txt,
                        save_conf=self.save_conf,
                        save_crop=self.save_crop,
                        project=PREDICT_PROJECT_DIR,
                        name=PREDICT_RUN_NAME,
                        exist_ok=True,
                        save=True,
                    )
                except Exception as e2:
                    self.show_message(QMessageBox.Critical, "检测失败", f"模型推理出错：{e2}")
                    return
            else:
                self.show_message(QMessageBox.Critical, "检测失败", f"模型推理出错：{e}")
                return
        # 如果你想要对结果进行单独的解析请使用下面的内容
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        # 显示并保存检测的结果
        result = results[0]  # 获取检测结果
        img_array = result.plot()  # 在图像上绘制检测结果
        im0 = img_array
        im_record = copy.deepcopy(im0)
        resize_scale = output_size / im0.shape[0]
        im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
        time_re = str(time.strftime('result_%Y-%m-%d_%H-%M-%S_%A'))
        cv2.imwrite("record/img/{}.jpg".format(time_re), im_record)
        # 显示每个类别中检测出来的样本数量
        result_names = result.names
        result_nums = [0 for i in range(0, len(result_names))]
        cls_ids = list(result.boxes.cls.cpu().numpy())
        for cls_id in cls_ids:
            result_nums[int(cls_id)] = result_nums[int(cls_id)] + 1
        result_info = ""
        for idx_cls, cls_num in enumerate(result_nums):
            # 添加对数据0的判断，如果当前数据的数目为0，则这个数据不需要加入到里面
            if cls_num > 0:
                result_info = result_info + "{}:{}\n".format(result_names[idx_cls], cls_num)
        # 更新检测结果显示（带动画效果）
        result_text = "📊 当前检测结果\n{}".format(result_info) if result_info else "📊 当前检测结果\n未检测到目标"
        self.img_num_label.setText(result_text)

        # 存储检测结果到ChatBot中，只存储检测到的类别
        self.latest_detection_results = {result_names[idx_cls]: cls_num for idx_cls, cls_num in enumerate(result_nums)
                                         if cls_num > 0}
        
        # 调试信息：打印检测结果
        print("=== 检测结果调试信息 ===")
        print(f"result_names: {result_names}")
        print(f"result_nums: {result_nums}")
        print(f"latest_detection_results: {self.latest_detection_results}")
        print("========================")
        
        self.chatbot.set_detection_results(self.latest_detection_results)

        # 检测完成后的动画效果
        self.animate_detection_result()
        self.show_success_animation(self.img_num_label)
        
        # 为右侧结果图片添加淡入动画
        self.animation_manager.fade_in_widget(self.right_img, duration=600)
        
        # 显示浮动通知（显示3秒后渐变消失）
        self.create_floating_notification("✅ 检测成功！日志已保存", duration=3000, notification_type="success")
        # self.show_message(QMessageBox.Information, "检测成功", "日志已保存！")

    def check_record(self):
        """打开历史记录文件夹"""
        os.startfile(osp.join(os.path.abspath(os.path.dirname(__file__)), "record"))

    def send_message(self):
        """发送消息到AI助手"""
        user_input = self.chat_input.text().strip()
        if not user_input:
            return

        # 停止之前的打字机效果（如果有的话）
        self.stop_typewriter_effect()

        # 显示用户消息（带动画）
        self.chat_display.append(f"<b>您:</b> {user_input}")
        self.chat_input.clear()
        self.animate_chat_message()  # 滚动动画

        # 显示等待提示（带打字机效果）
        self.chat_display.append("<i>🤔 AI正在思考中...</i>")
        QApplication.processEvents()  # 立即更新界面

        try:
            # 检查是否有检测结果，如果有则显示提示
            if self.latest_detection_results:
                print(f"=== AI聊天显示检测结果 ===")
                print(f"latest_detection_results: {self.latest_detection_results}")
                print(f"结果数量: {len(self.latest_detection_results)}")
                
                detection_summary = "📊 当前检测结果："
                for class_name, count in self.latest_detection_results.items():
                    detection_summary += f"\n   • {class_name}: {count}个"
                    print(f"显示: {class_name}: {count}个")
                print("===========================")
                
                self._append_preserved(detection_summary)

            # 调用AI API
            response = self.chatbot.chat(user_input, include_detection=True)

            # 移除等待提示并显示AI回复（优先解析JSON结构化输出）
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # 删除换行符

            displayed = False
            try:
                data = json.loads(response)
                if isinstance(data, dict) and isinstance(data.get("categories"), list):
                    parts = []
                    for item in data["categories"]:
                        name = str(item.get("name", "未知类别"))
                        count = item.get("count")
                        header = f"【类别：{name}{'' if count is None else f'（{count}个）'}】"
                        def bullets(key):
                            vals = item.get(key) or []
                            return "\n".join([f"• {v}" for v in vals]) if vals else "• （暂无要点）"
                        section = (
                            f"{header}\n"
                            f"物理防治\n{bullets('physical')}\n"
                            f"生物防治\n{bullets('biological')}\n"
                            f"化学防治\n{bullets('chemical')}\n"
                            f"其他建议\n{bullets('other')}"
                        )
                        parts.append(section)
                    # 使用打字机效果显示结构化回复
                    self.start_typewriter_effect("\n\n".join(parts), with_prefix="AI")
                    displayed = True
            except Exception:
                pass

            if not displayed:
                simplified = self._simplify_ai_text(response)
                # 使用打字机效果显示AI回复
                self.start_typewriter_effect(simplified, with_prefix="AI")

        except Exception as e:
            # 移除等待提示并显示错误信息
            cursor = self.chat_display.textCursor()
            cursor.movePosition(cursor.End)
            cursor.movePosition(cursor.StartOfLine, cursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # 删除换行符

            self.chat_display.append(f"<b>错误:</b> {str(e)}")

        # 滚动到底部（带动画）
        self.animate_chat_message()

    def show_detection_results(self):
        """显示当前检测结果"""
        if not self.latest_detection_results:
            self.show_message(QMessageBox.Information, "检测结果", "目前还没有进行图片检测，请先上传图片并进行检测。")
            return

        result_text = "📊 当前检测结果：\n\n"
        for class_name, count in self.latest_detection_results.items():
            result_text += f"• {class_name}: {count}个\n"

        result_text += "\n💡 提示：您可以询问AI关于这些病虫害的治理方法！"

        self.show_message(QMessageBox.Information, "检测结果详情", result_text)

    def clear_chat(self):
        """清空聊天记录"""
        # 停止打字机效果
        self.stop_typewriter_effect()
        
        self.chat_display.clear()
        self.chatbot = ChatBot()  # 重新初始化聊天机器人

    def reset_layout(self):
        """重置界面布局到默认状态"""
        # 重置主分割器的大小（左右分割）
        self.main_splitter.setSizes([600, 600])
        # 重置左侧分割器的大小（图片显示和控制区域）
        self.left_splitter.setSizes([400, 150])
        # 重置图片分割器的大小（左右图片）
        self.img_splitter.setSizes([250, 250])
        # 重置聊天区域分割器的大小（聊天显示和输入区域）
        self.chat_splitter.setSizes([300, 150])
        self.create_floating_notification("🔄 布局已重置为默认状态", notification_type="info")

    def save_layout(self):
        """保存当前布局设置"""
        try:
            layout_config = {
                'main_splitter': self.main_splitter.sizes(),
                'left_splitter': self.left_splitter.sizes(),
                'img_splitter': self.img_splitter.sizes(),
                'chat_splitter': self.chat_splitter.sizes(),
                'window_size': [self.width(), self.height()]
            }
            
            # 确保配置目录存在
            config_dir = "config"
            os.makedirs(config_dir, exist_ok=True)
            
            # 保存到JSON文件
            config_path = os.path.join(config_dir, "layout_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(layout_config, f, ensure_ascii=False, indent=2)
            
            self.create_floating_notification("💾 布局设置已保存", notification_type="success")
        except Exception as e:
            self.show_message(QMessageBox.Warning, "保存失败", f"保存布局设置失败：{str(e)}")

    def load_layout(self):
        """加载保存的布局设置"""
        try:
            config_path = os.path.join("config", "layout_config.json")
            if not os.path.exists(config_path):
                self.show_message(QMessageBox.Information, "提示", "未找到保存的布局设置，请先保存布局！")
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                layout_config = json.load(f)
            
            # 恢复分割器大小
            if 'main_splitter' in layout_config:
                self.main_splitter.setSizes(layout_config['main_splitter'])
            if 'left_splitter' in layout_config:
                self.left_splitter.setSizes(layout_config['left_splitter'])
            if 'img_splitter' in layout_config:
                self.img_splitter.setSizes(layout_config['img_splitter'])
            if 'chat_splitter' in layout_config:
                self.chat_splitter.setSizes(layout_config['chat_splitter'])
            
            # 恢复窗口大小
            if 'window_size' in layout_config:
                width, height = layout_config['window_size']
                self.resize(width, height)
            
            self.create_floating_notification("📂 布局设置已恢复", notification_type="success")
        except Exception as e:
            self.show_message(QMessageBox.Warning, "加载失败", f"加载布局设置失败：{str(e)}")

    def toggle_fullscreen_mode(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def auto_adjust_layout(self):
        """根据窗口大小自动调整布局"""
        window_width = self.width()
        window_height = self.height()
        
        # 根据窗口宽度调整左右分割比例
        if window_width < 1000:
            # 小窗口：左侧占更多空间
            self.main_splitter.setSizes([int(window_width * 0.6), int(window_width * 0.4)])
        else:
            # 大窗口：平均分配
            self.main_splitter.setSizes([int(window_width * 0.5), int(window_width * 0.5)])
        
        # 根据窗口高度调整上下分割比例
        if window_height < 700:
            # 小窗口：压缩控制区域
            self.left_splitter.setSizes([int(window_height * 0.7), int(window_height * 0.3)])
            self.chat_splitter.setSizes([int(window_height * 0.7), int(window_height * 0.3)])
        else:
            # 大窗口：给控制区域更多空间
            self.left_splitter.setSizes([int(window_height * 0.6), int(window_height * 0.4)])
            self.chat_splitter.setSizes([int(window_height * 0.6), int(window_height * 0.4)])

    def load_layout_on_startup(self):
        """启动时静默加载布局设置"""
        try:
            config_path = os.path.join("config", "layout_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    layout_config = json.load(f)
                
                # 恢复分割器大小
                if 'main_splitter' in layout_config:
                    self.main_splitter.setSizes(layout_config['main_splitter'])
                if 'left_splitter' in layout_config:
                    self.left_splitter.setSizes(layout_config['left_splitter'])
                if 'img_splitter' in layout_config:
                    self.img_splitter.setSizes(layout_config['img_splitter'])
                if 'chat_splitter' in layout_config:
                    self.chat_splitter.setSizes(layout_config['chat_splitter'])
                
                # 恢复窗口大小
                if 'window_size' in layout_config:
                    width, height = layout_config['window_size']
                    self.resize(width, height)
        except Exception:
            # 静默失败，使用默认布局
            pass

    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 可以在这里添加自动调整逻辑
        # self.auto_adjust_layout()  # 取消注释以启用自动调整
        
        # 更新拖拽覆盖层大小
        if hasattr(self, 'drag_overlay') and self.drag_overlay.isVisible():
            self.drag_overlay.resize(self.size())

    def keyPressEvent(self, event):
        """键盘事件处理"""
        # F11 切换全屏
        if event.key() == Qt.Key_F11:
            self.toggle_fullscreen_mode()
        # Ctrl+R 重置布局
        elif event.key() == Qt.Key_R and event.modifiers() == Qt.ControlModifier:
            self.reset_layout()
        # Ctrl+S 保存布局
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.save_layout()
        # Ctrl+L 加载布局
        elif event.key() == Qt.Key_L and event.modifiers() == Qt.ControlModifier:
            self.load_layout()
        # Ctrl+O 打开文件
        elif event.key() == Qt.Key_O and event.modifiers() == Qt.ControlModifier:
            self.upload_img()
        else:
            super().keyPressEvent(event)

    def toggle_theme(self):
        """切换主题模式"""
        # 添加切换动画
        self.animation_manager.fade_out_widget(self, duration=200)
        
        # 延迟切换主题
        QTimer.singleShot(200, self._switch_theme_delayed)
    
    def _switch_theme_delayed(self):
        """延迟切换主题"""
        # 切换主题
        new_theme = self.theme_manager.switch_theme()
        
        # 更新按钮文本（带动画）
        if new_theme == "dark":
            self.theme_button.setText("☀️ 浅色模式")
        else:
            self.theme_button.setText("🌙 深色模式")
        
        # 应用新主题
        self.apply_theme()
        
        # 刷新配置界面的值显示
        self.refresh_config_values()
        
        # 淡入动画
        self.animation_manager.fade_in_widget(self, duration=300)
        
        # 为主要组件添加主题切换动画
        for widget in [self.left_img, self.right_img, self.img_num_label, self.chat_display]:
            if hasattr(self, widget.__class__.__name__.lower().replace('q', '')):
                self.animation_manager.fade_in_widget(widget, duration=400)
        
        # 保存主题设置
        self.save_theme_preference()
        
        # 显示主题切换通知
        theme_name = "深色模式" if new_theme == "dark" else "浅色模式"
        self.create_floating_notification(f"🎨 已切换到{theme_name}", notification_type="info")
    
    def save_theme_preference(self):
        """保存主题偏好设置"""
        try:
            config_dir = "config"
            os.makedirs(config_dir, exist_ok=True)
            
            theme_config = {
                "current_theme": self.theme_manager.current_theme
            }
            
            config_path = os.path.join(config_dir, "theme_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(theme_config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 静默失败
    
    def load_theme_preference(self):
        """加载主题偏好设置"""
        try:
            config_path = os.path.join("config", "theme_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    theme_config = json.load(f)
                
                saved_theme = theme_config.get("current_theme", "light")
                if saved_theme != self.theme_manager.current_theme:
                    self.theme_manager.current_theme = saved_theme
                    
                    # 更新按钮文本
                    if saved_theme == "dark":
                        self.theme_button.setText("☀️ 浅色模式")
                    else:
                        self.theme_button.setText("🌙 深色模式")
        except Exception:
            pass  # 静默失败
    
    def show_startup_animation(self):
        """显示启动动画"""
        # 为主窗口添加淡入效果
        self.animation_manager.fade_in_widget(self, duration=800)
        
        # 为标签页添加滑入效果
        QTimer.singleShot(300, lambda: self.animate_tabs())
    
    def animate_tabs(self):
        """为标签页添加动画效果"""
        for i in range(self.count()):
            widget = self.widget(i)
            if widget:
                # 延迟显示每个标签页
                QTimer.singleShot(i * 100, lambda w=widget: self.animation_manager.fade_in_widget(w, duration=400))
    
    def animate_button_click(self, button):
        """按钮点击动画"""
        # 添加弹跳效果
        self.animation_manager.bounce_widget(button, duration=300)
        
        # 添加阴影效果
        shadow_color = QColor(0, 123, 255, 100) if self.theme_manager.current_theme == "light" else QColor(255, 255, 255, 50)
        self.animation_manager.add_shadow_effect(button, shadow_color)
    
    def animate_detection_result(self):
        """检测结果动画"""
        # 为检测结果标签添加脉冲动画
        if hasattr(self, 'img_num_label'):
            self.animation_manager.pulse_widget(self.img_num_label, duration=800, repeat=2)
    
    def animate_chat_message(self):
        """聊天消息动画"""
        # 为聊天显示区域添加滑入动画
        if hasattr(self, 'chat_display'):
            # 滚动到底部的动画效果
            scrollbar = self.chat_display.verticalScrollBar()
            current_value = scrollbar.value()
            target_value = scrollbar.maximum()
            
            # 创建滚动动画
            scroll_animation = QPropertyAnimation(scrollbar, b"value")
            scroll_animation.setDuration(300)
            scroll_animation.setStartValue(current_value)
            scroll_animation.setEndValue(target_value)
            scroll_animation.setEasingCurve(QEasingCurve.OutCubic)
            scroll_animation.start()
            
            self.animation_manager.animations.append(scroll_animation)
    
    def show_success_animation(self, widget):
        """成功动画"""
        # 绿色脉冲效果
        original_style = widget.styleSheet()
        
        # 临时改变样式
        success_style = original_style + """
            border: 3px solid #28A745;
            background: rgba(40, 167, 69, 0.1);
        """
        widget.setStyleSheet(success_style)
        
        # 脉冲动画
        self.animation_manager.pulse_widget(widget, duration=600, repeat=1)
        
        # 恢复原样式
        QTimer.singleShot(600, lambda: widget.setStyleSheet(original_style))
    
    def show_error_animation(self, widget):
        """错误动画"""
        # 红色摇摆效果
        original_style = widget.styleSheet()
        
        # 临时改变样式
        error_style = original_style + """
            border: 3px solid #DC3545;
            background: rgba(220, 53, 69, 0.1);
        """
        widget.setStyleSheet(error_style)
        
        # 摇摆动画
        self.animation_manager.shake_widget(widget, duration=500)
        
        # 恢复原样式
        QTimer.singleShot(500, lambda: widget.setStyleSheet(original_style))

    def upload_img_with_animation(self, button):
        """带动画的上传图片"""
        self.animate_button_click(button)
        # 添加按钮文字变化动画
        original_text = button.text()
        button.setText("📂 选择中...")
        QTimer.singleShot(100, self.upload_img)
        QTimer.singleShot(500, lambda: button.setText(original_text))
    
    def detect_img_with_animation(self, button):
        """带动画的检测图片"""
        self.animate_button_click(button)
        # 添加按钮状态变化
        original_text = button.text()
        button.setText("🔄 检测中...")
        button.setEnabled(False)  # 防止重复点击
        
        # 延迟执行检测，让动画先播放
        QTimer.singleShot(100, self.detect_img)
        
        # 检测完成后恢复按钮状态
        QTimer.singleShot(2000, lambda: self._restore_detect_button(button, original_text))
    
    def send_message_with_animation(self, button):
        """带动画的发送消息"""
        self.animate_button_click(button)
        # 添加发送状态动画
        original_text = button.text()
        button.setText("📤 发送中...")
        button.setEnabled(False)
        QTimer.singleShot(100, self.send_message)
        QTimer.singleShot(1000, lambda: self._restore_send_button(button, original_text))
    
    def add_hover_effects(self):
        """为组件添加悬停效果"""
        # 为所有按钮添加悬停动画
        buttons = self.findChildren(QPushButton)
        for button in buttons:
            button.enterEvent = lambda event, btn=button: self.on_button_hover_enter(btn, event)
            button.leaveEvent = lambda event, btn=button: self.on_button_hover_leave(btn, event)
    
    def on_button_hover_enter(self, button, event):
        """按钮悬停进入"""
        # 添加轻微的放大效果
        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(150)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        
        current_rect = button.geometry()
        hover_rect = QRect(current_rect)
        hover_rect.adjust(-2, -2, 2, 2)  # 轻微放大
        
        animation.setStartValue(current_rect)
        animation.setEndValue(hover_rect)
        animation.start()
        
        # 保存动画引用
        button._hover_animation = animation
    
    def on_button_hover_leave(self, button, event):
        """按钮悬停离开"""
        # 恢复原始大小
        if hasattr(button, '_hover_animation'):
            animation = QPropertyAnimation(button, b"geometry")
            animation.setDuration(150)
            animation.setEasingCurve(QEasingCurve.OutCubic)
            
            current_rect = button.geometry()
            original_rect = QRect(current_rect)
            original_rect.adjust(2, 2, -2, -2)  # 恢复原始大小
            
            animation.setStartValue(current_rect)
            animation.setEndValue(original_rect)
            animation.start()
    
    def create_floating_notification(self, message, duration=3000, notification_type="info"):
        """创建浮动通知"""
        notification = QLabel(message)
        notification.setParent(self)
        
        theme_color = self.theme_manager.get_theme()[f"{notification_type}_color"]
        notification.setStyleSheet(f"""
            QLabel {{
                background: {theme_color};
                color: white;
                border-radius: 20px;
                padding: 15px 25px;
                font-weight: bold;
                font-size: 14px;
                border: none;
                min-width: 200px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }}
        """)
        notification.setAlignment(Qt.AlignCenter)
        notification.adjustSize()
        notification.setFixedSize(notification.width(), notification.height())
        
        # 定位到窗口顶部中央
        x = (self.width() - notification.width()) // 2
        y = 50
        notification.move(x, y)
        
        # 添加阴影效果
        self.animation_manager.add_shadow_effect(notification)
        
        # 初始设置为透明
        effect = QGraphicsOpacityEffect()
        notification.setGraphicsEffect(effect)
        effect.setOpacity(0.0)
        
        # 显示通知
        notification.show()
        notification.raise_()  # 确保在最上层
        
        # 淡入动画
        fade_in_animation = QPropertyAnimation(effect, b"opacity")
        fade_in_animation.setDuration(500)
        fade_in_animation.setStartValue(0.0)
        fade_in_animation.setEndValue(1.0)
        fade_in_animation.setEasingCurve(QEasingCurve.OutCubic)
        fade_in_animation.start()
        
        # 保存动画引用
        notification._fade_in_animation = fade_in_animation
        notification._opacity_effect = effect
        
        # 延迟后开始淡出动画（duration - 1000ms用于淡出）
        fade_out_delay = max(duration - 1000, 1000)  # 确保至少显示1秒
        QTimer.singleShot(fade_out_delay, lambda: self.hide_notification(notification))
    
    def hide_notification(self, notification):
        """隐藏通知"""
        if not notification or not notification.isVisible():
            return
            
        # 获取透明度效果对象
        effect = getattr(notification, '_opacity_effect', None)
        if not effect:
            # 如果没有透明度效果，创建一个
            effect = QGraphicsOpacityEffect()
            notification.setGraphicsEffect(effect)
            effect.setOpacity(1.0)
        
        # 创建淡出动画
        fade_out_animation = QPropertyAnimation(effect, b"opacity")
        fade_out_animation.setDuration(1000)  # 1秒的淡出时间
        fade_out_animation.setStartValue(effect.opacity())
        fade_out_animation.setEndValue(0.0)
        fade_out_animation.setEasingCurve(QEasingCurve.InCubic)
        
        # 动画完成后删除通知
        fade_out_animation.finished.connect(notification.deleteLater)
        
        # 开始淡出动画
        fade_out_animation.start()
        
        # 保存动画引用防止被垃圾回收
        notification._fade_out_animation = fade_out_animation
    
    def show_progress_overlay(self, title="处理中...", subtitle="请稍候"):
        """显示现代化进度覆盖层"""
        # 创建覆盖层
        self.progress_overlay = QWidget(self)
        self.progress_overlay.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 0.7);
                border-radius: 0px;
            }
        """)
        
        # 创建进度内容容器
        progress_container = QWidget()
        progress_container.setFixedSize(350, 200)
        theme = self.theme_manager.get_theme()
        progress_container.setStyleSheet(f"""
            QWidget {{
                background: {theme["secondary_bg"]};
                border: 2px solid {theme["accent_color"]};
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }}
        """)
        
        # 布局
        container_layout = QVBoxLayout()
        container_layout.setAlignment(Qt.AlignCenter)
        container_layout.setSpacing(20)
        
        # 标题
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {theme["text_color"]};
                font-size: 18px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 10px;
            }}
        """)
        
        # 副标题
        subtitle_label = QLabel(subtitle)
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: {theme["text_color"]};
                font-size: 14px;
                background: transparent;
                border: none;
                opacity: 0.8;
            }}
        """)
        
        # 创建现代化进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 无限进度条
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {theme["border_color"]};
                border-radius: 10px;
                background: {theme["primary_bg"]};
                text-align: center;
                font-weight: bold;
                color: {theme["text_color"]};
                height: 20px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {theme["accent_color"]}, stop:1 {theme["hover_color"]});
                border-radius: 8px;
                margin: 2px;
            }}
        """)
        
        # 添加旋转动画图标
        spinner_label = QLabel("⟳")
        spinner_label.setAlignment(Qt.AlignCenter)
        spinner_label.setStyleSheet(f"""
            QLabel {{
                color: {theme["accent_color"]};
                font-size: 24px;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
        """)
        
        # 创建旋转动画
        self.spinner_animation = QPropertyAnimation(spinner_label, b"rotation")
        self.spinner_animation.setDuration(1000)
        self.spinner_animation.setStartValue(0)
        self.spinner_animation.setEndValue(360)
        self.spinner_animation.setLoopCount(-1)  # 无限循环
        
        container_layout.addWidget(spinner_label)
        container_layout.addWidget(title_label)
        container_layout.addWidget(subtitle_label)
        container_layout.addWidget(self.progress_bar)
        progress_container.setLayout(container_layout)
        
        # 覆盖层布局
        overlay_layout = QVBoxLayout()
        overlay_layout.setAlignment(Qt.AlignCenter)
        overlay_layout.addWidget(progress_container)
        self.progress_overlay.setLayout(overlay_layout)
        
        # 设置覆盖层大小和位置
        self.progress_overlay.resize(self.size())
        self.progress_overlay.show()
        self.progress_overlay.raise_()
        
        # 开始动画
        self.spinner_animation.start()
        
        # 淡入效果
        self.animation_manager.fade_in_widget(self.progress_overlay, duration=300)
    
    def hide_progress_overlay(self):
        """隐藏进度覆盖层"""
        if hasattr(self, 'progress_overlay') and self.progress_overlay:
            # 停止动画
            if hasattr(self, 'spinner_animation'):
                self.spinner_animation.stop()
            
            # 淡出效果
            fade_out = self.animation_manager.fade_out_widget(self.progress_overlay, duration=300)
            fade_out.finished.connect(self.progress_overlay.deleteLater)
            
            # 清理引用
            self.progress_overlay = None
    
    def add_loading_animation(self, widget):
        """添加加载动画"""
        # 创建加载指示器
        loading_label = QLabel("⏳ 处理中...")
        loading_label.setParent(widget)
        loading_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.7);
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
            }
        """)
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.adjustSize()
        
        # 居中显示
        x = (widget.width() - loading_label.width()) // 2
        y = (widget.height() - loading_label.height()) // 2
        loading_label.move(x, y)
        
        loading_label.show()
        
        # 脉冲动画
        self.animation_manager.pulse_widget(loading_label, duration=1000, repeat=-1)  # 无限循环
        
        return loading_label
    
    def create_status_indicator(self, status="ready"):
        """创建状态指示器"""
        if not hasattr(self, 'status_indicator'):
            self.status_indicator = QLabel()
            self.status_indicator.setFixedSize(20, 20)
            self.status_indicator.setAlignment(Qt.AlignCenter)
        
        status_styles = {
            "ready": {
                "color": "#28A745",
                "text": "●",
                "tooltip": "系统就绪"
            },
            "processing": {
                "color": "#FFC107", 
                "text": "●",
                "tooltip": "正在处理"
            },
            "error": {
                "color": "#DC3545",
                "text": "●", 
                "tooltip": "发生错误"
            },
            "success": {
                "color": "#17A2B8",
                "text": "●",
                "tooltip": "处理完成"
            }
        }
        
        style_info = status_styles.get(status, status_styles["ready"])
        self.status_indicator.setStyleSheet(f"""
            QLabel {{
                color: {style_info["color"]};
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
        """)
        self.status_indicator.setText(style_info["text"])
        self.status_indicator.setToolTip(style_info["tooltip"])
        
        return self.status_indicator
    
    def add_glassmorphism_effect(self, widget):
        """添加毛玻璃效果"""
        widget.setStyleSheet(widget.styleSheet() + """
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        """)
    
    def _reset_right_image(self):
        """重置右侧图片并添加淡入动画"""
        self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
        self.animation_manager.fade_in_widget(self.right_img, duration=300)
    
    def _restore_detect_button(self, button, original_text):
        """恢复检测按钮状态"""
        button.setText(original_text)
        button.setEnabled(True)
        # 添加恢复动画
        self.animation_manager.bounce_widget(button, duration=300)
    
    def _restore_send_button(self, button, original_text):
        """恢复发送按钮状态"""
        button.setText(original_text)
        button.setEnabled(True)
    
    def start_typewriter_effect(self, text, with_prefix=None):
        """开始打字机效果显示文本"""
        # 停止之前的打字机效果（如果有的话）
        self.stop_typewriter_effect()
        
        # 准备纯文本内容
        self.typewriter_text = text or ""
        self.typewriter_prefix = with_prefix
        self.typewriter_current_index = 0
        
        # 添加AI回复的前缀
        if with_prefix:
            self.chat_display.append(f"<b>{html.escape(with_prefix)}:</b>")
        
        # 添加一个空的段落用于打字机效果
        self.chat_display.append("")
        
        # 创建打字机定时器
        self.typewriter_timer = QTimer()
        self.typewriter_timer.timeout.connect(self.update_typewriter_text)
        
        # 设置基础打字速度
        text_length = len(self.typewriter_text)
        if text_length > 200:
            self.base_typing_speed = 25  # 长文本快一点
        elif text_length > 100:
            self.base_typing_speed = 35  # 中等文本
        else:
            self.base_typing_speed = 50  # 短文本慢一点，更有打字感
            
        # 开始打字
        self.typewriter_timer.start(self.base_typing_speed)
    
    def stop_typewriter_effect(self):
        """停止打字机效果"""
        if hasattr(self, 'typewriter_timer') and self.typewriter_timer:
            self.typewriter_timer.stop()
            self.typewriter_timer = None
    
    def update_typewriter_text(self):
        """更新打字机文本显示"""
        if not hasattr(self, 'typewriter_text') or self.typewriter_current_index >= len(self.typewriter_text):
            # 打字完成 - 显示最终文本（不带光标）
            final_text = self.typewriter_text
            safe_text = html.escape(final_text)
            formatted_text = safe_text.replace("\n", "<br>")
            final_html = f"<div style='white-space: normal;'>{formatted_text}</div>"
            
            # 更新最后一条消息为最终版本
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertHtml(final_html)
            
            self.stop_typewriter_effect()
            # 滚动到底部
            self.animate_chat_message()
            return
        
        # 获取当前要显示的文本片段
        current_text = self.typewriter_text[:self.typewriter_current_index + 1]
        
        # 处理换行和HTML转义
        safe_text = html.escape(current_text)
        formatted_text = safe_text.replace("\n", "<br>")
        
        # 添加闪烁光标
        cursor_style = "color: #007BFF; font-weight: bold; animation: blink 1s infinite;"
        display_html = f"<div style='white-space: normal;'>{formatted_text}<span style='{cursor_style}'>▌</span></div>"
        
        # 更新最后一条消息
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertHtml(display_html)
        
        # 增加字符索引
        self.typewriter_current_index += 1
        
        # 添加自然的打字停顿
        import random
        current_char = self.typewriter_text[self.typewriter_current_index - 1] if self.typewriter_current_index > 0 else ''
        
        # 在标点符号后添加稍长的停顿
        if current_char in '。！？，、；：':
            next_delay = self.base_typing_speed + random.randint(100, 300)
        elif current_char in ' \n':
            next_delay = self.base_typing_speed + random.randint(50, 150)
        else:
            # 随机变化打字速度，模拟真实打字
            next_delay = self.base_typing_speed + random.randint(-10, 20)
        
        # 设置下一次的延迟
        self.typewriter_timer.setInterval(max(20, next_delay))
        
        # 滚动到底部
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # 处理事件以保持界面响应
        QApplication.processEvents()

    def handle_dropped_file(self, file_path):
        """处理拖拽的文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.create_floating_notification("❌ 文件不存在", notification_type="danger")
                return
            
            # 检查文件大小（限制为50MB）
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
            if file_size > 50:
                self.create_floating_notification("❌ 文件过大，请选择小于50MB的图片", notification_type="danger")
                return
            
            # 复制文件到临时目录
            suffix = os.path.splitext(file_path)[1]
            temp_dir = "images/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            save_path = os.path.join(temp_dir, f"tmp_upload{suffix}")
            
            # 复制文件
            shutil.copy2(file_path, save_path)
            
            # 读取并显示图片
            im0 = cv2.imread(save_path)
            if im0 is None:
                self.create_floating_notification("❌ 无法读取图片文件", notification_type="danger")
                return
            
            # 调整图像尺寸
            resize_scale = self.output_size / max(im0.shape[0], im0.shape[1])
            im0_resized = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            
            # 保存调整后的图片用于显示
            display_path = os.path.join(temp_dir, "upload_show_result.jpg")
            cv2.imwrite(display_path, im0_resized)
            
            # 更新界面
            self.img2predict = save_path  # 保存原始文件路径用于检测
            
            # 显示图片（带淡入动画）
            pixmap = QPixmap(display_path)
            if not pixmap.isNull():
                # 缩放图片以适应显示区域
                scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.left_img.set_image(scaled_pixmap)
                
                # 为左侧图片添加淡入动画
                self.animation_manager.fade_in_widget(self.left_img, duration=500)
            else:
                self.create_floating_notification("❌ 图片格式不支持", notification_type="danger")
                return
            
            # 重置右侧图片（带淡出再淡入效果）
            fade_out = self.animation_manager.fade_out_widget(self.right_img, duration=200)
            fade_out.finished.connect(lambda: self._reset_right_image())
            
            # 更新状态标签（带动画）
            self.img_num_label.setText("📊 当前检测结果：待检测")
            self.animation_manager.bounce_widget(self.img_num_label, duration=400)
            
            # 显示成功通知
            file_name = os.path.basename(file_path)
            self.create_floating_notification(f"✅ 图片 {file_name} 上传成功！", notification_type="success")
            
            # 添加上传动画效果
            self.animation_manager.bounce_widget(self.left_img, duration=400)
            
        except Exception as e:
            self.create_floating_notification(f"❌ 上传失败：{str(e)}", notification_type="danger")
            print(f"拖拽上传错误：{e}")

    def enable_drag_drop_for_window(self):
        """为整个窗口启用拖拽功能"""
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        """窗口拖拽进入事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            # 检查是否有图片文件
            has_image = any(self.is_image_file(url.toLocalFile()) for url in urls)
            if has_image:
                event.acceptProposedAction()
                # 显示拖拽提示
                self.show_drag_overlay(True)
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        """窗口拖拽离开事件"""
        self.show_drag_overlay(False)
        super().dragLeaveEvent(event)
    
    def show_drag_overlay(self, show):
        """显示或隐藏拖拽覆盖层"""
        if show:
            # 创建拖拽提示覆盖层
            if not hasattr(self, 'drag_overlay'):
                self.drag_overlay = QLabel(self)
                self.drag_overlay.setText("📤 释放以上传图片")
                self.drag_overlay.setAlignment(Qt.AlignCenter)
                self.drag_overlay.setStyleSheet("""
                    QLabel {
                        background: rgba(0, 123, 255, 0.8);
                        color: white;
                        font-size: 24px;
                        font-weight: bold;
                        border: 3px dashed white;
                        border-radius: 15px;
                    }
                """)
            
            # 设置覆盖层大小和位置
            self.drag_overlay.resize(self.size())
            self.drag_overlay.move(0, 0)
            self.drag_overlay.show()
            self.drag_overlay.raise_()  # 置于最前
        else:
            # 隐藏覆盖层
            if hasattr(self, 'drag_overlay'):
                self.drag_overlay.hide()
    
    def dropEvent(self, event):
        """窗口拖拽放置事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                # 查找第一个图片文件
                image_file = None
                for url in urls:
                    file_path = url.toLocalFile()
                    if self.is_image_file(file_path):
                        image_file = file_path
                        break
                
                if image_file:
                    self.handle_dropped_file(image_file)
                    event.acceptProposedAction()
                    
                    # 如果拖拽了多个文件，提示只处理第一个图片
                    if len(urls) > 1:
                        self.create_floating_notification("ℹ️ 检测到多个文件，已处理第一个图片文件", notification_type="info")
                else:
                    self.create_floating_notification("❌ 未找到支持的图片文件", notification_type="danger")
        else:
            event.ignore()
        
        # 隐藏拖拽覆盖层
        self.show_drag_overlay(False)
    
    def is_image_file(self, file_path):
        """检查文件是否为支持的图片格式"""
        if not file_path:
            return False
        
        supported_formats = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif']
        file_extension = os.path.splitext(file_path.lower())[1]
        return file_extension in supported_formats

    def create_status_bar(self):
        """创建状态栏显示操作提示"""
        # 创建状态栏
        self.status_bar = QLabel()
        self.status_bar.setText("💡 提示：拖拽图片文件到界面上传 | 拖拽分割条调整界面 | F11:全屏 | Ctrl+O:上传 | Ctrl+R:重置 | Ctrl+S:保存 | Ctrl+L:恢复")
        self.status_bar.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FA, stop:1 #E9ECEF);
                border-top: 2px solid #DEE2E6;
                padding: 8px 15px;
                font-size: 12px;
                color: #495057;
                font-weight: bold;
            }
        """)
        
        # 将状态栏添加到主窗口底部
        main_layout = QVBoxLayout()
        main_widget = QWidget()
        
        # 获取当前的标签页内容
        current_widget = self.widget(1)  # 图片检测页面
        if current_widget:
            # 创建新的容器来包含原内容和状态栏
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)
            
            # 移除原来的布局，重新添加
            original_layout = current_widget.layout()
            if original_layout:
                # 获取分割器
                splitter = original_layout.itemAt(0).widget()
                original_layout.removeWidget(splitter)
                
                # 添加到新容器
                container_layout.addWidget(splitter)
                container_layout.addWidget(self.status_bar)
                container.setLayout(container_layout)
                
                # 替换标签页内容
                self.removeTab(1)
                self.insertTab(1, container, '图片检测+AI助手')
                self.setTabIcon(1, QIcon(ICON_IMAGE))

    def refresh_config_values(self):
        """刷新配置界面的显示值"""
        if hasattr(self, 'config_output_size_value'):
            self.config_output_size_value.setText(str(self.output_size))
            self.config_imgsz_value.setText(str(self.imgsz))
            self.config_conf_thres_value.setText(str(self.conf_thres))
            self.config_iou_thres_value.setText(str(self.iou_thres))
            
            # 强制重绘输入框
            self.config_output_size_value.repaint()
            self.config_imgsz_value.repaint()
            self.config_conf_thres_value.repaint()
            self.config_iou_thres_value.repaint()
            
            print(f"配置值已刷新: output_size={self.output_size}, imgsz={self.imgsz}, conf_thres={self.conf_thres}, iou_thres={self.iou_thres}")

    def save_config_change(self):
        #
        print("保存配置修改的结果")
        try:
            self.output_size = int(self.config_output_size_value.text())
            self.imgsz = int(self.config_imgsz_value.text())
            self.conf_thres = float(self.config_conf_thres_value.text())
            self.iou_thres = float(self.config_iou_thres_value.text())
            ###
            self.save_txt = self.config_save_txt_value.isChecked()
            self.save_conf = self.config_save_conf_value.isChecked()
            self.save_crop = self.config_save_crop_value.isChecked()
            self.use_tta = self.config_tta_value.isChecked()

            self.show_message(QMessageBox.Information, "配置文件保存成功", "配置文件保存成功")
        except Exception as e:
            print(f"配置保存失败: {e}")
            self.show_message(QMessageBox.Warning, "配置文件保存失败", f"配置文件保存失败: {str(e)}")

    def closeEvent(self, event):
        """用户退出事件"""
        reply = self.show_message(QMessageBox.Question,
                                  'quit',
                                  "Are you sure?",
                                  QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        font_title = QFont('Microsoft YaHei UI', 14)
        self.setWindowTitle("识别系统登陆界面")
        self.resize(800, 600)
        # 深蓝主题
        self.setStyleSheet(
            """
            QWidget { background: #000000; color: #E9EEF6; }
            QLineEdit { background: #FFFFFF; border: 2px solid #00E5FF; border-radius: 8px; padding: 8px 10px; color: #000000; }
            QLineEdit:focus { border: 2px solid #66F0FF; background: #FFFFFF; }
            QLineEdit::placeholder { color: #777777; }
            QLabel { color: #E9EEF6; }
            QPushButton { background: #0A0A0A; color: #E9EEF6; border: 2px solid #00E5FF; border-radius: 10px; padding: 10px 18px; font-weight: bold; }
            QPushButton:hover { border-color: #66F0FF; color: #FFFFFF; }
            QPushButton:pressed { background: #111111; }
            QDialog, QFrame { background: transparent; }
            """
        )

        mid_widget = QWidget()
        window_layout = QFormLayout()
        self.user_name = QLineEdit()
        self.u_password = QLineEdit()
        self.user_name.setPlaceholderText("请输入账号")
        self.u_password.setPlaceholderText("请输入密码")
        window_layout.addRow("账 号：", self.user_name)
        window_layout.addRow("密 码：", self.u_password)
        self.user_name.setEchoMode(QLineEdit.Normal)
        self.u_password.setEchoMode(QLineEdit.Password)
        mid_widget.setLayout(window_layout)

        main_layout = QVBoxLayout()
        a = QLabel(LOGIN_TITLE)
        a.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(a)
        main_layout.addWidget(mid_widget)

        login_button = QPushButton("立即登陆")
        login_button.clicked.connect(self.login)
        reg_button = QPushButton("注册账号")
        reg_button.clicked.connect(self.open_register)
        main_layout.addWidget(login_button)
        main_layout.addWidget(reg_button)

        self.setLayout(main_layout)
        self.mainWindow = MainWindow()
        self.setFont(font_title)

    def login(self):
        user_name = self.user_name.text()
        pwd = self.u_password.text()
        is_ok = self._validate_user(user_name, pwd)
        if is_ok:
            self.mainWindow.show()
            self.close()
        else:
            QMessageBox.warning(self, "账号密码不匹配", "请输入正确的账号密码")

    def open_register(self):
        self.reg_window = RegisterWindow(self)
        self.reg_window.exec()

    def _validate_user(self, username: str, password: str) -> bool:
        # 读取本地 users.json；若不存在则创建包含默认账号
        users = {}
        try:
            if os.path.exists(USERS_DB_PATH):
                with open(USERS_DB_PATH, 'r', encoding='utf-8') as f:
                    users = json.load(f) or {}
        except Exception:
            users = {}

        # 注入默认账号，便于首次使用
        if USERNAME and PASSWORD:
            users.setdefault(USERNAME, PASSWORD)

        # 校验
        ok = username in users and users.get(username) == password

        # 将users回写，保证默认账号持久化
        try:
            with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return ok

class RegisterWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("注册新账号")
        self.resize(420, 260)
        # 深蓝主题
        self.setStyleSheet(
            """
            QWidget { background: #000000; color: #E9EEF6; }
            QLineEdit { background: #FFFFFF; border: 2px solid #00E5FF; border-radius: 8px; padding: 8px 10px; color: #000000; }
            QLineEdit:focus { border: 2px solid #66F0FF; background: #FFFFFF; }
            QLineEdit::placeholder { color: #777777; }
            QLabel { color: #E9EEF6; }
            QPushButton { background: #0A0A0A; color: #E9EEF6; border: 2px solid #00E5FF; border-radius: 10px; padding: 10px 18px; font-weight: bold; }
            QPushButton:hover { border-color: #66F0FF; color: #FFFFFF; }
            QPushButton:pressed { background: #111111; }
            """
        )
        layout = QFormLayout()
        self.username = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.password2 = QLineEdit()
        self.password2.setEchoMode(QLineEdit.Password)
        self.username.setPlaceholderText("请输入账号")
        self.password.setPlaceholderText("请输入密码")
        self.password2.setPlaceholderText("请再次输入密码")
        layout.addRow("账 号：", self.username)
        layout.addRow("密 码：", self.password)
        layout.addRow("确认密码：", self.password2)
        btn = QPushButton("注册")
        btn.clicked.connect(self.register)
        v = QVBoxLayout()
        v.addLayout(layout)
        v.addWidget(btn)
        self.setLayout(v)

    def register(self):
        name = self.username.text().strip()
        pwd = self.password.text()
        pwd2 = self.password2.text()
        if not name or not pwd:
            QMessageBox.warning(self, "输入不完整", "请输入账号和密码")
            return
        if pwd != pwd2:
            QMessageBox.warning(self, "两次密码不一致", "请重新输入")
            return
        users = {}
        try:
            if os.path.exists(USERS_DB_PATH):
                with open(USERS_DB_PATH, 'r', encoding='utf-8') as f:
                    users = json.load(f) or {}
        except Exception:
            users = {}

        if name in users:
            QMessageBox.warning(self, "账号已存在", "请更换账号")
            return
        users[name] = pwd
        try:
            with open(USERS_DB_PATH, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "注册成功", "账号已创建，可以登录")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入用户库失败：{e}")


# todo 添加模型参数的修改，以及添加对文件夹图像的加载
if __name__ == "__main__":
    app = QApplication(sys.argv)
    login = LoginWindow()
    login.show()
    sys.exit(app.exec())