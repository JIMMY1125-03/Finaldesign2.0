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
# 添加API调用相关导入
import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role

# 设置API密钥
dashscope.api_key = 'sk-0cb0cdd81cc74719b30bd6458338a340'

# 常用的字符串常量
WINDOW_TITLE = "农业害虫检测系统"  # 系统上方标题
WELCOME_SENTENCE = "欢迎使用基于yolov11的农业害虫检测系统"  # 欢迎的句子
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
        self.detection_results = results

    def get_detection_context(self):
        """获取检测结果的上下文信息"""
        if not self.detection_results:
            return "目前还没有进行图片检测，请先上传图片并进行检测。"

        context = "根据刚才的图片检测结果：\n"
        for class_name, count in self.detection_results.items():
            context += f"- 检测到 {count} 个 {class_name}\n"
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

        # 初始化ChatBot
        self.chatbot = ChatBot()
        # 存储最新的检测结果
        self.latest_detection_results = {}

        # self.model_path = "runs/detect/yolo11-n/weights/best.pt"  # todo 指明模型加载的位置的设备
        self.model_path = r"D:\校内\新建文件夹\Finaldesign2.0\yolo12-litchi\42_demo\runs\best.pt"  # todo 指明模型加载的位置的设备
        self.model = self.model_load(weights=self.model_path)

        # 设置现代化样式
        self.setup_modern_style()

        self.initUI()  # 初始化图形化界面

    def setup_modern_style(self):
        """设置现代化样式"""
        # 设置主窗口样式
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #C0C0C0;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FA, stop:1 #E9ECEF);
                border-radius: 8px;
            }

            QTabWidget::tab-bar {
                alignment: center;
            }

            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6C757D, stop:1 #495057);
                color: white;
                padding: 12px 24px;
                margin: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }

            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #007BFF, stop:1 #0056B3);
                color: white;
            }

            QTabBar::tab:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #007BFF, stop:1 #0056B3);
                color: white;
            }

            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FA, stop:1 #E9ECEF);
            }
        """)

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
        img_detection_layout = QHBoxLayout()  # 改为水平布局

        # 左侧：图片检测区域
        left_detection_widget = QWidget()
        left_detection_layout = QVBoxLayout()

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

        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap(IMAGE_LEFT_INIT))
        self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)

        self.img_num_label = QLabel("📊 当前检测结果：待检测")
        self.img_num_label.setFont(font_main)
        self.img_num_label.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFF3E0, stop:1 #FFE0B2);
                border: 2px solid #FF9800;
                border-radius: 8px;
                padding: 8px;
                margin: 5px;
                font-weight: bold;
            }
        """)

        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        # 现代化按钮样式
        modern_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #007BFF, stop:1 #0056B3);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
                margin: 5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0056B3, stop:1 #004085);
                transform: translateY(-2px);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #004085, stop:1 #003366);
                transform: translateY(0px);
            }
        """

        up_img_button.setStyleSheet(modern_button_style)
        det_img_button.setStyleSheet(modern_button_style)

        left_detection_layout.addWidget(img_detection_title)
        left_detection_layout.addWidget(mid_img_widget)
        left_detection_layout.addWidget(self.img_num_label)
        left_detection_layout.addWidget(up_img_button)
        left_detection_layout.addWidget(det_img_button)
        left_detection_widget.setLayout(left_detection_layout)

        # 右侧：AI聊天区域
        right_chat_widget = QWidget()
        right_chat_layout = QVBoxLayout()

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
        self.chat_display.setMinimumHeight(300)
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
        send_button.clicked.connect(self.send_message)
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

        # 查看检测结果按钮
        view_results_button = QPushButton("查看当前检测结果")
        view_results_button.clicked.connect(self.show_detection_results)
        view_results_button.setFont(font_main)
        view_results_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #17A2B8, stop:1 #138496);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
                margin: 3px;
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
        clear_button = QPushButton("清空聊天记录")
        clear_button.clicked.connect(self.clear_chat)
        clear_button.setFont(font_main)
        clear_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #DC3545, stop:1 #C82333);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13px;
                margin: 3px;
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

        right_chat_layout.addWidget(chat_title)
        right_chat_layout.addWidget(self.chat_display)
        right_chat_layout.addWidget(input_widget)
        right_chat_layout.addWidget(view_results_button)
        right_chat_layout.addWidget(clear_button)
        right_chat_widget.setLayout(right_chat_layout)

        # 将左右两个区域添加到主布局
        img_detection_layout.addWidget(left_detection_widget, 1)  # 1表示拉伸比例
        img_detection_layout.addWidget(right_chat_widget, 1)  # 1表示拉伸比例
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
        config_save_txt_label.setStyleSheet("color: black;")
        self.config_save_txt_value = QRadioButton("True")
        self.config_save_txt_value.setChecked(False)
        self.config_save_txt_value.setAutoExclusive(False)
        self.config_save_txt_value.setStyleSheet("""
            QRadioButton {
                font-size: 12px;
                color: #2C3E50;
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
        """)
        config_grid_layout.addWidget(config_save_txt_label, 0, 0)  # 注意：调整网格布局的行号（避免重叠）
        config_grid_layout.addWidget(self.config_save_txt_value, 0, 1)

        # 2. 再定义 config_tta_value（此时可以安全复制 config_save_txt_value 的样式）
        config_tta_label = QLabel("🧪 启用测试时增强(TTA)")
        config_tta_label.setStyleSheet("color: black;")
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
        config_output_size_label.setStyleSheet("color: black;")
        self.config_output_size_value = QLineEdit("")
        self.config_output_size_value.setText(str(self.output_size))
        self.config_output_size_value.setStyleSheet("""
            QLineEdit {
                background: white;
                border: 2px solid #E9ECEF;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
                color: black;
            }
            QLineEdit:focus {
                border: 2px solid #007BFF;
                background: #F8F9FF;
            }
        """)
        config_grid_layout.addWidget(config_output_size_label, 2, 0)  # 行号继续递增
        config_grid_layout.addWidget(self.config_output_size_value, 2, 1)

        # 推理图像尺寸
        config_imgsz_label = QLabel("🧩 推理图像尺寸(imgsz)")
        config_imgsz_label.setStyleSheet("color: black;")
        self.config_imgsz_value = QLineEdit("")
        self.config_imgsz_value.setText(str(self.imgsz))
        self.config_imgsz_value.setStyleSheet(self.config_output_size_value.styleSheet())
        config_grid_layout.addWidget(config_imgsz_label, 3, 0)
        config_grid_layout.addWidget(self.config_imgsz_value, 3, 1)

        # 检测模型置信度阈值
        config_conf_thres_label = QLabel("🎯 检测模型置信度阈值")
        config_conf_thres_label.setStyleSheet("color: black;")
        self.config_conf_thres_value = QLineEdit("")
        self.config_conf_thres_value.setText(str(self.conf_thres))
        self.config_conf_thres_value.setStyleSheet(self.config_output_size_value.styleSheet())
        config_grid_layout.addWidget(config_conf_thres_label, 4, 0)
        config_grid_layout.addWidget(self.config_conf_thres_value, 4, 1)

        # 检测模型IOU阈值
        config_iou_thres_label = QLabel("📏 检测模型IOU阈值")
        config_iou_thres_label.setStyleSheet("color: black;")
        self.config_iou_thres_value = QLineEdit("")
        self.config_iou_thres_value.setText(str(self.iou_thres))
        self.config_iou_thres_value.setStyleSheet(self.config_output_size_value.styleSheet())
        config_grid_layout.addWidget(config_iou_thres_label, 5, 0)
        config_grid_layout.addWidget(self.config_iou_thres_value, 5, 1)

        # 推理时是否保存置信度（原有组件，行号继续递增）
        config_save_conf_label = QLabel("📊 推理时是否保存置信度")
        config_save_conf_label.setStyleSheet("color: black;")
        self.config_save_conf_value = QRadioButton("True")
        self.config_save_conf_value.setChecked(False)
        self.config_save_conf_value.setAutoExclusive(False)
        self.config_save_conf_value.setStyleSheet(self.config_save_txt_value.styleSheet())
        config_grid_layout.addWidget(config_save_conf_label, 6, 0)
        config_grid_layout.addWidget(self.config_save_conf_value, 6, 1)

        # 推理时是否保存切片文件（原有组件）
        config_save_crop_label = QLabel("✂️ 推理时是否保存切片文件")
        config_save_crop_label.setStyleSheet("color: black;")
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

        # ********************* todo 布局修改和颜色变换等相关插件 *****************************

    def show_message(self, icon, title, text, buttons=QMessageBox.Ok):
        """统一的消息框，强制文本为黑色。"""
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        msg.setStyleSheet("QLabel { color: black; } QPushButton { color: black; }")
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
        """上传图像，图像要尽可能保证是中文格式"""
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')  # 选择图像
        if fileName:  # 如果存在文件名称则对图像进行处理
            # 将图像转移到当前目录下，解决中文的问题
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)  # 将图像转移到images目录下并且修改为英文的形式
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            # 调整图像的尺寸，让图像可以适应图形化的界面
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = save_path  # 给变量进行赋值方便后面实际进行读取
            # 将图像显示在界面上并将预测的文字内容进行初始化
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
            self.img_num_label.setText("当前检测结果：待检测")

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

        # 2) 确保输出目录存在（首次运行可能未创建，避免写文件失败）
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
        self.img_num_label.setText("当前检测结果\n{}".format(result_info))

        # 存储检测结果到ChatBot中，只存储检测到的类别
        self.latest_detection_results = {result_names[idx_cls]: cls_num for idx_cls, cls_num in enumerate(result_nums)
                                         if cls_num > 0}
        self.chatbot.set_detection_results(self.latest_detection_results)

        self.show_message(QMessageBox.Information, "检测成功", "日志已保存！")

    def check_record(self):
        """打开历史记录文件夹"""
        os.startfile(osp.join(os.path.abspath(os.path.dirname(__file__)), "record"))

    def send_message(self):
        """发送消息到AI助手"""
        user_input = self.chat_input.text().strip()
        if not user_input:
            return

        # 显示用户消息
        self.chat_display.append(f"<b>您:</b> {user_input}")
        self.chat_input.clear()

        # 显示等待提示
        self.chat_display.append("<i>AI正在思考中...</i>")
        QApplication.processEvents()  # 立即更新界面

        try:
            # 检查是否有检测结果，如果有则显示提示
            if self.latest_detection_results:
                detection_summary = "📊 当前检测结果："
                for class_name, count in self.latest_detection_results.items():
                    detection_summary += f"\n   • {class_name}: {count}个"
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
                    self._append_preserved("\n\n".join(parts), with_prefix="AI")
                    displayed = True
            except Exception:
                pass

            if not displayed:
                simplified = self._simplify_ai_text(response)
                self._append_preserved(simplified, with_prefix="AI")

        except Exception as e:
            # 移除等待提示并显示错误信息
            cursor = self.chat_display.textCursor()
            cursor.movePosition(cursor.End)
            cursor.movePosition(cursor.StartOfLine, cursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # 删除换行符

            self.chat_display.append(f"<b>错误:</b> {str(e)}")

        # 滚动到底部
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

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
        self.chat_display.clear()
        self.chatbot = ChatBot()  # 重新初始化聊天机器人

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
        except:
            self.show_message(QMessageBox.Warning, "配置文件保存失败", "配置文件保存失败")

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