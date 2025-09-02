import os
import cv2
import csv
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from collections import Counter
import A_identify_v8

from CONFIG import *
class SystemUi(QMainWindow):
    def __init__(self):
        # 鼠标拖动界面设置参数
        self.press_x = 0
        self.press_y = 0
        # 页面参数
        self.identify_api = A_identify_v8.Identify()  # 加载检测类
        self.input_image = None  # 输入图像
        self.output_image = None  # 输出图像
        self.output_video = None  # 输出视频
        self.identify_labels = []  # 检测结果
        self.save_video_flag = False  # 保存视频标志位
        self.save_path = "./A_output/"  # 保存的路径目录
        self.timer_video = QTimer()  # 设置定时器
        self.timer_video.timeout.connect(self.show_video)  # 定时调用视频流检测函数
        self.identify_record = []  # 查询记录
        self.identify_record_path = ""  # 查询图像视频路径
        self.admin_list = []  # 账号密码
        self.functions = []  # 左侧按钮列表
        self.function_flag = 1  # 目前按钮标识
        self.right_widgets = []  # 右侧界面列表
        # 页面生成
        super(SystemUi, self).__init__()
        self.resize(1200, 720)
        self.setWindowFlag(Qt.FramelessWindowHint)  # 隐藏边框
        self.setStyleSheet("QMainWindow{border: 1px solid #303030;}")

        self.label = QLabel(self)
        self.label.setText(MAIN_TITLE)
        self.label.setFixedSize(1050, 60)
        self.label.move(0, 0)
        self.label.setStyleSheet("QLabel{background:#303030;}QLabel{color:#ffffff;border:none;font-weight:600;"
                                 "font-size:16px;font-family:'微软雅黑';padding-left:30px}")

        self.exit = QPushButton(self)
        self.exit.setText("☀ 退出系统")
        self.exit.resize(150, 60)
        self.exit.move(1050, 0)
        self.exit.setStyleSheet("QPushButton{background:#303030;text-align:center;border:none;font-weight:600;"
                                "color:#909090;font-size:14px;}")
        self.exit.setCursor(Qt.PointingHandCursor)
        self.exit.clicked.connect(self.close)
        # 左侧文本显示区
        self.function_label = QLabel(self)
        self.function_label.setText("YOLOv5\n目标检测系统")
        self.function_label.setFixedSize(219, 220)
        self.function_label.move(1, 60)
        self.function_label.setAlignment(Qt.AlignCenter)
        self.function_label.setStyleSheet(
            "QLabel{background:#ffffff;color:#2c3a45;border:none;font-weight:600;font-size:20px;font-family:'微软雅黑';}")
        # 左侧按钮1
        self.function1 = QPushButton(self)
        self.function1.setText("⚪  目标检测")
        self.function1.resize(219, 70)
        self.function1.move(1, 280)
        self.function1.setStyleSheet(
            "QPushButton{background:#e6e6e6;color:#2c3a45;text-align:left;padding-left:50px;"
            "border:none;font-weight:600;font-size:15px;font-family:'微软雅黑';}"
            "QPushButton:hover{background:#e6e6e6;}")
        self.function1.setCursor(Qt.PointingHandCursor)
        self.function1.clicked.connect(lambda: self.change_button(self.functions[self.function_flag - 1], self.function1, 1))
        # 左侧按钮2
        self.function2 = QPushButton(self)
        self.function2.setText("⚪  检测查询")
        self.function2.resize(219, 70)
        self.function2.move(1, 350)
        self.function2.setStyleSheet(
            "QPushButton{background:#ffffff;color:#2c3a45;text-align:left;padding-left:50px;"
            "border:none;font-weight:600;font-size:15px;font-family:'微软雅黑';}"
            "QPushButton:hover{background:#e6e6e6;}")
        self.function2.setCursor(Qt.PointingHandCursor)
        self.function2.clicked.connect(lambda: self.change_button(self.functions[self.function_flag - 1], self.function2, 2))
        # 左侧按钮3
        self.function3 = QPushButton(self)
        self.function3.setText("⚪  账号管理")
        self.function3.resize(219, 70)
        self.function3.move(1, 420)
        self.function3.setStyleSheet(
            "QPushButton{background:#ffffff;color:#2c3a45;text-align:left;padding-left:50px;"
            "border:none;font-weight:600;font-size:15px;font-family:'微软雅黑';}"
            "QPushButton:hover{background:#e6e6e6;}")
        self.function3.setCursor(Qt.PointingHandCursor)
        self.function3.clicked.connect(lambda: self.change_button(self.functions[self.function_flag - 1], self.function3, 3))
        # 左侧下方补白区
        self.function_label2 = QLabel(self)
        self.function_label2.setFixedSize(219, 229)
        self.function_label2.move(1, 490)
        self.function_label2.setStyleSheet("QLabel{background:#ffffff;}")

        # ===== 右侧界面1 ===== #
        self.right_widget1 = QWidget(self)
        self.right_widget1.resize(979, 659)
        self.right_widget1.move(220, 60)
        # 原图显示区域
        self.input_img = QLabel(self.right_widget1)
        self.input_img.setText("输入显示区")
        self.input_img.setFixedSize(740, 290)
        self.input_img.move(19, 10)
        self.input_img.setAlignment(Qt.AlignCenter)
        self.input_img.setStyleSheet("QLabel{border: 2px solid gray;font-size:30px;font-family:'黑体';color:#999999;}")
        # 检测图显示区域
        self.output_img = QLabel(self.right_widget1)
        self.output_img.setText("输出显示区")
        self.output_img.setFixedSize(740, 290)
        self.output_img.move(19, 305)
        self.output_img.setAlignment(Qt.AlignCenter)
        self.output_img.setStyleSheet("QLabel{border: 2px solid gray;font-size:30px;font-family:'黑体';color:#999999;}")
        # 右侧按钮1(图像检测)
        self.btn1 = QPushButton(self.right_widget1)
        self.btn1.setText("图像检测")
        self.btn1.resize(180, 45)
        self.btn1.move(19, 605)
        self.btn1.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                "QPushButton:hover{background:#e6e6e6;}")
        self.btn1.setFocusPolicy(Qt.NoFocus)
        self.btn1.setCursor(Qt.PointingHandCursor)
        self.btn1.clicked.connect(self.show_image)
        # 右侧按钮2(视频检测)
        self.btn2 = QPushButton(self.right_widget1)
        self.btn2.setText("开启视频检测")
        self.btn2.resize(180, 45)
        self.btn2.move(205, 605)
        self.btn2.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                "QPushButton:hover{background:#e6e6e6;}")
        self.btn2.setFocusPolicy(Qt.NoFocus)
        self.btn2.setCursor(Qt.PointingHandCursor)
        self.btn2.clicked.connect(self.video_identify)
        # 右侧按钮3(实时检测)
        self.btn3 = QPushButton(self.right_widget1)
        self.btn3.setText("开启实时检测")
        self.btn3.resize(180, 45)
        self.btn3.move(390, 605)
        self.btn3.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                "QPushButton:hover{background:#e6e6e6;}")
        self.btn3.setFocusPolicy(Qt.NoFocus)
        self.btn3.setCursor(Qt.PointingHandCursor)
        self.btn3.clicked.connect(self.camera_identify)
        # 右侧按钮4(重置)
        self.btn4 = QPushButton(self.right_widget1)
        self.btn4.setText("重置系统操作")
        self.btn4.resize(180, 45)
        self.btn4.move(575, 605)
        self.btn4.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                "QPushButton:hover{background:#e6e6e6;}")
        self.btn4.setFocusPolicy(Qt.NoFocus)
        self.btn4.setCursor(Qt.PointingHandCursor)
        self.btn4.clicked.connect(self.reset1_2)
        # 创建一个小区域
        self.right_widget1_widget = QWidget(self.right_widget1)
        self.right_widget1_widget.resize(200, 288)
        self.right_widget1_widget.move(778, 0)
        self.right_widget1_widget.setStyleSheet("QWidget{background:#ffffff;border:none;}")
        # 选择模型文本框
        self.pt_label = QLabel(self.right_widget1_widget)
        self.pt_label.setText("选择模型")
        self.pt_label.setFixedSize(190, 30)
        self.pt_label.move(5, 5)
        self.pt_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # 选择模型显示框
        self.pt_show = QLabel(self.right_widget1_widget)
        self.pt_show.setText("")
        self.pt_show.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        self.pt_show.setAlignment(Qt.AlignCenter)
        self.pt_show.setStyleSheet("QLabel{background:#ffffff;color:#999999;border:none;font-weight:600;"
                                   "font-size:15px;font-family:'黑体';}")
        self.scroll_area_pt = QScrollArea(self.right_widget1_widget)
        self.scroll_area_pt.resize(100, 50)
        self.scroll_area_pt.move(5, 40)
        self.scroll_area_pt.setWidget(self.pt_show)
        self.scroll_area_pt.setWidgetResizable(True)  # 设置 QScrollArea 大小可调整
        self.scroll_area_pt.setStyleSheet("QScrollArea{border: 1px solid #dddddd;}")
        # 选择模型按钮
        self.pt_button = QPushButton(self.right_widget1_widget)
        self.pt_button.setText("选择模型")
        self.pt_button.resize(85, 50)
        self.pt_button.move(110, 40)
        self.pt_button.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                     "font-size:18px;font-family:'黑体';border: 1px solid #dddddd;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.pt_button.setFocusPolicy(Qt.NoFocus)
        self.pt_button.setCursor(Qt.PointingHandCursor)
        self.pt_button.clicked.connect(self.input_pt)
        # 置信度阈值文本框
        self.conf_label = QLabel(self.right_widget1_widget)
        self.conf_label.setText("置信度conf")
        self.conf_label.setFixedSize(190, 30)
        self.conf_label.move(5, 95)
        self.conf_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # 置信度阈值调节框
        self.conf_spin_box = QDoubleSpinBox(self.right_widget1_widget)
        self.conf_spin_box.resize(55, 25)
        self.conf_spin_box.move(5, 130)
        self.conf_spin_box.setMinimum(0.0)  # 最小值
        self.conf_spin_box.setMaximum(1.0)  # 最大值
        self.conf_spin_box.setSingleStep(0.01)  # 步长
        self.conf_spin_box.setValue(self.identify_api.conf_thres)  # 当前值
        self.conf_spin_box.setStyleSheet("QDoubleSpinBox{background:#ffffff;color:#999999;font-size:14px;"
                                         "font-weight:600;border: 1px solid #dddddd;}")
        self.conf_spin_box.valueChanged.connect(self.change_conf_spin_box)  # 绑定函数
        # 置信度阈值滚动条
        self.conf_slider = QSlider(Qt.Horizontal, self.right_widget1_widget)
        self.conf_slider.resize(130, 25)
        self.conf_slider.move(65, 130)
        self.conf_slider.setMinimum(0)  # 最小值
        self.conf_slider.setMaximum(100)  # 最大值
        self.conf_slider.setSingleStep(1)  # 步长
        self.conf_slider.setValue(int(self.identify_api.conf_thres * 100))  # 当前值
        self.conf_slider.setCursor(Qt.PointingHandCursor)  # 鼠标光标变为手指
        self.conf_slider.setStyleSheet("QSlider::groove:horizontal{border:1px solid #999999;height:25px;}"
                                       "QSlider::handle:horizontal{background:#ffcc00;width:24px;border-radius:12px;}"
                                       "QSlider::add-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                       "x2:0,y2:0,stop:0 #d9d9d9,stop:0.25 #d9d9d9,stop:0.5 #d9d9d9,stop:1 #d9d9d9);}"
                                       "QSlider::sub-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                       "x2:0,y2:0,stop:0 #777777,stop:0.25 #777777,stop:0.5 #777777,stop:1 #777777);}")
        self.conf_slider.valueChanged.connect(self.change_conf_slider)  # 绑定函数
        # 非极大抑制IoU阈值文本框
        self.iou_label = QLabel(self.right_widget1_widget)
        self.iou_label.setText("非极大抑制IoU")
        self.iou_label.setFixedSize(190, 30)
        self.iou_label.move(5, 160)
        self.iou_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # 非极大抑制IoU阈值调节框
        self.iou_spin_box = QDoubleSpinBox(self.right_widget1_widget)
        self.iou_spin_box.resize(55, 25)
        self.iou_spin_box.move(5, 195)
        self.iou_spin_box.setMinimum(0.0)  # 最小值
        self.iou_spin_box.setMaximum(1.0)  # 最大值
        self.iou_spin_box.setSingleStep(0.01)  # 步长
        self.iou_spin_box.setValue(self.identify_api.iou_thres)  # 当前值
        self.iou_spin_box.setStyleSheet("QDoubleSpinBox{background:#ffffff;color:#999999;font-size:14px;"
                                        "font-weight:600;border: 1px solid #dddddd;}")
        self.iou_spin_box.valueChanged.connect(self.change_iou_spin_box)  # 绑定函数
        # 非极大抑制IoU阈值滚动条
        self.iou_slider = QSlider(Qt.Horizontal, self.right_widget1_widget)
        self.iou_slider.resize(130, 25)
        self.iou_slider.move(65, 195)
        self.iou_slider.setMinimum(0)  # 最小值
        self.iou_slider.setMaximum(100)  # 最大值
        self.iou_slider.setSingleStep(1)  # 步长
        self.iou_slider.setValue(int(self.identify_api.iou_thres * 100))  # 当前值
        self.iou_slider.setCursor(Qt.PointingHandCursor)  # 鼠标光标变为手指
        self.iou_slider.setStyleSheet("QSlider::groove:horizontal{border:1px solid #999999;height:25px;}"
                                      "QSlider::handle:horizontal{background:#ffcc00;width:24px;border-radius:12px;}"
                                      "QSlider::add-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                      "x2:0,y2:0,stop:0 #d9d9d9,stop:0.25 #d9d9d9,stop:0.5 #d9d9d9,stop:1 #d9d9d9);}"
                                      "QSlider::sub-page:horizontal{background:qlineargradient(spread:pad,x1:0,y1:1,"
                                      "x2:0,y2:0,stop:0 #777777,stop:0.25 #777777,stop:0.5 #777777,stop:1 #777777);}")
        self.iou_slider.valueChanged.connect(self.change_iou_slider)  # 绑定函数
        # 保存检测结果文本框
        self.save_label = QLabel(self.right_widget1_widget)
        self.save_label.setText("是否保存检测结果")
        self.save_label.resize(190, 25)
        self.save_label.move(5, 225)
        self.save_label.setStyleSheet("QLabel{font-size: 18px;color:#999999;font-weight:600;font-family:'微软雅黑'; }")
        # 保存检测结果单选框
        self.save_button_yes = QRadioButton(self.right_widget1_widget)
        self.save_button_yes.setText("  是")
        self.save_button_yes.resize(70, 25)
        self.save_button_yes.move(5, 255)
        self.save_button_yes.setCursor(Qt.PointingHandCursor)  # 鼠标光标变为手指
        self.save_button_yes.setStyleSheet(
            "QRadioButton{font-size: 16px;color:#999999;font-weight:600;font-family:'黑体'; }")
        self.save_button_no = QRadioButton(self.right_widget1_widget)
        self.save_button_no.setText("  否")
        self.save_button_no.resize(70, 25)
        self.save_button_no.move(80, 255)
        self.save_button_no.setCursor(Qt.PointingHandCursor)  # 鼠标光标变为手指
        self.save_button_no.setStyleSheet(
            "QRadioButton{font-size: 16px;color:#999999;font-weight:600;font-family:'黑体'; }")
        self.save_button_no.setChecked(True)  # 默认选中不保存
        # 结果显示区域
        self.result_label = QLabel(self.right_widget1)
        self.result_label.setText("检 测 结 果")
        self.result_label.setFixedSize(200, 32)
        self.result_label.move(778, 290)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("QLabel{background:#ffffff;color:#999999;border:none;font-weight:600;"
                                        "font-size:18px;font-family:'黑体';}")
        self.result = QLabel(self.right_widget1)
        self.result.setText("")
        self.result.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        self.result.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 设置对齐方式为居上居左
        self.result.setStyleSheet("QLabel{background:#ffffff;color:#999999;border:none;font-weight:600;"
                                  "font-size:15px;font-family:'黑体';padding:5px;}")
        self.scroll_area = QScrollArea(self.right_widget1)
        self.scroll_area.resize(200, 336)
        self.scroll_area.move(778, 323)
        self.scroll_area.setWidget(self.result)
        self.scroll_area.setWidgetResizable(True)  # 设置 QScrollArea 大小可调整
        self.scroll_area.setStyleSheet("QScrollArea{border:none;}")
        self.right_widget1.show()

        # ===== 右侧界面2 ===== #
        self.right_widget2 = QWidget(self)
        self.right_widget2.resize(979, 659)
        self.right_widget2.move(220, 55)
        # 日期显示框
        self.date_show = QLabel(self.right_widget2)
        self.date_show.setText("--")
        self.date_show.setFixedSize(350, 40)
        self.date_show.move(40, 20)
        self.date_show.setAlignment(Qt.AlignCenter)
        self.date_show.setStyleSheet(
            " QLabel{padding-left:10px;color:#303030;font-weight:600;font-size:15px;border: 2px solid #000000;"
            "font-family:'微软雅黑'; }")
        # 选择日期按钮
        self.button_date = QPushButton(self.right_widget2)  # 查询
        self.button_date.setText("日期查询")
        self.button_date.resize(152, 40)
        self.button_date.move(388, 20)
        self.button_date.setStyleSheet(
            "QPushButton{background:#ffffff;color:#2c3a45;font-weight:600;font-size:18px;"
            "font-family:'微软雅黑';border:2px solid #000000;}"
            "QPushButton:hover{background:#e6e6e6;}")
        self.button_date.setCursor(Qt.PointingHandCursor)
        self.button_date.clicked.connect(self.select_record_by_date)
        # 全部查询按钮
        self.button_select = QPushButton(self.right_widget2)  # 查询
        self.button_select.setText("全部查询")
        self.button_select.resize(150, 40)
        self.button_select.move(590, 20)
        self.button_select.setStyleSheet(
            "QPushButton{background:#ffffff;color:#2c3a45;font-weight:600;font-size:18px;"
            "font-family:'微软雅黑';border:2px solid #000000;}"
            "QPushButton:hover{background:#e6e6e6;}")
        self.button_select.setCursor(Qt.PointingHandCursor)
        self.button_select.clicked.connect(self.select_all_record)
        # 删除按钮
        self.button_delete = QPushButton(self.right_widget2)  # 删除
        self.button_delete.setText("删除记录")
        self.button_delete.resize(150, 40)
        self.button_delete.move(790, 20)
        self.button_delete.setStyleSheet(
            "QPushButton{background:#ffffff;color:#2c3a45;font-weight:600;font-size:18px;"
            "font-family:'微软雅黑';border:2px solid #000000;}"
            "QPushButton:hover{background:#e6e6e6;}")
        self.button_delete.setCursor(Qt.PointingHandCursor)
        self.button_delete.clicked.connect(self.del_record)
        # 表格
        self.table_view = QTableWidget(self.right_widget2)
        self.table_view.setFocusPolicy(Qt.NoFocus)
        self.table_view.resize(900, 560)
        self.table_view.move(40, 80)
        self.table_view.setColumnCount(4)
        self.table_view.setHorizontalHeaderLabels(['编号', '名称', '检测时间', '查看'])
        self.table_view.verticalHeader().setVisible(False)  # 隐藏列头
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 不可编辑表格内容
        header = self.table_view.horizontalHeader()  # 获取表头
        header.setSectionResizeMode(QHeaderView.Stretch)  # 使表头自适应宽度
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)  # 选择行为选择一行单元格
        self.table_view.setStyleSheet("QTableWidget{color:#555555;font-size:15px;border:2px solid #000000;"
                                      "font-weight:bold;font-family:'黑体';}")
        self.right_widget2.hide()

        # ===== 右侧界面3 ===== #
        self.right_widget3 = QWidget(self)
        self.right_widget3.resize(979, 659)
        self.right_widget3.move(220, 55)
        self.admin_id1 = QLabel(self.right_widget3)
        self.admin_id1.setText("账号")
        self.admin_id1.setFixedSize(150, 40)
        self.admin_id1.move(100, 40)
        self.admin_id1.setAlignment(Qt.AlignCenter)
        self.admin_id1.setStyleSheet(
            " QLabel{color:#303030;border:none;font-weight:600;font-size:25px;font-family:'微软雅黑'; "
            "border: 2px solid #000000;}")

        self.admin_id2 = QLabel(self.right_widget3)
        self.admin_id2.setFixedSize(632, 40)
        self.admin_id2.move(248, 40)
        self.admin_id2.setStyleSheet(
            " QLabel{padding-left:10px;color:#303030;font-weight:600;font-size:18px;font-family:'微软雅黑'; "
            "border: 2px solid #000000; background:#ffffff;}")
        self.admin_pwd1 = QLabel(self.right_widget3)
        self.admin_pwd1.setText("密码")
        self.admin_pwd1.setFixedSize(150, 40)
        self.admin_pwd1.move(100, 100)
        self.admin_pwd1.setAlignment(Qt.AlignCenter)
        self.admin_pwd1.setStyleSheet(
            " QLabel{color:#303030;border:none;font-weight:600;font-size:25px;font-family:'微软雅黑'; border: 2px solid #000000;}")

        self.admin_pwd2 = QLineEdit(self.right_widget3)
        self.admin_pwd2.setEchoMode(QLineEdit.Normal)
        self.admin_pwd2.setFixedSize(632, 40)
        self.admin_pwd2.move(248, 100)
        self.admin_pwd2.setStyleSheet(
            " QLineEdit{padding-left:10px;color:#303030;font-weight:600;font-size:18px;font-family:'微软雅黑'; border: 2px solid #000000;}")

        self.upd_btn = QPushButton(self.right_widget3)
        self.upd_btn.setText("修改密码")
        self.upd_btn.resize(780, 50)
        self.upd_btn.move(100, 160)
        self.upd_btn.setStyleSheet("QPushButton{;text-align:center;border:2px solid #000000;font-weight:600;"
                                   "font-size:25px;border-radius: 25px;}"
                                   "QPushButton:hover{background:#e6e6e6;}")
        self.upd_btn.setCursor(Qt.PointingHandCursor)
        self.upd_btn.clicked.connect(self.upd_password)

        self.reset_btn = QPushButton(self.right_widget3)
        self.reset_btn.setText("注销账号")
        self.reset_btn.resize(780, 50)
        self.reset_btn.move(100, 230)
        self.reset_btn.setStyleSheet("QPushButton{;text-align:center;border:2px solid #000000;font-weight:600;"
                                     "font-size:25px;border-radius: 25px;}"
                                     "QPushButton:hover{background:#e6e6e6;}")
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.clicked.connect(self.reset_admin)
        self.right_widget3.hide()

        #  左边列表按钮列表和右边界面列表
        self.functions = [self.function1, self.function2, self.function3]
        self.right_widgets = [self.right_widget1, self.right_widget2, self.right_widget3, self.right_widget1]

    # ================================ 实现鼠标长按移动窗口功能 ================================ #
    def mousePressEvent(self, event):
        self.press_x = event.x()  # 记录鼠标按下的时候的坐标
        self.press_y = event.y()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()  # 获取移动后的坐标
        if 0 < x < 1050 and 0 < y < 60:
            move_x = x - self.press_x
            move_y = y - self.press_y  # 计算移动了多少
            position_x = self.frameGeometry().x() + move_x
            position_y = self.frameGeometry().y() + move_y  # 计算移动后主窗口在桌面的位置
            self.move(position_x, position_y)  # 移动主窗口

    # ================================ 函数区域 ================================#
    # 点击列表按钮切换界面和切换按钮颜色
    def change_button(self, button_1, button_2, flag):
        if self.btn2.text() == "开启视频检测" and self.btn3.text() == "开启实时检测":
            button_1.setStyleSheet(
                "QPushButton{background:#ffffff;color:#2c3a45;text-align:left;padding-left:50px;"
                "border:none;font-weight:600;font-size:15px;font-family:'微软雅黑';}"
                "QPushButton:hover{background:#e6e6e6;}")
            button_2.setStyleSheet(
                "QPushButton{background:#e6e6e6;color:#2c3a45;text-align:left;padding-left:50px;"
                "border:none;font-weight:600;font-size:15px;font-family:'微软雅黑';}"
                "QPushButton:hover{background:#e6e6e6;}")
            self.right_widgets[self.function_flag - 1].hide()  # 隐藏界面
            self.right_widgets[flag - 1].show()  # 显示界面
            self.function_flag = flag
            if flag == 1:
                self.reset1_1()
            if flag == 2:
                self.reset2()
                self.select_all()
                self.show_table()
            if flag == 3:
                self.show_admin()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("你还处于检测运行中，请先关闭再切换页面！")
            # 显示消息框
            msg_box.exec_()

    # ========== 检测相关函数 =============
    # 选择模型
    def input_pt(self):
        if len(self.pt_show.text()) == 0:
            pt_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型", "./", "*.pt")
            if len(pt_path) > 3:
                pt_flag = self.identify_api.load_pt(pt_path)
                if pt_flag:
                    pt_name = os.path.basename(pt_path)
                    self.pt_show.setText(pt_name)
                    # 创建一个消息框
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("提示")
                    msg_box.setText("加载模型成功！")
                    # 显示消息框
                    msg_box.exec_()
                else:
                    # 创建一个消息框
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("提示")
                    msg_box.setText("加载模型失败！")
                    # 显示消息框
                    msg_box.exec_()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("已经加载模型，请先重置系统操作再进行导入！")
            # 显示消息框
            msg_box.exec_()

    # 调节框改变检测置信度
    def change_conf_spin_box(self):
        conf_thres = round(self.conf_spin_box.value(), 2)
        self.conf_slider.setValue(int(conf_thres * 100))
        self.identify_api.conf_thres = conf_thres

    # 滚动条改变检测置信度
    def change_conf_slider(self):
        conf_thres = round(self.conf_slider.value() * 0.01, 2)
        self.conf_spin_box.setValue(conf_thres)
        self.identify_api.conf_thres = conf_thres

    # 调节框改变检测交并比
    def change_iou_spin_box(self):
        iou_thres = round(self.iou_spin_box.value(), 2)
        self.iou_slider.setValue(int(iou_thres * 100))
        self.identify_api.iou_thres = iou_thres

    # 滚动条改变检测交并比
    def change_iou_slider(self):
        iou_thres = round(self.iou_slider.value() * 0.01, 2)
        self.iou_spin_box.setValue(iou_thres)
        self.identify_api.iou_thres = iou_thres

    # 图片检测
    def show_image(self):
        if len(self.pt_show.text()) != 0:
            image_path, _ = QFileDialog.getOpenFileName(
                self, "打开图片", "./", "*.jpg;*.png;;All Files(*)")
            if len(image_path) > 4:
                self.input_image = cv2.imread(image_path)
                self.input_image, self.output_image, self.identify_labels = self.identify_api.show_frame(self.input_image,
                                                                                                         False)
                if self.output_image is not None:
                    # ===== 显示界面和结果 ===== #
                    show_input_img = self.change_image(self.input_image, 740, 290)
                    show_output_img = self.change_image(self.output_image, 740, 290)
                    # 将检测图像画面显示在界面
                    show_input_img = cv2.cvtColor(show_input_img, cv2.COLOR_BGR2RGB)
                    show_input_img = QImage(show_input_img.data, show_input_img.shape[1], show_input_img.shape[0],
                                            show_input_img.shape[1] * 3, QImage.Format_RGB888)
                    self.input_img.setPixmap(QPixmap.fromImage(show_input_img))
                    show_output_img = cv2.cvtColor(show_output_img, cv2.COLOR_BGR2RGB)
                    show_output_img = QImage(show_output_img.data, show_output_img.shape[1], show_output_img.shape[0],
                                             show_output_img.shape[1] * 3, QImage.Format_RGB888)
                    self.output_img.setPixmap(QPixmap.fromImage(show_output_img))
                    # 检测结果在结果显示区
                    identify_result = ""
                    counter = Counter(self.identify_labels)  # 使用 Counter 统计元素出现次数
                    for element, count in counter.items():
                        identify_result = identify_result + str(element) + ": " + str(count) + "\n"
                    self.result.setText(identify_result)
                    # ===== 保存结果 ===== #
                    if self.save_button_yes.isChecked():
                        # 保存图像
                        file_time = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
                        file_id = file_time.replace(":", "").replace("-", "").replace(" ", "")
                        file_name = "i" + file_id + ".jpg"
                        file_path = os.path.join(self.save_path, "images/" + file_name)
                        cv2.imwrite(file_path, self.output_image)
                        # 保存记录
                        rows = []
                        with open('A_data/record.csv', 'r', newline='', encoding='utf-8') as csv_file:
                            reader = csv.reader(csv_file)
                            for row in reader:
                                if len(row) == 4:
                                    rows.append(row)
                            rows.append([file_id, self.admin_list[0], file_name, file_time])
                        with open('A_data/record.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            for row in rows:
                                writer.writerow(row)
                        # 创建一个消息框
                        msg_box = QMessageBox()
                        msg_box.setWindowTitle("提示")
                        msg_box.setText("检测记录已保存！")
                        # 显示消息框
                        msg_box.exec_()
                else:
                    self.reset1_1()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先导入模型！")
            # 显示消息框
            msg_box.exec_()

    # 视频检测
    def video_identify(self):
        if len(self.pt_show.text()) != 0:
            if self.btn2.text() == "开启视频检测" and not self.timer_video.isActive():
                video_path, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;*.avi;;All Files(*)")
                if len(video_path) > 5:
                    flag = self.identify_api.cap.open(video_path)
                    if flag is False:
                        QMessageBox.warning(
                            self, u"Warning", u"打开视频失败", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
                    else:
                        self.timer_video.start(30)
                        self.btn1.setDisabled(True)
                        self.btn3.setDisabled(True)
                        self.btn4.setDisabled(True)
                        self.btn2.setText("关闭视频检测")
            else:
                self.identify_api.cap.release()
                self.timer_video.stop()
                self.btn1.setDisabled(False)
                self.btn3.setDisabled(False)
                self.btn4.setDisabled(False)
                self.btn2.setText("开启视频检测")
                self.reset1_1()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先导入模型！")
            # 显示消息框
            msg_box.exec_()

    # 摄像头检测
    def camera_identify(self):
        if len(self.pt_show.text()) != 0:
            if self.btn3.text() == "开启实时检测" and not self.timer_video.isActive():
                # 默认使用第一个本地camera
                flag = self.identify_api.cap.open(0)
                if flag is False:
                    QMessageBox.warning(
                        self, u"Warning", u"打开摄像头失败", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
                else:
                    self.timer_video.start(30)
                    self.btn1.setDisabled(True)
                    self.btn2.setDisabled(True)
                    self.btn3.setText("关闭实时检测")
                    self.btn4.setDisabled(True)
            else:
                self.identify_api.cap.release()
                self.timer_video.stop()
                self.btn1.setDisabled(False)
                self.btn2.setDisabled(False)
                self.btn3.setText("开启实时检测")
                self.btn4.setDisabled(False)
                self.reset1_1()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先导入模型！")
            # 显示消息框
            msg_box.exec_()

    # 展示图像与显示结果(视频与摄像头)
    def show_video(self):
        self.input_image, self.output_image, self.identify_labels = self.identify_api.show_frame(None, True)
        if self.output_image is not None:
            # ===== 保存结果 ===== #
            if self.save_button_yes.isChecked():
                # 保存视频
                if self.save_video_flag is False:
                    self.save_video_flag = True
                    fps = self.identify_api.cap.get(cv2.CAP_PROP_FPS)
                    w = int(self.identify_api.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.identify_api.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    file_time = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
                    file_id = file_time.replace(":", "").replace("-", "").replace(" ", "")
                    if self.btn2.text() == "关闭视频检测":
                        file_name = "v" + file_id + ".mp4"
                        save_path = self.save_path + "/videos/" + file_name
                    else:
                        file_name = "c" + file_id + ".mp4"
                        save_path = self.save_path + "/camera/" + file_name
                    self.output_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 保存记录
                    rows = []
                    with open('A_data/record.csv', 'r', newline='', encoding='utf-8') as csv_file:
                        reader = csv.reader(csv_file)
                        for row in reader:
                            if len(row) == 4:
                                rows.append(row)
                        rows.append([file_id, self.admin_list[0], file_name, file_time])
                    with open('A_data/record.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        for row in rows:
                            writer.writerow(row)
                self.output_video.write(self.output_image)
            # ===== 显示界面和结果 ===== #
            show_input_img = self.change_image(self.input_image, 740, 290)
            show_output_img = self.change_image(self.output_image, 740, 290)
            # 将检测图像画面显示在界面
            show_input_img = cv2.cvtColor(show_input_img, cv2.COLOR_BGR2RGB)
            show_input_img = QImage(show_input_img.data, show_input_img.shape[1], show_input_img.shape[0],
                                    show_input_img.shape[1] * 3, QImage.Format_RGB888)
            self.input_img.setPixmap(QPixmap.fromImage(show_input_img))
            show_output_img = cv2.cvtColor(show_output_img, cv2.COLOR_BGR2RGB)
            show_output_img = QImage(show_output_img.data, show_output_img.shape[1], show_output_img.shape[0],
                                     show_output_img.shape[1] * 3, QImage.Format_RGB888)
            self.output_img.setPixmap(QPixmap.fromImage(show_output_img))
            # 检测结果在结果显示区
            identify_result = ""
            counter = Counter(self.identify_labels)  # 使用 Counter 统计元素出现次数
            for element, count in counter.items():
                identify_result = identify_result + str(element) + ": " + str(count) + "\n"
            self.result.setText(identify_result)
        else:
            self.timer_video.stop()
            self.btn1.setDisabled(False)
            self.btn2.setDisabled(False)
            self.btn3.setDisabled(False)
            self.btn4.setDisabled(False)
            self.btn2.setText("开启视频检测")
            self.btn3.setText("开启实时检测")
            self.reset1_1()

    # 清空重置数据
    def reset1_1(self):
        if self.save_button_yes.isChecked() and self.output_video is not None:
            self.output_video.release()
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("检测记录已保存！")
            # 显示消息框
            msg_box.exec_()
        self.input_image = None  # 输入图像
        self.output_image = None  # 输出图像
        self.output_video = None  # 输出视频
        self.identify_labels = []  # 检测结果
        self.save_video_flag = False  # 保存视频标志位
        self.input_img.clear()  # 清空输入图像显示区
        self.input_img.setText("输入显示区")
        self.output_img.clear()  # 清空输出图像显示区
        self.output_img.setText("输出显示区")
        self.result.clear()  # 清空结果显示区

    def reset1_2(self):
        self.reset1_1()
        self.pt_show.clear()
        self.save_button_no.setChecked(True)

    # 改变图像大小在界面显示
    @staticmethod
    def change_image(input_image, width, height):
        if input_image is not None:
            # 更换为界面适应性大小显示
            wh = float(int(input_image.shape[1]) / int(input_image.shape[0]))
            show_wh = width / height
            if int(input_image.shape[1]) > height or int(input_image.shape[0]) > width:
                if show_wh - wh < 0:
                    w = width
                    h = int(w / wh)
                    output_image = cv2.resize(input_image, (w, h))
                else:
                    h = height
                    w = int(h * wh)
                    output_image = cv2.resize(input_image, (w, h))
            else:
                output_image = input_image
            return output_image
        else:
            return input_image

    # ========== 检测管理相关函数 =============
    # 获取所有记录
    def select_all(self):
        self.identify_record = []
        with open('A_data/record.csv', 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if len(row) == 4 and row[1] == self.admin_list[0]:
                    self.identify_record.append(row)

    # 显示数据
    def show_table(self):
        self.table_view.setRowCount(len(self.identify_record))  # 行
        for i in range(len(self.identify_record)):
            checkbox = QCheckBox()
            checkbox.setStyleSheet("padding-left:5px;")
            self.table_view.setCellWidget(i, 0, checkbox)
            self.table_data(self.table_view, i, 0, self.identify_record[i][0])
            self.table_data(self.table_view, i, 1, self.identify_record[i][2])
            self.table_data(self.table_view, i, 2, self.identify_record[i][3])
            table_btn = QPushButton('查看')
            table_btn.setCursor(Qt.PointingHandCursor)
            table_btn.setStyleSheet(
                "QPushButton{background:#dddddd;color:#2c3a45;font-weight:600;margin:5px;"
                "font-size:15px;font-family:'微软雅黑';border-radius:14px;border: 2px solid #000000;}"
                "QPushButton:hover{background:#e6e6e6;}")
            self.table_view.setCellWidget(i, 3, table_btn)
            table_btn.clicked.connect(lambda _, record_path=self.identify_record[i][2]: self.record_show(record_path))

    # 返回表格填充页面
    @staticmethod
    def table_data(table, i, j, data):
        table.setRowHeight(i, 40)  # 设置高40
        item = QTableWidgetItem()
        table.setItem(i, j, item)
        item = table.item(i, j)
        item.setText(str(data))
        item.setToolTip(str(data))
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 水平垂直居中

    # 查询所有记录
    def select_all_record(self):
        self.date_show.setText("--")
        self.select_all()
        self.show_table()

    # 按日期查询
    def select_record_by_date(self):
        picker = DateWin()
        if picker.exec_() == QDialog.Accepted:
            selected_date = picker.calendar.selectedDate()
            year = str(selected_date.year())
            month = str(selected_date.month())
            day = str(selected_date.day())
            if len(month) == 1:
                month = "0" + month
            if len(day) == 1:
                day = "0" + day
            date_result = year + "-" + month + "-" + day
            record_list = []
            with open('A_data/record.csv', 'r', newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if len(row) == 4 and row[3][0:10] == date_result and row[1] == self.admin_list[0]:
                        record_list.append(row)
            if len(record_list) != 0:
                self.identify_record = record_list
                self.date_show.setText(date_result)
                self.show_table()
            else:
                # 创建一个消息框
                msg_box = QMessageBox()
                msg_box.setWindowTitle("提示")
                msg_box.setText("没有该天记录，查询失败！")
                # 显示消息框
                msg_box.exec_()

    # 批量删除
    def del_record(self):
        select_list = []
        for i in range(self.table_view.rowCount()):
            item = self.table_view.cellWidget(i, 0)
            if item.isChecked():
                select_list.append(self.identify_record[i])
        if len(select_list) == 0:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先选择要删除的记录！")
            msg_box.exec_()
        else:
            select_name = []
            for i in select_list:
                select_name.append(i[2])
                if i[2][0:1] == 'v':
                    os.remove("./A_output/videos/" + i[2])
                elif i[1][0:1] == 'c':
                    os.remove("./A_output/camera/" + i[2])
                else:
                    os.remove("./A_output/images/" + i[2])
            rows = []
            with open('A_data/record.csv', 'r', newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if len(row) == 4 and row[2] not in select_name:
                        rows.append(row)
            with open('A_data/record.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for row in rows:
                    writer.writerow(row)
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("删除成功！")
            msg_box.exec_()
            if self.date_show.text() == "--":
                self.select_all()
            else:
                record_list = []
                with open('A_data/record.csv', 'r', newline='', encoding='utf-8') as csv_file:
                    reader = csv.reader(csv_file)
                    for row in reader:
                        if len(row) == 4 and row[3][0:10] == self.date_show.text() and row[1] == self.admin_list[0]:
                            record_list.append(row)
                    self.identify_record = record_list
            self.show_table()

    # 查看图像或视频
    def record_show(self, path):
        if path[0:1] == 'v':
            self.identify_record_path = "./A_output/videos/" + path
        elif path[0:1] == 'c':
            self.identify_record_path = "./A_output/camera/" + path
        else:
            self.identify_record_path = "./A_output/images/" + path
        if self.identify_record_path[-3:] == "jpg":
            image_win = ImageWin(self)
            image_win.exec_()
        else:
            video_win = VideoWin(self)
            video_win.exec_()

    # 清空重置界面2数据
    def reset2(self):
        self.identify_record = []
        self.date_show.setText("--")
        self.identify_record_path = ""

    # ========== 账号管理相关函数 =============
    def show_admin(self):
        if len(self.admin_list) != 0:
            self.admin_id2.setText(self.admin_list[0])
            self.admin_pwd2.setText(self.admin_list[1])

    def upd_password(self):
        if len(self.admin_pwd2.text()) != 0:
            if self.admin_pwd2.text() != self.admin_list[1]:
                self.admin_list[1] = self.admin_pwd2.text()
                rows = []
                with open('A_data/admin.csv', 'r', newline='', encoding='utf-8') as csv_file:
                    reader = csv.reader(csv_file)
                    for row in reader:
                        if len(row) == 2:
                            rows.append(row)
                for row in rows:
                    if row[0] == self.admin_list[0]:
                        row[1] = self.admin_list[1]
                        break
                with open('A_data/admin.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    for row in rows:
                        writer.writerow(row)
                # 创建一个消息框
                msg_box = QMessageBox()
                msg_box.setWindowTitle("提示")
                msg_box.setText("修改成功!")
                # 显示消息框
                msg_box.exec_()
            else:
                # 创建一个消息框
                msg_box = QMessageBox()
                msg_box.setWindowTitle("提示")
                msg_box.setText("密码与原密码一致!")
                # 显示消息框
                msg_box.exec_()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先输入账号信息!")
            # 显示消息框
            msg_box.exec_()

    # 注销账号
    def reset_admin(self):
        reply = QMessageBox.question(None, '确认', '你确定要执行这个操作吗？', QMessageBox.Yes | QMessageBox.No)
        # 处理用户的回答
        if reply == QMessageBox.Yes:
            rows = []
            with open('A_data/admin.csv', 'r', newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if len(row) == 2:
                        if row[0] != self.admin_list[0]:
                            rows.append(row)
            with open('A_data/admin.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for row in rows:
                    writer.writerow(row)
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("注销成功，程序退出!")
            # 显示消息框
            msg_box.exec_()
            self.close()


# 日期选择界面类
class DateWin(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("日期查询")
        self.selected_date = ""
        self.calendar = QCalendarWidget()
        self.button = QPushButton('选择日期')
        self.button.clicked.connect(self.accept)
        # 创建布局并添加部件
        layout = QVBoxLayout()
        layout.addWidget(self.calendar)
        layout.addWidget(self.button)
        self.setLayout(layout)


# 查看图像界面类
class ImageWin(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像")
        self.resize(1000, 600)
        self.show = QLabel(self)
        self.show.setText("")
        self.show.setFixedSize(900, 500)
        self.show.move(50, 50)
        self.show.setAlignment(Qt.AlignCenter)
        self.show.setStyleSheet("QLabel{border: 2px solid gray;font-size:30px;font-family:'黑体';color:#999999;}")
        input_image = cv2.imread(parent.identify_record_path)
        show_input_img = parent.change_image(input_image, 900, 500)
        # 将检测图像画面显示在界面
        show_input_img = cv2.cvtColor(show_input_img, cv2.COLOR_BGR2RGB)
        show_input_img = QImage(show_input_img.data, show_input_img.shape[1], show_input_img.shape[0],
                                show_input_img.shape[1] * 3, QImage.Format_RGB888)
        self.show.setPixmap(QPixmap.fromImage(show_input_img))


class VideoWin(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture()
        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.show_video)
        self.path = parent.identify_record_path
        self.setWindowTitle("视频")
        self.resize(1000, 600)
        self.show = QLabel(self)
        self.show.setText("显示区")
        self.show.setFixedSize(900, 500)
        self.show.move(50, 20)
        self.show.setAlignment(Qt.AlignCenter)
        self.show.setStyleSheet("QLabel{border: 2px solid gray;font-size:30px;font-family:'黑体';color:#999999;}")
        self.play_button = QPushButton(self)
        self.play_button.setFixedSize(440, 50)
        self.play_button.move(50, 530)
        self.play_button.setText("播放")
        self.play_button.clicked.connect(self.play)
        self.play_button.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                       "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                       "QPushButton:hover{background:#e6e6e6;}")
        self.stop_button = QPushButton(self)
        self.stop_button.setText("关闭")
        self.stop_button.setFixedSize(440, 50)
        self.stop_button.move(510, 530)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setStyleSheet("QPushButton{background:#ffffff;color:#999999;font-weight:600;"
                                       "font-size:18px;font-family:'黑体';border: 2px solid gray;}"
                                       "QPushButton:hover{background:#e6e6e6;}")

    def play(self):
        if not self.timer_video.isActive():
            self.cap.open(self.path)
            self.timer_video.start(30)

    def stop(self):
        self.timer_video.stop()
        self.show.clear()
        self.show.setText("显示区")

    def show_video(self):
        flag, image = self.cap.read()
        if image is not None:
            show_input_img = self.change_image(image, 900, 500)
            # 将检测图像画面显示在界面
            show_input_img = cv2.cvtColor(show_input_img, cv2.COLOR_BGR2RGB)
            show_input_img = QImage(show_input_img.data, show_input_img.shape[1], show_input_img.shape[0],
                                    show_input_img.shape[1] * 3, QImage.Format_RGB888)
            self.show.setPixmap(QPixmap.fromImage(show_input_img))
        else:
            self.stop()

    # 改变图像大小在界面显示
    @staticmethod
    def change_image(input_image, width, height):
        if input_image is not None:
            # 更换为界面适应性大小显示
            wh = float(int(input_image.shape[1]) / int(input_image.shape[0]))
            show_wh = width / height
            if int(input_image.shape[1]) > height or int(input_image.shape[0]) > width:
                if show_wh - wh < 0:
                    w = width
                    h = int(w / wh)
                    output_image = cv2.resize(input_image, (w, h))
                else:
                    h = height
                    w = int(h * wh)
                    output_image = cv2.resize(input_image, (w, h))
            else:
                output_image = input_image
            return output_image
        else:
            return input_image

    def closeEvent(self, event):
        # 在窗口关闭时执行的操作
        self.stop()
        event.accept()



