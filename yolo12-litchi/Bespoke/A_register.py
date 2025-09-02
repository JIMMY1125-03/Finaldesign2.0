from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import A_windows
import csv

from CONFIG import *
class RegisterUi(QMainWindow):
    def __init__(self):
        # 鼠标拖动界面设置参数
        self.press_x = 0
        self.press_y = 0
        # 页面生成
        super(RegisterUi, self).__init__()
        self.resize(600, 400)
        self.setWindowFlag(Qt.FramelessWindowHint)  # 隐藏边框
        self.setStyleSheet("QMainWindow{border: 1px solid #303030;}")

        self.label = QLabel(self)
        self.label.setText(REGISTER_TILE)
        self.label.setFixedSize(450, 50)
        self.label.move(0, 0)
        self.label.setStyleSheet("QLabel{background:#303030;}QLabel{color:#ffffff;border:none;font-weight:600;"
                                 "font-size:16px;font-family:'微软雅黑';padding-left:30px}")

        self.exit = QPushButton(self)
        self.exit.setText("☀ 退出系统")
        self.exit.resize(150, 50)
        self.exit.move(450, 0)
        self.exit.setStyleSheet("QPushButton{background:#303030;text-align:center;border:none;font-weight:600;"
                                "color:#909090;font-size:14px;}")
        self.exit.setCursor(Qt.PointingHandCursor)
        self.exit.clicked.connect(self.close)

        self.title = QLabel(self)
        self.title.setText("请输入注册信息")
        self.title.setFixedSize(600, 40)
        self.title.move(0, 70)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("QLabel{color:#303030;font-weight:600;font-size:30px;font-family:'微软雅黑'; }")

        self.user_id1 = QLabel(self)
        self.user_id1.setText("账号")
        self.user_id1.setFixedSize(60, 40)
        self.user_id1.move(100, 140)
        self.user_id1.setAlignment(Qt.AlignCenter)
        self.user_id1.setStyleSheet(
            " QLabel{color:#303030;border:none;font-weight:600;font-size:25px;font-family:'微软雅黑'; border: 2px solid #000000;}")

        self.user_id2 = QLineEdit(self)
        self.user_id2.setEchoMode(QLineEdit.Normal)  # 正常显示字符
        self.user_id2.setPlaceholderText("请输入账号...")
        self.user_id2.setFixedSize(342, 40)
        self.user_id2.move(158, 140)
        self.user_id2.setStyleSheet(
            " QLineEdit{padding-left:10px;color:#303030;font-weight:600;font-size:18px;font-family:'微软雅黑'; border: 2px solid #000000;}")

        self.user_pwd1 = QLabel(self)
        self.user_pwd1.setText("密码")
        self.user_pwd1.setFixedSize(60, 40)
        self.user_pwd1.move(100, 200)
        self.user_pwd1.setAlignment(Qt.AlignCenter)
        self.user_pwd1.setStyleSheet(
             " QLabel{color:#303030;border:none;font-weight:600;font-size:25px;font-family:'微软雅黑'; border: 2px solid #000000;}")

        self.user_pwd2 = QLineEdit(self)
        self.user_pwd2.setEchoMode(QLineEdit.Password)  # 显示密码字符
        self.user_pwd2.setPlaceholderText("请输入密码...")
        self.user_pwd2.setFixedSize(342, 40)
        self.user_pwd2.move(158, 200)
        self.user_pwd2.setStyleSheet(
            " QLineEdit{padding-left:10px;color:#303030;font-weight:600;font-size:18px;font-family:'微软雅黑'; border: 2px solid #000000;}")

        self.user_pwd3 = QLabel(self)
        self.user_pwd3.setText("确认密码")
        self.user_pwd3.setFixedSize(120, 40)
        self.user_pwd3.move(100, 260)
        self.user_pwd3.setAlignment(Qt.AlignCenter)
        self.user_pwd3.setStyleSheet(
            " QLabel{color:#303030;border:none;font-weight:600;font-size:25px;font-family:'微软雅黑'; border: 2px solid #000000;}")

        self.user_pwd4 = QLineEdit(self)
        self.user_pwd4.setEchoMode(QLineEdit.Password)  # 显示密码字符
        self.user_pwd4.setPlaceholderText("请再次输入密码...")
        self.user_pwd4.setFixedSize(282, 40)
        self.user_pwd4.move(218, 260)
        self.user_pwd4.setStyleSheet(
            " QLineEdit{padding-left:10px;color:#303030;font-weight:600;font-size:18px;font-family:'微软雅黑'; border: 2px solid #000000;}")

        self.Enter_btn = QPushButton(self)
        self.Enter_btn.setText("确定")
        self.Enter_btn.resize(190, 50)
        self.Enter_btn.move(100, 320)
        self.Enter_btn.setStyleSheet("QPushButton{background:#303030;text-align:center;border:none;font-weight:600;"
                                     "color:#B0B0B0;font-size:25px;border-radius: 25px;}")
        self.Enter_btn.setCursor(Qt.PointingHandCursor)
        self.Enter_btn.clicked.connect(self.Register)

        self.Return_btn = QPushButton(self)
        self.Return_btn.setText("返回")
        self.Return_btn.resize(190, 50)
        self.Return_btn.move(310, 320)
        self.Return_btn.setStyleSheet("QPushButton{background:#303030;text-align:center;border:none;font-weight:600;"
                                      "color:#B0B0B0;font-size:25px;border-radius: 25px;}")
        self.Return_btn.setCursor(Qt.PointingHandCursor)
        self.Return_btn.clicked.connect(self.toLogin)

    # ================================ 实现鼠标长按移动窗口功能 ================================ #
    def mousePressEvent(self, event):
        self.press_x = event.x()  # 记录鼠标按下的时候的坐标
        self.press_y = event.y()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()  # 获取移动后的坐标
        if 0 < x < 450 and 0 < y < 50:
            move_x = x - self.press_x
            move_y = y - self.press_y  # 计算移动了多少
            position_x = self.frameGeometry().x() + move_x
            position_y = self.frameGeometry().y() + move_y  # 计算移动后主窗口在桌面的位置
            self.move(position_x, position_y)  # 移动主窗口

    # ================================ 按钮功能函数 ================================ #
    # 注册
    def Register(self):
        admin_id = self.user_id2.text()
        admin_pwd1 = self.user_pwd2.text()
        admin_pwd2 = self.user_pwd4.text()
        if len(admin_id) != 0 and len(admin_pwd1) != 0 and len(admin_pwd2) != 0:
            admin_ids = []
            with open('A_data/admin.csv', 'r', newline='', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    if len(row) == 2:
                        admin_ids.append(row[0])
            if admin_id not in admin_ids:
                if admin_pwd1 == admin_pwd2:
                    # 插入账户数据
                    new_row = [admin_id, admin_pwd1]
                    with open('A_data/admin.csv', 'r') as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                    rows.append(new_row)
                    with open('A_data/admin.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(rows)
                    # 创建一个消息框
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("提示")
                    msg_box.setText("注册成功，返回登录页面!")
                    # 显示消息框
                    msg_box.exec_()
                    self.hide()
                    A_windows.Windows.login.show()
                else:
                    # 创建一个消息框
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("提示")
                    msg_box.setText("两次密码不一致，请重试!")
                    # 显示消息框
                    msg_box.exec_()
            else:
                # 创建一个消息框
                msg_box = QMessageBox()
                msg_box.setWindowTitle("提示")
                msg_box.setText("该账号已被注册，请重试!")
                # 显示消息框
                msg_box.exec_()
        else:
            # 创建一个消息框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("提示")
            msg_box.setText("请输入完整信息进行注册!")
            # 显示消息框
            msg_box.exec_()

    # 返回登录页面
    def toLogin(self):
        self.user_id2.setText("")
        self.user_pwd2.setText("")
        self.user_pwd4.setText("")
        self.hide()
        A_windows.Windows.login.show()