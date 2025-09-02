import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from  mail_tool import send_mail
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from CONFIG import *

class Identify:
    def __init__(self):
        self.cap = cv2.VideoCapture()
        # 模型相关
        self.device = ''  # 使用的设备，可以是CUDA设备的 ID（例如 0、0,1,2,3）或者是 'cpu'，默认为 '0'
        self.conf_thres = 0.25  # 识别置信度阈值
        self.iou_thres = 0.25  # 交并比IoU阈值
        self.img_size = 640  # 预测时网络输入图片的尺寸，默认值为 [640]
        self.classes = None  # 指定要检测的目标类别，默认为None，表示检测所有类别
        self.max_det = 1000  # 每张图像的最大检测框数，默认为1000。
        self.agnostic_nms = False  # 是否使用类别无关的非极大值抑制，默认为False
        self.line_thickness = 3  # 检测框的线条宽度，默认为3
        self.augment = False  # 是否使用数据增强进行推理
        self.visualize = False  # 是否可视化模型中的特征图，默认为False
        self.dnn = False  # 是否使用 OpenCV DNN 进行 ONNX 推理
        self.half = False  # 是否使用 FP16 半精度进行推理
        self.model = None
        self.stride = None
        self.names = None
        self.pt = None

        self.is_send = 0
        self.send_gap = 30
        self.send_flag = 0

    # 加载模型
    def load_pt(self, weights):
        try:
            # 清除已加载的模型
            torch.cuda.empty_cache()
            # 加载新模型
            # device = select_device(self.device)
            # self.model = YOLO("yolo11n.pt")  # load an official model
            self.model = YOLO(weights)  # load an official model
            self.names = self.model.names
            return True
        except Exception:
            return False

    # YOLOv5检测核心函数
    # 输入：输入图像，摄像头开启标识
    # 输出：输入图像, 输出图像，检测内容列表
    def show_frame(self, image, cap_flag):
        if cap_flag:
            flag, image = self.cap.read()
            self.send_gap = 30
        else:
            self.send_gap = 0
        if image is not None:
            img = image.copy()
            show_img = img
            labels = []
            results = self.model.predict(img, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres)
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                for box in boxes:
                    c = int(box.cls)  # integer class
                    labels.append(self.names[c])  # 获取识别类别

            if "fire" in labels:
                self.is_send = self.is_send + 1
                self.send_flag = 0
            # if "fall" in labels:
            #     self.is_send = self.is_send + 1
            #     self.send_flag = 1
            # if "no_helmet" in labels:
            #     self.is_send = self.is_send + 1
            #     self.send_flag = 2
            if self.is_send > self.send_gap:
                print("发送短信")
                if self.send_flag == 0:
                    send_mail(user_mail=USER_MAIL, info='当前位置发现火情，请及时查看')

                self.is_send = 0
            show_img = results[0].plot()
                # masks = result.masks  # Masks object for segmentation masks outputs
                # keypoints = result.keypoints  # Keypoints object for pose outputs
                # probs = result.probs  # Probs object for classification outputs
                # obb = result.obb  # Oriented boxes object for OBB outputs
                # result.show()  # display to screen
                # result.save(filename="result.jpg")  # save to disk
            # with torch.no_grad():
            #     # 图像转换
            #     img = letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]  # padded resize
            #     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            #     img = np.ascontiguousarray(img)  # contiguous
            #     dt = (Profile(), Profile(), Profile())
            #     with dt[0]:
            #         img = torch.from_numpy(img).to(self.model.device)
            #         img = img.half() if self.half else img.float()  # uint8 to fp16/32
            #         img /= 255.0  # 0 - 255 to 0.0 - 1.0
            #         if len(img.shape) == 3:
            #             img = img[None]  # expand for batch dim
            #     # Inference
            #     with dt[1]:
            #         pred = self.model(img, augment=self.augment, visualize=self.visualize)
            #     # NMS
            #     with dt[2]:
            #         pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
            #                                    self.classes, self.agnostic_nms, max_det=self.max_det)
            #         # Process detectionss
            #     annotator = Annotator(show_img, line_width=self.line_thickness, example=str(self.names))
            #     for i, det in enumerate(pred):  # detections per image
            #         if det is not None and len(det):
            #             # Rescale boxes from img_size to im0 size
            #             det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], show_img.shape).round()
            #             # Write results
            #             for *xyxy, conf, cls in reversed(det):
            #                 c = int(cls)  # integer class
            #                 label = f'{self.names[c]} {conf:.2f}'
            #                 labels.append(self.names[c])  # 获取识别类别
            #                 annotator.box_label(xyxy, label, color=colors(c, True))
            #     show_img = annotator.result()
            # 返回原始图像，需要进行测试的图像，以及标签（名称格式的标签）
            return image, show_img, labels
        else:
            return image, None, None