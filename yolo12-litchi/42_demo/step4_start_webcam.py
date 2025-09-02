#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77 
@File    ：start_single_detect.py
@IDE     ：PyCharm 
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/8/15 15:15 
'''
import cv2

from ultralytics import YOLO
from cv2 import getTickCount, getTickFrequency
# Load the YOLOv8 model
# model = YOLO("yolov8n.pt")


# model = YOLO("yolov8n.onnx", task="detect")
model = YOLO("yolov8n.pt", task="detect")
# model = YOLO("yolov8n.engine", task="detect")
# model = YOLO("yolov8n_ncnn_model", task="detect")

# Open the video file
video_path = "images/resources/output1.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    loop_start = getTickCount()
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        loop_time = getTickCount() - loop_start
        total_time = loop_time / (getTickFrequency())
        FPS = int(1 / total_time)
        # 在图像左上角添加FPS文本
        fps_text = f"FPS: {FPS:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 255)  # 红色
        text_position = (10, 30)  # 左上角位置

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, fps_text, text_position, font, font_scale, text_color, font_thickness)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()