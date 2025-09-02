import torch
model = torch.load('D:\\JM\\毕业设计\\农业害虫检测系统\\基于yolov11的农业害虫检测系统\\yolo12-litchi\\42_demo\\runs\\yolo12n_pretrained_6\\train\\weights\\best.pt',weights_only=False)
print("model:")
print(model)