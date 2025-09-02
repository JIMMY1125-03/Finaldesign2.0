import io
import json
import numpy as np
from PIL import Image
import torch
import cv2
import requests
from flask import Flask, request, jsonify
import base64

app = Flask(__name__)

model = None
class_names = None

DOUBAO_API_KEY = "6c84373e-ed7b-4534-aeac-d61a32e71a76" #豆包API_KEY
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DOUBAO_MODEL = "doubao-seed-1-6-thinking-250715"#模型版本

def load_model(model_path, classes_path):
    """加载YOLO模型和类别名称"""
    global model, class_names
    try:
        model = torch.hub.load('D:/JM/毕业设计/农业害虫检测系统/基于yolov11的农业害虫检测系统/yolo12-litchi/42_demo/runs/yolo12n_pretrained_1/train/weights/best.pt','custom',path=model_path)
        model.eval()
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"YOLO模型加载成功，共{len(class_names)}个类别")
    except Exception as e:
        print(f"模型加载失败:{str(e)}")
        raise e

def call_doubao_analysis(detection_data, image_base64=None):
    """调用豆包大模型进行智能分析"""
    #构建分析提示词
    prompt = f"""
    你是一个专业的计算机视觉分析专家。请分析一下的目标检测结果：
    【检测结果统计】：
     - 总检测对象数：{detection_data['basic_stats']['total_object']}
     - 平均置信度：{detection_data['confidence_analysis']['average_confidence']:.3f}
     - 检测类别分布：{json.dumps(detection_data['class_analysis']['distribution'], ensure_ascii=False)}
     【详细检测信息】：
      {json.dumps(detection_data['detailed_detections'], ensure_ascii=False, indent=2)}
     请从以下角度进行专业分析：
     1.场景理解：这是什么场景？检测结果是否合理？
     2.质量评估：检测质量如何？置信度分布是否可靠？
     3.异常识别：是否有异常或值得注意的情况？
     4.建议改进：对检测结果有什么改进建议？
     5.应用建议：这个检测结果可以用于什么实际应用？
     
     请用中文回复，保持专业但易于理解。
     """

    #构建请求数据
    messages = [
        {"role" : "system", "content" : "你是一个专业的计算机视觉分析专家，擅长分析目标检测结果并提供深入的见解。"},
        {"role":"user","content":prompt}
    ]
    #如果有图像，可以添加到消息中（如果需要）
    if image_base64:
        messages.append({
            "role":"user",
            "content" : f"这是检测的原图:[图像数据已省略]"
        })
    payload = {
        "model" : DOUBAO_MODEL,
        "messages" : messages,
        "temperature" : 0.3,
        "max_tokens" : 2000
    }
    headers = {
        "Authorization" : f"Bearer{DOUBAO_API_KEY}",
        "Content-Type" : "application/json"
    }

    try:
        response = requests.post(DOUBAO_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"豆包API调用失败：{str(e)}")
        return f"分析服务暂不可用：{str(e)}"

def analyze_detections_with_doubao(detections, image_shape):
    """综合分析并调用豆包api"""

    #基础统计分析
    basic_stats = {
        'total_objects' : len(detections),
        'image_width':image_shape[1],
        'image_height':image_shape[0]
    }

    #置信度分析
    if len(detections) > 0:
        confidences = [d['confidence'] for d in detections]
        confidence_analysis = {
            'average_confidence' : round(np.mean(confidences), 3),
            'max_confidence': round(max(confidences), 3),
            'min_confidence': round(min(confidences), 3),
            'high_confidence_count': sum(1 for c in confidences if c > 0.7)
        }
    else:
        confidence_analysis = {'average_confidence':0}

    class_distribution = {}
    for detection in detections:
        class_id = int(detection['class'])
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
    #详细检测信息
    detailed_detections = []
    for detection in detections:
        class_id = int(detection['class'])
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else str (class_id)

        detailed_detections.append({
            'class':class_name,
            'confidence' : round(detection['confidence'], 3),
            'bbox':{
                'xmin':int(detection['xmin']),
                'ymin':int(detection['ymin']),
                'xmax':int(detection['xmax']),
                'ymax':int(detection['ymax'])
            },
            'area':(detection['xmax'] - detection['xmin']) * (detection['ymax'] - detection['ymain'])
        })
    analysis_data = {
        'basic_stats': basic_stats,
        'confidence_analysis': confidence_analysis,
        'class_analysis':{
            'distribution':class_distribution,
            'unique_classes':len(class_distribution)
        },
        'detailed_detections':detailed_detections
    }

    #调用豆包大模型进行深度分析
    ai_analysis = call_doubao_analysis(analysis_data)

    return{
        'basic_analysis':analysis_data,
        'ai_analysis':ai_analysis
    }

@app.route('/analuze', methods=['POST'])
def analyze_with_ai():
    """主分析接口：YOLO检测+豆包分析"""
    if 'file' not in request.files:
        return jsonify({'error':'未提供图像文件'}),400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':'未选择文件'}), 400

    try:
        #读取和处理图像
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        #YOLO检测
        results = model(opencv_image)
        predictions = results.pandas().xyxy[0]
        detections = predictions.to_dict('record')

        #豆包智能分析
        analysis_result = analyze_detections_with_doubao(detections, opencv_image.shape)

        return jsonify({
            'status':'success',
            'detection_count':len(detections),
            'basic_analysis':analysis_result['basic_analysis'],
            'ai_analysis':analysis_result['ai_analysis'],
            'raw_detections':detections[:10] #只返回前10个检测结果避免响应过大
        })
    except Exception as e:
        return jsonify({'error':f'处理失败：{str(e)}'}), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status':'healthy',
            'model_loaded':model is not None,
            'doubao_configured':DOUBAO_API_KEY !="你的豆包API_Key"
        })

    if __name__ == '__main__':
        load_model("model.pt","calsses.txt")
        app.run(host='0.0.0.0', port=5000, debug=True)

