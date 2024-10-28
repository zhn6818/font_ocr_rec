from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import sys
import numpy as np

# 将 EasyOCR 的路径添加到系统路径中
sys.path.append("./EasyOCR/")
from easyocr.easyocr import Reader

app = Flask(__name__)

# 初始化EasyOCR的Reader对象
languages = ['ch_sim', 'en']
reader = Reader(languages)

# 接收图片并返回OCR识别结果（包括文字区域）
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 打开上传的图片
        image = Image.open(file.stream)

        # 将图像转换为numpy数组以进行OCR处理
        image_np = np.array(image)

        # 使用EasyOCR的readtext方法进行OCR识别
        results = reader.readtext(image_np)

        # 将识别结果格式化为JSON可返回的形式
        ocr_results = []
        for result in results:
            # 获取原始的四点坐标
            bbox = result[0]
            # 计算 (x, y, w, h) 并转换为int类型
            x = int(min(point[0] for point in bbox))
            y = int(min(point[1] for point in bbox))
            w = int(max(point[0] for point in bbox) - x)
            h = int(max(point[1] for point in bbox) - y)
            # OCR识别的文本和置信度
            text = result[1]
            confidence = float(result[2])
            # 添加结果
            ocr_results.append({'bbox': {'x': x, 'y': y, 'w': w, 'h': h}, 'text': text, 'confidence': confidence})

        # 返回OCR识别结果，包括转换后的bbox信息
        return jsonify({'ocr_results': ocr_results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)