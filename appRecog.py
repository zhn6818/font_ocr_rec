from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from PIL import Image

from inference_detect import process_image, initialize_readers, calculate_bbox

app = Flask(__name__)

detector, reader, font_recognizer = initialize_readers()


# 图像推理接口
@app.route('/predict', methods=['POST'])
def predict():
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


        # 获取推理结果
        results = process_image(detector, reader, font_recognizer, image_np)

        ocr_results = []
        for result in results:
            box = result['boxes']
            x, y, w, h = calculate_bbox(box)
            # OCR识别的文本和置信度
            text = result["text"]
            confidence = float(result["confident"])
            font = result['font']
            ocr_results.append({'bbox': {'x': x, 'y': y, 'w': w, 'h': h}, 'text': text, 'confidence': confidence, 'font': font})

        # 返回JSON格式的结果
        return jsonify({'status': 'success', 'data': ocr_results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)