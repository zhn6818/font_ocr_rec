from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
from PIL import Image
import asyncio
import logging
from aiohttp import ClientSession
import json

from inference_detect import process_image, initialize_readers, calculate_bbox

app = Flask(__name__)

CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

detector, reader, font_recognizer = initialize_readers()

logging.basicConfig(level=logging.DEBUG)
# 异步处理图像推理接口
@app.route('/predict', methods=['POST'])
async def predict():
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
        logging.info("Starting OCR processing...")
        # 使用异步处理OCR推理
        results = await process_image_async(detector, reader, font_recognizer, image_np)
        logging.info("OCR processing finished.")

        ocr_results = []
        for result in results:
            box = result['boxes']
            x, y, w, h = calculate_bbox(box)
            # OCR识别的文本和置信度
            text = result["text"]
            confidence = float(result["confident"])
            font = result['font']
            ocr_results.append({"bbox": {'x': x, 'y': y, 'w': w, 'h': h}, "text": text, "confidence": confidence, "font": font})


        # 将数据转换为 JSON 字符串，确保双引号
        json_output = json.dumps({"status": "success", "data": ocr_results}, ensure_ascii=False)
        return json_output

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 异步处理OCR推理函数
async def process_image_async(detector, reader, font_recognizer, image_np):
    loop = asyncio.get_event_loop()
    # 使用线程池将阻塞任务移到后台执行
    results = await loop.run_in_executor(None, process_image, detector, reader, font_recognizer, image_np)
    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=False, use_reloader=False)  # use_reloader=False to avoid asyncio event loop issues