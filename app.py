from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import os

app = Flask(__name__)

# 图片处理函数，可以自定义
def process_image(image):
    # 示例：将图片转为灰度图像
    grayscale_image = image.convert('L')
    return grayscale_image

# 接收图片并返回处理后的结果
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

        # 处理图片
        processed_image = process_image(image)

        # 将处理后的图片保存到内存中
        img_io = io.BytesIO()
        processed_image.save(img_io, 'JPEG')
        img_io.seek(0)

        # 返回处理后的图片（这里返回的是图片的字节流，前端需要处理显示）
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)