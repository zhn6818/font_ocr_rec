import sys
import os
from PIL import Image
import cv2

sys.path.append("./EasyOCR/")

from easyocr.easyocr import Reader

languages = ['ch_sim', 'en']

# 初始化Reader对象
reader = Reader(languages)

# 图片路径（也可以使用图像的numpy数组）
image_path = 'test.png'

# 使用readtext方法进行OCR识别
results = reader.readtext(image_path)

# 读取图片
image = cv2.imread(image_path)

# 创建一个名为 'img' 的文件夹，如果它不存在
output_dir = './img'
os.makedirs(output_dir, exist_ok=True)
print(f"文件夹已创建：{output_dir}")

# 循环输出识别结果，并根据rect坐标分割汉字区域
for idx, result in enumerate(results):
    bbox, text, confidence = result
    print(f"识别到的文字: {text}, 置信度: {confidence}")

    # 获取rect区域的坐标
    x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
    x_max, y_max = int(bbox[2][0]), int(bbox[2][1])

    # 裁剪出该区域
    cropped_image = image[y_min:y_max, x_min:x_max]

    # 定义文件的路径，将分割的图像保存在 'img' 文件夹中
    cropped_image_path = os.path.join(output_dir, f'cropped_{idx}.png')
    
    # 保存分割的汉字区域到 'img' 文件夹
    cv2.imwrite(cropped_image_path, cropped_image)
    print(f"保存分割的汉字区域: {cropped_image_path}")
