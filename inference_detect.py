import cv2
import numpy as np
import sys
import os
from PIL import Image


sys.path.append("./EasyOCR/")

# 导入修改后的 ReaderRecog 类
from easyocr.easyocr import ReaderDetect, ReaderRecog

# 初始化 ReaderDetect 对象
detector = ReaderDetect(gpu=False, detect_network="craft", verbose=True)

languages = ['ch_sim', 'en']

# 创建 ReaderRecog 实例
reader = ReaderRecog(languages, gpu=True)

# 测试图片路径
image_path = 'test.png'  # 替换为你的图片路径

horizontal_boxes, free_boxes = detector.detect_img(image_path, text_threshold=0.7, low_text=0.4, link_threshold=0.4)

def crop_image_aspect_ratio(image, aspect_ratio=5):
    """根据给定的宽高比裁剪图像，保持宽高比为5:1"""
    img_height, img_width = image.shape[:2]
    
    # 每个小图的宽度和高度
    crop_height = img_height
    crop_width = int(crop_height * aspect_ratio)
    
    # 如果宽度超过原始图像宽度，则进行调整
    if crop_width > img_width:
        crop_width = img_width
    
    crops = []
    
    # 从图像中裁剪出多个5:1比例的图像
    for start_x in range(0, img_width, crop_width):
        end_x = start_x + crop_width
        crop = image[:, start_x:end_x]
        if crop.shape[1] > 0:
            crops.append(crop)
    
    return crops

# 加载图片
img = cv2.imread(image_path)
maximum_y,maximum_x,_ = img.shape

all_results = []

# 绘制检测到的区域
for box in horizontal_boxes[0]:
    # 获取rect区域的坐标
    x_min = max(0,box[0])
    x_max = min(box[1],maximum_x)
    y_min = max(0,box[2])
    y_max = min(box[3],maximum_y)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    img_crop = img[y_min:y_max, x_min:x_max]
    cropped_images = crop_image_aspect_ratio(img_crop)

    result_crop = reader.read_fulltext(img_crop, output_format='dict')

    all_results.extend(result_crop)


print(all_results)

# 保存检测结果
result_path = 'result.png'
cv2.imwrite(result_path, img)
print(f"检测结果已保存至 {result_path}")
