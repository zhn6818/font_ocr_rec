import cv2
import numpy as np
import sys
import os
from PIL import Image
from PIL import ImageDraw, ImageFont
from collections import Counter

sys.path.append("./EasyOCR/")
from easyocr.easyocr import ReaderDetect, ReaderRecog
sys.path.append("./font_detect/")
from detectFont import FontDetectionInterface


def crop_image_aspect_ratio(font_recognizer, image, aspect_ratio=5):
    """根据给定的宽高比裁剪图像，保持宽高比为5:1"""
    img_height, img_width = image.shape[:2]
    
    crop_height = img_height
    crop_width = int(crop_height * aspect_ratio)
    
    if crop_width > img_width:
        crop_width = img_width
    
    crops = []
    font_counts = Counter()  # 用于统计字体出现次数
    
    for start_x in range(0, img_width, crop_width):
        end_x = start_x + crop_width
        crop = image[:, start_x:end_x]
        if crop.shape[1] > 0:
            crops.append(crop)
            # 将 BGR 格式的 OpenCV 图像转换为 RGB 格式
            rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # 将 NumPy 数组转换为 PIL 图像
            pil_image = Image.fromarray(rgb_image)
            out = font_recognizer.recognize_font(pil_image)
            # 更新字体统计
            if isinstance(out, dict):
                for font, confidence in out.items():
                    font_counts[font] += 1  # 统计字体出现次数
            # print(out)
    # 找到出现次数最多的字体类别
    most_common_font = font_counts.most_common(1)  # 获取出现次数最多的字体
    if most_common_font:
        most_common_font_name, count = most_common_font[0]
        result = {
            "most_common_font": most_common_font_name,
            "count": count,
            "all_font_counts": font_counts
        }
    else:
        result = {
            "most_common_font": None,
            "count": 0,
            "all_font_counts": font_counts
        }

    return result

def initialize_readers():
    """初始化检测和识别对象"""
    detector = ReaderDetect(gpu=False, detect_network="craft", verbose=True)
    languages = ['ch_sim', 'en']
    reader = ReaderRecog(languages, gpu=True)
    font_recognizer = FontDetectionInterface()
    return detector, reader, font_recognizer

def process_image(detector, reader, font_recognizer, image_path):
    """处理单张图片"""
    horizontal_boxes, free_boxes = detector.detect_img(image_path, text_threshold=0.7, low_text=0.4, link_threshold=0.4)

    img = cv2.imread(image_path)
    maximum_y, maximum_x, _ = img.shape

    all_results = []

    # 绘制检测到的区域
    for box in horizontal_boxes[0]:
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        img_crop = img[y_min:y_max, x_min:x_max]
        result_font = crop_image_aspect_ratio(font_recognizer, img_crop)
        # 使用 split 方法分割字符串
        font_string = result_font['most_common_font']
        font_name = font_string.split('/')[-1]  # 获取最后一部分
        font_name = font_name.split('_')[0]      # 获取“仿宋”部分

        result_crop = reader.read_fulltext(img_crop, output_format='dict')
        # print(result_crop)
        for result in result_crop:
            result['font'] = font_name
            for point in result['boxes']:
                point[0] += x_min  # x坐标平移
                point[1] += y_min  # y坐标平移

        all_results.extend(result_crop)

    # 将 OpenCV 图像转换为 PIL 图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = "SimHei.ttf"  # 请确保该字体文件存在于指定路径
    font = ImageFont.truetype(font_path, 20)

    # 在图像上绘制文字识别结果
    for result in all_results:
        text = result['text']
        confidence = result['confident']
        box = result['boxes']
        font_line = result['font']
        x_min, y_min = int(box[0][0]), int(box[0][1])

        label = f" ({font_line}) {text} ({confidence:.2f})"
        draw.text((x_min, y_min - 10), label, font=font, fill=(255, 0, 0))

    img_pil_path = f'result_{os.path.basename(image_path)}'
    img_pil.save(img_pil_path)
    print(f"检测结果已保存至 {img_pil_path}")
    print(all_results)

def main():
    detector, reader, font_recognizer = initialize_readers()
    image_paths = ['test1.png', 'test2.png', 'test3.png']  # 替换为你的图片路径列表

    for image_path in image_paths:
        process_image(detector, reader, font_recognizer, image_path)

if __name__ == "__main__":
    main()