import cv2
import numpy as np
import sys
import os
from PIL import Image
from PIL import ImageDraw, ImageFont

sys.path.append("./EasyOCR/")

# 导入修改后的 ReaderRecog 类
from easyocr.easyocr import ReaderDetect, ReaderRecog

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

def main():
    # 初始化 ReaderDetect 对象
    detector = ReaderDetect(gpu=False, detect_network="craft", verbose=True)

    languages = ['ch_sim', 'en']

    # 创建 ReaderRecog 实例
    reader = ReaderRecog(languages, gpu=True)

    # 测试图片路径
    image_path = 'test3.png'  # 替换为你的图片路径

    horizontal_boxes, free_boxes = detector.detect_img(image_path, text_threshold=0.7, low_text=0.4, link_threshold=0.4)

    # 加载图片
    img = cv2.imread(image_path)
    maximum_y, maximum_x, _ = img.shape

    all_results = []

    # 绘制检测到的区域
    for box in horizontal_boxes[0]:
        # 获取rect区域的坐标
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        img_crop = img[y_min:y_max, x_min:x_max]
        cropped_images = crop_image_aspect_ratio(img_crop)

        result_crop = reader.read_fulltext(img_crop, output_format='dict')
        # 调整 all_results 中的坐标为原图坐标
        for result in result_crop:
            for point in result['boxes']:
                point[0] += x_min  # x坐标平移
                point[1] += y_min  # y坐标平移

        all_results.extend(result_crop)

    # 将 OpenCV 图像转换为 PIL 图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 设置字体，确保字体文件路径正确
    font_path = "SimHei.ttf"  # 请确保该字体文件存在于指定路径
    font = ImageFont.truetype(font_path, 20)  # 字体大小可以根据需要调整

    # 在图像上绘制文字识别结果
    for result in all_results:
        # 获取文字内容和置信度
        text = result['text']
        confidence = result['confident']
        
        # 获取文字区域的坐标框 boxes
        box = result['boxes']
        x_min, y_min = int(box[0][0]), int(box[0][1])  # 左上角坐标

        # 绘制文字和置信度
        label = f"{text} ({confidence:.2f})"
        draw.text((x_min, y_min - 10), label, font=font, fill=(255, 0, 0))  # 红色文字

    img_pil_path = 'result_pil.png'
    img_pil.save(img_pil_path)
    print(f"检测结果已保存至 {img_pil_path}")

    print(all_results)

if __name__ == "__main__":
    main()