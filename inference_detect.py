import cv2
import numpy as np
import sys
import os
from PIL import Image
from PIL import ImageDraw, ImageFont

sys.path.append("./EasyOCR/")
from easyocr.easyocr import ReaderDetect, ReaderRecog

def crop_image_aspect_ratio(image, aspect_ratio=5):
    """根据给定的宽高比裁剪图像，保持宽高比为5:1"""
    img_height, img_width = image.shape[:2]
    
    crop_height = img_height
    crop_width = int(crop_height * aspect_ratio)
    
    if crop_width > img_width:
        crop_width = img_width
    
    crops = []
    
    for start_x in range(0, img_width, crop_width):
        end_x = start_x + crop_width
        crop = image[:, start_x:end_x]
        if crop.shape[1] > 0:
            crops.append(crop)
    
    return crops

def initialize_readers():
    """初始化检测和识别对象"""
    detector = ReaderDetect(gpu=False, detect_network="craft", verbose=True)
    languages = ['ch_sim', 'en']
    reader = ReaderRecog(languages, gpu=True)
    return detector, reader

def process_images(image_paths):
    """处理多张图片"""
    detector, reader = initialize_readers()

    for image_path in image_paths:
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
            cropped_images = crop_image_aspect_ratio(img_crop)

            result_crop = reader.read_fulltext(img_crop, output_format='dict')
            for result in result_crop:
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
            x_min, y_min = int(box[0][0]), int(box[0][1])

            label = f"{text} ({confidence:.2f})"
            draw.text((x_min, y_min - 10), label, font=font, fill=(255, 0, 0))

        img_pil_path = f'result_{os.path.basename(image_path)}'
        img_pil.save(img_pil_path)
        print(f"检测结果已保存至 {img_pil_path}")
        print(all_results)

def main():
    image_paths = ['test1.png', 'test2.png', 'test3.png']  # 替换为你的图片路径列表
    process_images(image_paths)

if __name__ == "__main__":
    main()