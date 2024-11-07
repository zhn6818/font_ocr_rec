import gradio as gr 
import pandas as pd 

from PIL import Image
# from torchkeras import plots 
# from torchkeras.data import get_url_img

from pathlib import Path
# from ultralytics import YOLO
# import ultralytics
#from ultralytics.yolo.data import utils 
import torch
from torchvision import models, transforms
import os







# model = YOLO('yolov8n.pt')



# #load class_names
# #yaml_path = str(Path(ultralytics.__file__).parent/'datasets/coco128.yaml') 
# class_names = []
# for i in range(1000):
#     class_names.append(str(i))

# def detect(img):
#     if isinstance(img,str):
#         img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
#     result = model.predict(source=img)
#     print()
#     if len(result[0].boxes.data)>0:
#         vis = plots.plot_detection(img,boxes=result[0].boxes.data,
#                      class_names=class_names, min_score=0.2)
#     else:
#         vis = img
#     return vis



# def load_model_without_dataparallel(model, checkpoint_path):
#     state_dict = torch.load(checkpoint_path)
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key[len('module.'):] if key.startswith('module.') else key
#         new_state_dict[new_key] = value
#     model.load_state_dict(new_state_dict)
#     return model

# # 图像处理函数，与训练时保持一致
# def pad_and_resize(img):
#     max_size = 64
#     width, height = img.size
#     if width > height:
#         new_width = max_size
#         new_height = int((height / width) * max_size)
#     else:
#         new_height = max_size
#         new_width = int((width / height) * max_size)

#     img = img.resize((new_width, new_height), Image.LANCZOS)
#     new_img = Image.new("RGB", (64, 64), (0, 0, 0))  # 用黑色填充
#     x_offset = (64 - new_width) // 2
#     y_offset = (64 - new_height) // 2
#     new_img.paste(img, (x_offset, y_offset))

#     return new_img

# # # 单张图片推理
# # def predict_single_image(model, device, image_path, class_names):
# #     # 数据预处理
# #     transform = transforms.Compose([
# #         transforms.ToTensor(),
# #         # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 根据实际情况选择是否应用标准化
# #     ])
    
# #     # 加载并处理图片
# #     image = Image.open(image_path).convert('RGB')
# #     image = pad_and_resize(image)  # 与训练阶段保持一致
# #     image = transform(image).unsqueeze(0)  # 添加批次维度
# #     image = image.to(device)

# #     # 模型推理
# #     model.eval()
# #     with torch.no_grad():
# #         output = model(image)
# #         _, predicted = torch.max(output, 1)

# #     predicted_class = class_names[predicted.item()]
# #     print(f"预测类别: {predicted_class}")

# #     return predicted_class


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 加载测试数据集以获取类别名称
# #test_dataset = CustomImageFolder(root=extract_path, transform=None)
# #class_names = test_dataset.get_class_names()
# class_namesfile = '20241014/label.txt'
# class_names = []
# with open(class_namesfile,'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.split(' ')[0].strip('\r\n')
#         class_names.append(line)
# print(class_names)
# print(len(class_names))
# #class_names = ['FangSong','Kaiti','SimHei','STLiti','YouYuan']

# # 加载模型
# model = models.resnet50(pretrained=True)
# num_classes = len(class_names)
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# checkpoint_path = '20241014/best_model_acc0.9952_20241012-175253.pth'
# if os.path.exists(checkpoint_path):
#     model = load_model_without_dataparallel(model, checkpoint_path)
#     print(f"成功加载模型: {checkpoint_path}")
# else:
#     print("找不到模型，使用预训练模型进行推理")

# model.to(device)

# def detect(img):
#     # 数据预处理
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 根据实际情况选择是否应用标准化
#     ])
#         # 加载并处理图片
#     if isinstance(img,str):
#         img = get_url_img(img) if img.startswith('http') else Image.open(img).convert('RGB')
#     import datetime
#     time = str(datetime.datetime.now())
#     img.save('images/'+time+'.jpg')
#     image = pad_and_resize(img)  # 与训练阶段保持一致
#     image = transform(image).unsqueeze(0)  # 添加批次维度
#     image = image.to(device)

#     # 模型推理
#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)

#     predicted_class = class_names[predicted.item()]
#     print(f"预测类别: {predicted_class}")

#     return predicted_class

# def crop_image_aspect_ratio(font_recognizer, image, aspect_ratio=5):
#     """根据给定的宽高比裁剪图像，保持宽高比为5:1"""
#     img_height, img_width = image.shape[:2]
    
#     crop_height = img_height
#     crop_width = int(crop_height * aspect_ratio)
    
#     if crop_width > img_width:
#         crop_width = img_width
    
#     crops = []
#     font_counts = Counter()  # 用于统计字体出现次数
    
#     for start_x in range(0, img_width, crop_width):
#         end_x = start_x + crop_width
#         crop = image[:, start_x:end_x]
#         if crop.shape[1] > 0:
#             crops.append(crop)
#             # 将 BGR 格式的 OpenCV 图像转换为 RGB 格式
#             rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

#             # 将 NumPy 数组转换为 PIL 图像
#             pil_image = Image.fromarray(rgb_image)
#             out = font_recognizer.recognize_font(pil_image)
#             # 更新字体统计
#             if isinstance(out, dict):
#                 for font, confidence in out.items():
#                     font_counts[font] += 1  # 统计字体出现次数
#             # print(out)
#     # 找到出现次数最多的字体类别
#     most_common_font = font_counts.most_common(1)  # 获取出现次数最多的字体
#     if most_common_font:
#         most_common_font_name, count = most_common_font[0]
#         result = {
#             "most_common_font": most_common_font_name,
#             "count": count,
#             "all_font_counts": font_counts
#         }
#     else:
#         result = {
#             "most_common_font": None,
#             "count": 0,
#             "all_font_counts": font_counts
#         }

#     return result

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

# def initialize_readers():
#     """初始化检测和识别对象"""
#     detector = ReaderDetect(gpu=False, detect_network="craft", verbose=True)
#     languages = ['ch_sim', 'en']
#     reader = ReaderRecog(languages, gpu=True)
#     font_recognizer = FontDetectionInterface()
#     return detector, reader, font_recognizer
languages = ['ch_sim', 'en']
detector = ReaderDetect(gpu=False, detect_network="craft", verbose=True)
reader = ReaderRecog(languages, gpu=True)
font_recognizer = FontDetectionInterface()
def detect(image_path):
    """处理单张图片"""
    # print(type(image_path))
    # image_path.save('res.jpg')
    # image_path = 'res.jpg'
    
    # 将 PIL 图像换为 OpenCV 图像（BGR 格式）
    numpy_image = np.array(image_path)
    img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    
    horizontal_boxes, free_boxes = detector.detect_img(img, text_threshold=0.7, low_text=0.4, link_threshold=0.4)

    # img = cv2.imread(image_path)
    # img = Image.open(image_path).convert('RGB')
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
    font_size = 10  # 调整字体大小
    font = ImageFont.truetype(font_path, font_size)

    # 在图像上绘制文字识别结果
    for result in all_results:
        text = result['text']
        confidence = result['confident']
        box = result['boxes']
        font_line = result['font']
        x_min, y_min = int(box[0][0]), int(box[0][1])

        label = f" ({font_line}) {text} ({confidence:.2f})"

        # 创建一个新的透明图层
        text_layer = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))

        # 创建 ImageDraw 对象
        draw_layer = ImageDraw.Draw(text_layer)

        # 设置文本透明度（0-255，0为完全透明，255为完全不透明）
        opacity = 200  # 设置为200以控制透明度
        rgba_color = (255, 0, 0, opacity)  # 红色文本，带透明度

        # 在透明图层上绘制文本
        draw_layer.text((x_min, y_min - 10), label, font=font, fill=rgba_color)

        # 合成文本图层和原图像
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), text_layer)

    return img_pil
    #img_pil_path = f'result_{os.path.basename(image_path)}'
    # img_pil.save(img_pil_path)
    # print(f"检测结果已保存至 {img_pil_path}")
    # print(all_results)


with gr.Blocks() as demo:

    with gr.Tab("上传本地图片"):
        input_img = gr.Image(type='pil')
        button = gr.Button("执行检测",variant="primary")
        
        gr.Markdown("## 预测输出")
        out_img = gr.Image(type='pil')
        #out_img = gr.Text()
        
        button.click(detect,
                     inputs=input_img, 
                     outputs=out_img)


gr.close_all() 
demo.queue()
demo.launch(server_name="0.0.0.0",server_port=7868)

# import gradio as gr

# def greet(name, intensity):
#     return "Hello " * intensity + name + "!"

# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "slider"],
#     outputs=["text"],
# )

# demo.launch(server_name="0.0.0.0")