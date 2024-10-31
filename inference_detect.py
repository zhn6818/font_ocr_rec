import cv2
import numpy as np
import sys
import os
from PIL import Image


sys.path.append("./EasyOCR/")

# 导入修改后的 ReaderRecog 类
from easyocr.easyocr import ReaderDetect

# 初始化 ReaderDetect 对象
detector = ReaderDetect(gpu=False, detect_network="craft", verbose=True)

# 测试图片路径
image_path = 'test.png'  # 替换为你的图片路径

horizontal_boxes, free_boxes = detector.detect_img(image_path, text_threshold=0.7, low_text=0.4, link_threshold=0.4)

# # 打印检测结果
# print("Horizontal Boxes Detected:")
# for box in horizontal_boxes[0]:
#     print(box)

# print("\nFree Form Boxes Detected:")
# for box in free_boxes:
#     print(box)

# 加载图片
img = cv2.imread(image_path)
maximum_y,maximum_x,_ = img.shape

# 绘制检测到的区域
for box in horizontal_boxes[0]:
    # 获取rect区域的坐标
    x_min = max(0,box[0])
    x_max = min(box[1],maximum_x)
    y_min = max(0,box[2])
    y_max = min(box[3],maximum_y)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# for box in free_boxes:
#     # 确保将自由形状框的每个点坐标转换为整数
#     pts = np.array([[int(pt[0]), int(pt[1])] for pt in box], np.int32)
#     pts = pts.reshape((-1, 1, 2))
#     cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

# 保存检测结果
result_path = 'result.png'
cv2.imwrite(result_path, img)
print(f"检测结果已保存至 {result_path}")
