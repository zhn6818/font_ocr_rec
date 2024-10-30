# import sys
# import os
# from PIL import Image
# import cv2

# sys.path.append("./EasyOCR/")

# from easyocr.easyocr import Reader

# languages = ['ch_sim', 'en']

# reader = Reader(languages)

# # result = reader.read_fulltext('./img/cropped_0.png', output_format='dict')
# result = reader.read_fulltext('zhong.png', output_format='dict')

# for item in result:
#     print(f"Text: {item['text']}, Confidence: {item['confident']}")

import sys
import os
from PIL import Image
import cv2

sys.path.append("./EasyOCR/")

# 导入修改后的 ReaderRecog 类
from easyocr.easyocr import ReaderRecog

# 选择语言
languages = ['ch_sim', 'en']

# 创建 ReaderRecog 实例
reader = ReaderRecog(languages, gpu=True)

# 使用 read_fulltext 读取并识别图片
# result = reader.read_fulltext('./img/cropped_0.png', output_format='dict')
result = reader.read_fulltext('zhong.png', output_format='dict')

# 输出识别结果
for item in result:
    print(f"Text: {item['text']}, Confidence: {item.get('confident', 'N/A')}")