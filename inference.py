import sys

sys.path.append("./EasyOCR/")

from easyocr.easyocr import Reader

languages = ['zh', 'en']

# 初始化Reader对象
reader = Reader(languages)

# 图片路径（也可以使用图像的numpy数组）
image_path = 'test.png'

# 使用readtext方法进行OCR识别
results = reader.readtext(image_path)

# 输出识别结果
for result in results:
    print(result)


