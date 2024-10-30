import sys
import os
from PIL import Image
import cv2

sys.path.append("./EasyOCR/")

from easyocr.easyocr import Reader

languages = ['ch_sim', 'en']

reader = Reader(languages)

result = reader.read_fulltext('./img/cropped_3.png', output_format='dict')
# result = reader.read_fulltext('zhong.png', output_format='dict')

for item in result:
    print(f"Text: {item['text']}, Confidence: {item['confident']}")