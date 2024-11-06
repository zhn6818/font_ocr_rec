import requests
import argparse
import json

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="上传图片并请求预测")
parser.add_argument('image_path', type=str, help="输入要上传的图片路径")
args = parser.parse_args()

# 设置Flask服务器地址
url = 'http://127.0.0.1:5000/predict'

# 打开你想上传的图像
with open(args.image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)
# 假设 response 是请求的返回结果
response_data = response.json()  # 获取 Python 字典
# 获取并打印响应
json_string = json.dumps(response_data, ensure_ascii=False)
print(json_string)  # 现在会以双引号显示

# import concurrent.futures
# import requests

# def send_request():
#     url = "http://localhost:5000/predict"
#     files = {'file': open('test1.png', 'rb')}
#     response = requests.post(url, files=files)
#     print(response.status_code)

# # 使用 ThreadPoolExecutor 发送多个请求
# with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#     for _ in range(5):  # 发送5个并发请求
#         executor.submit(send_request)