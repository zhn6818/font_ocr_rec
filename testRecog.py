import requests

# 设置Flask服务器地址
url = 'http://127.0.0.1:5000/predict'

# 打开你想上传的图像
with open('test1.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# 获取并打印响应
print(response.json())

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