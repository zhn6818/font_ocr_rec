import requests

# 设置Flask服务器地址
url = 'http://127.0.0.1:5000/predict'

# 打开你想上传的图像
with open('test1.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# 获取并打印响应
print(response.json())