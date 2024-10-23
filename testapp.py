import requests

url = 'http://127.0.0.1:5000/upload'
file_path = '/data1/zhn/macdata/code/github/python/font_ocr_rec/test.png'

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open('processed_image.jpg', 'wb') as f:
        f.write(response.content)
    print("Processed image saved as 'processed_image.jpg'")
else:
    print("Error:", response.json())