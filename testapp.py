import requests

url = 'http://127.0.0.1:5000/upload'
file_path = '/data1/zhn/macdata/code/github/python/font_ocr_rec/test.png'

try:
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        # 从响应中获取OCR结果并打印
        ocr_results = response.json().get('ocr_results', [])
        if ocr_results:
            print("OCR Results:")
            for item in ocr_results:
                bbox = item['bbox']
                # 使用 f-string 格式化输出
                print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}, Bounding Box: {bbox['x']} {bbox['y']} {bbox['w']} {bbox['h']}")
        else:
            print("No text detected in the image.")
    else:
        print("Error:", response.status_code, response.json())
        
except FileNotFoundError:
    print(f"Error: File not found at path '{file_path}'")
except requests.RequestException as e:
    print(f"Request failed: {e}")