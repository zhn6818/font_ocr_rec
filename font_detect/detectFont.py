import pickle
import torch
from PIL import Image
from torchvision import transforms
from detector.fontmodel import ResNet50Regressor,FontDetector
from detector import config

class FontDetectionInterface:
    def __init__(self, model_path="/data4/qwy/font_ocr_rec/font_detect/model/7class/font_torch1.pth", 
                 font_cache_path = "/data4/qwy/font_ocr_rec/font_detect/model/7class/font_demo_cache.bin", device_id=2):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        self.model = ResNet50Regressor().to(self.device)

        self.detector = FontDetector(
            model=self.model,
            lambda_font=1,
            lambda_direction=1,
            lambda_regression=1,
            font_classification_only=False,
            lr=1,
            betas=(1, 1),
            num_warmup_iters=1,
            num_iters=int(1e9),
            num_epochs=int(1e9),
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.detector.load_state_dict(state_dict)
        self.detector.eval()

        self.font_list = pickle.load(open(font_cache_path, 'rb'))

        self.transform = transforms.Compose([
            transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
            transforms.ToTensor(),
        ])
        
    def recognize_font(self,image):
        # 增加图片等比例缩放
        max_size = config.INPUT_SIZE
        width, height = image.size

        new_height = max_size
        if width > height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_height = max_size
            new_width = int((width / height) * max_size)

        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        new_img = Image.new("RGB", (config.INPUT_SIZE, config.INPUT_SIZE), (255, 255, 255))
        x_offset = (config.INPUT_SIZE - new_width) // 2
        y_offset = (config.INPUT_SIZE - new_height) // 2
        new_img.paste(image, (x_offset, y_offset))
        
        # 增加图片等比例缩放
        transformed_image = self.transform(new_img)
        with torch.no_grad():
            transformed_image = transformed_image.to(self.device)
            output = self.detector(transformed_image.unsqueeze(0))
            prob = output[0][: config.FONT_COUNT].softmax(dim=0)

            top_index = torch.argmax(prob).item()
            # 返回识别结果
            return {
                self.font_list[top_index].path: float(prob[top_index])
            }

if __name__ == "__main__":
    font_recognizer = FontDetectionInterface()
    path = "/data4/qwy/temp/YuzuMarker.FontDetection/dataset/font_img/test/font_4_img_19.jpg"
    image = Image.open(path)
    out = font_recognizer.recognize_font(image)
    print(out)