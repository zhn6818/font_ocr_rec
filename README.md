# font_ocr_rec


## git clone --recurse-submodules https://github.com/zhn6818/font_ocr_rec.git

* 预训练权重地址： 90  /data1/zhn/model/EasyOCR/
* 你需要将对应的文件， 移动到 /root/.EasyOCR/model/ 文件夹中，ps 你自己的docker环境都需要能访问到这些权重文件
* 即 cp /data1/zhn/model/EasyOCR/   /root/.EasyOCR/model/   文件夹不存在，自己创建

## python inference_detect.py
* 批量处理传入的图片，处理结果绘制在原图上。结果保存在result_(文件名)文件中

## python appRecog.py
* 启动5000端口服务，等待请求

## python testRecog.py test4.png
* 上述脚本实例为，传入图片进行测试