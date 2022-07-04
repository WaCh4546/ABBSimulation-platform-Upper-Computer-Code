#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO
from ZEDCamera import ZEDCamera


class detect_airplane():
    def __init__(self):
        self.yolo = YOLO()

    def get_airplane_info(self,image):
        frame = cv2.cvtColor(camera1.RGBimage, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        inform, _ = self.yolo.detect_image(frame)

        return inform

if __name__ == "__main__":
    cameras = ZEDCamera.enum_cameras()
    camera1 = ZEDCamera(cameras[0], resolution=720, camera_fps=15)
    da=detect_airplane()
    while True:
        camera1.refresh()
        # 格式转变，BGRtoRGB
        da.get_airplane_info(camera1.RGBimage)
