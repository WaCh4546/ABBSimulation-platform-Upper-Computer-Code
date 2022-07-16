import cv2
import sys
import numpy as np
from DepthCamera import DepthCamera
import plot3dUtils

sys.path.insert(1, './pyKinectAzure/')
from pyKinectAzure import pyKinectAzure, _k4a

# 添加 Azure Kinect SDK 路径
#modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll'


def color_depth_image(depth_image):
    depth_color_image = cv2.convertScaleAbs(depth_image,
                                            alpha=0.05)  # alpha is fitted by visual comparison with Azure k4aviewer results
    depth_color_image = cv2.applyColorMap(depth_color_image, cv2.COLORMAP_JET)

    return depth_color_image



class KinectCamera(DepthCamera):
    def __init__(self):
        DepthCamera.__init__(self, (1280, 720))

        # 初始化
        modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll'
        self.pyK4A = pyKinectAzure(modulePath)
        self.pyK4A.device_open()
        device_config = self.pyK4A.config
        device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
        print(device_config)

        # 开启摄像头
        self.pyK4A.device_start_cameras(device_config)
        # 获取相机序列号
        serial_number = self.pyK4A.device_get_serialnum()
        print(serial_number)

    def refresh(self):
        self.pyK4A.device_get_capture()
        depth_image_handle = self.pyK4A.capture_get_depth_image()
        color_image_handle = self.pyK4A.capture_get_color_image()

        success = False
        if depth_image_handle and color_image_handle:
            self.RGBimage = self.pyK4A.image_convert_to_numpy(color_image_handle)[:, :, :3]
            self.Zmap = self.pyK4A.image_convert_to_numpy(depth_image_handle)
            success = True

        self.pyK4A.image_release(depth_image_handle)
        self.pyK4A.image_release(color_image_handle)
        self.pyK4A.capture_release()

        return success

    def get_depth_image(self, ROIonly=False, width=None, min_depth=None, max_depth=None):
        return color_depth_image(self.Zmap)


if __name__ == '__main__':
    camera = KinectCamera()

    key = 0
    while key != ord('e') and key != ord('E'):
        key = cv2.waitKey(50) & 0xFF
        if camera.refresh():
            cv2.imshow("RGB", camera.RGBimage)
            cv2.imshow("Depth", camera.get_depth_image())

