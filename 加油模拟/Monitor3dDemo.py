from ZEDCamera import ZEDCamera
from Monitor3D import Monitor3D
import cv2


if __name__ == '__main__':
    cameras = ZEDCamera.enum_cameras()
    camera = ZEDCamera(cameras[0], resolution=1080, camera_fps=15, depth_min=400, depth_max=5000)

    monitor = Monitor3D(trans_mat_file="monitor_cali.mat")

    key = 0
    while key != ord('e') and key != ord('E'):
        key = cv2.waitKey(50) & 0xFF

        camera.refresh()
        RGB_img1 = camera.RGBimage
        RGB_img2 = camera.RGBimage_right

        monitor.show(RGB_img1, RGB_img2)