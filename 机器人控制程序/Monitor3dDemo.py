from ZEDCamera import ZEDCamera
from Monitor3D import Monitor3D
import cv2


if __name__ == '__main__':
    cameras = ZEDCamera.enum_cameras()
    camera = ZEDCamera(cameras[0], resolution=720, camera_fps=15, depth_min=400, depth_max=5000,
                       streaming=False, stream_host="10.101.27.249")

    screens = (0, 2)
    monitor = Monitor3D(trans_mat_file="monitor_cali.mat", screens=screens)

    key = 0
    while key != ord('e') and key != ord('E'):
        key = cv2.waitKey(50) & 0xFF

        camera.refresh()
        RGB_img1 = camera.get_RGBimage()
        RGB_img2 = camera.get_RGBimage_right()

        monitor.show(RGB_img1, RGB_img2)