from DepthCamera import DepthCamera
import pyzed.sl as sl
import time
import cv2
import numpy as np

"""
Modification record
2022.06.22: 1. Streaming is added such that images can be transported through net.
            2. The refresh() function is improved. Images and depth-map are not obtained in refresh. 
               They are retrieved when needed.
2022.06.28：1. get_XYZ_ROI() is improved.
"""


class ZEDCamera(DepthCamera):
    def __init__(self, cam_number=None, resolution=2200, camera_fps=15, depth_min=400, depth_max=5000,
                 streaming=False, stream_host=None):
        """
        There are maybe multi cameras in one computer, in this case it is necessary to specify
        which camera is initialized. Each camera is recognized by its serial number.
        2.2K	4416x1242	15	                Wide
        1080p	3840x1080	30, 15	            Wide
        720p	2560x720	60, 30, 15	        Extra Wide
        WVGA	1344x376	100, 60, 30, 15	    Extra Wide
        :param cam_number: The serial number of the camera.
        :param resolution: HD2K, HD1080, HD720, VGA
        :param camera_fps: 15, 30, 60, 100.
        """
        if resolution == 2200:
            DepthCamera.__init__(self, (2208, 1242))
            resolution = sl.RESOLUTION.HD2K
        elif resolution == 1080:
            DepthCamera.__init__(self, (1920, 1080))
            resolution = sl.RESOLUTION.HD1080
        elif resolution == 720:
            DepthCamera.__init__(self, (1280, 720))
            resolution = sl.RESOLUTION.HD720
        else:
            DepthCamera.__init__(self, (672, 376))
            resolution = sl.RESOLUTION.VGA

        # init camera
        self.camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use PERFORMANCE depth mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
        init_params.camera_resolution = resolution
        init_params.camera_fps = camera_fps
        init_params.depth_minimum_distance = depth_min
        init_params.depth_maximum_distance = depth_max

        # the camera is used as a web-camera, so open it from the ip address specified by stream_host
        if streaming:
            if stream_host is not None:
                init_params.set_from_stream(stream_host)
        # normal camera
        else:
            if cam_number is None:
                cam_number = self.get_serial_number(0)
                if cam_number is None:
                    raise Exception("Camera does not exist!")
            init_params.set_from_serial_number(cam_number)
            self.camera_number = cam_number

        # 打开摄相机
        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            if streaming and stream_host is not None:
                raise Exception("Failed to open the remote camera!")
            else:
                raise Exception("Failed to open ZED camera!")

        # config the camera as a web-camera
        if streaming and stream_host is None:
            stream = sl.StreamingParameters()
            stream.codec = sl.STREAMING_CODEC.H264
            stream.bitrate = 4000
            status = self.camera.enable_streaming(stream)
            if status != sl.ERROR_CODE.SUCCESS:
                raise Exception("Failed to set ZED camera as stream sender")

        # 打开相机后创建并设置运行时参数
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
        self.runtime_parameters.confidence_threshold = 100
        self.runtime_parameters.textureness_confidence_threshold = 100

        # 获取相机内参fx,fy,Cx,Cy
        # 通过内参和深度数据，可以计算每个像素点的(x, y)坐标
        calibration_params = self.camera.get_camera_information().calibration_parameters
        self.cam_params = (calibration_params.left_cam.fx,
                           calibration_params.left_cam.fy,
                           calibration_params.left_cam.cx,
                           calibration_params.left_cam.cy)

        # 创建Mat对象，用于获取图像
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        self.point_cloud = sl.Mat()

    def camera_identity(self):
        return str(self.camera_number)

    @staticmethod
    def enum_cameras():
        cameras = sl.Camera.get_device_list()
        cam_numbers = list()
        for cam in cameras:
            cam_numbers.append(cam.serial_number)
        return cam_numbers

    @staticmethod
    def get_serial_number(camera_id):
        cameras = sl.Camera.get_device_list()
        if len(cameras) >= camera_id:
            return cameras[camera_id].serial_number
        else:
            return None

    def refresh(self):
        # left, right images and point_cloud may or may not be used, so they are set None.
        # the data will be retrieved used only when they are actually needed.
        DepthCamera.refresh(self)

        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.timestamp = time.time()
            return True
        else:
            return False

    def get_RGBimage(self, ROIonly=False, width=None, mark_infeasible=False):
        if self._RGBimage is None:
            self.camera.retrieve_image(self.left_image, sl.VIEW.LEFT)
            self._RGBimage = self.left_image.get_data()[:, :, 0:3]  # the last channel, alpha, is discarded

            # rotate and flip image if necessary
            if int(self.rot_angle / 90):
                self._RGBimage = np.rot90(self._RGBimage, int(self.rot_angle / 90))
            if self.flipud:
                self._RGBimage = np.flipud(self._RGBimage)

        return DepthCamera.get_RGBimage(self, ROIonly, width, mark_infeasible)

    def get_RGBimage_right(self, ROIonly=False, width=None):
        if self._RGBimage_right is None:
            self.camera.retrieve_image(self.right_image, sl.VIEW.RIGHT)
            self._RGBimage_right = self.right_image.get_data()[:, :, 0:3]  # the last channel, alpha, is discarded

            if int(self.rot_angle / 90):
                self._RGBimage_right = np.rot90(self._RGBimage_right, int(self.rot_angle / 90))
            if self.flipud:
                self._RGBimage_right = np.flipud(self._RGBimage_right)

        img = self.__get_ROI(self._RGBimage_right, self.ROI) if ROIonly else self._RGBimage_right
        if width is not None:
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
        return img

    def get_point_cloud(self):
        if self._Zmap is None:
            self.camera.retrieve_measure(self.point_cloud, sl.MEASURE.XYZ)
            self._Xmap = self.point_cloud.get_data()[:, :, 0]
            self._Ymap = self.point_cloud.get_data()[:, :, 1]
            self._Zmap = self.point_cloud.get_data()[:, :, 2]

            if int(self.rot_angle / 90):
                self._Xmap = np.rot90(self._Xmap, int(self.rot_angle / 90))
                self._Ymap = np.rot90(self._Ymap, int(self.rot_angle / 90))
                self._Zmap = np.rot90(self._Zmap, int(self.rot_angle / 90))
            if self.flipud:
                self._Xmap = np.flipud(self._Xmap)
                self._Ymap = np.flipud(self._Ymap)
                self._Zmap = np.flipud(self._Zmap)
        return self._Xmap, self._Ymap, self._Zmap

    def get_XYZ_ROI(self, ROI=None):
        self.get_point_cloud()
        return DepthCamera.get_XYZ_ROI(self, ROI)

if __name__ == '__main__':
    cameras = ZEDCamera.enum_cameras()
    print("installed zed cameras:")
    print(cameras)

    camera1 = ZEDCamera(cameras[1], resolution=720, camera_fps=15, depth_min=400, depth_max=5000, streaming=False)

    key = 0
    while key != ord('e') and key != ord('E'):
        camera1.refresh()
        img = camera1.get_RGBimage()
        cv2.imshow("", img)
        key = cv2.waitKey(50) & 0xFF





