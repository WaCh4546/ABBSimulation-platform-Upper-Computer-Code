import cv2
import platform
import ctypes

from RobotEye import RobotEye
from RobotABB import RobotABB
from ZEDCamera import ZEDCamera
from PlaneTracker import PlaneTracker


def on_mouse_event(event, x, y, flags, interface):
    if event == cv2.EVENT_LBUTTONDOWN:
        interface.on_mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        interface.on_mouse_up(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        interface.on_mouse_move(x, y)


class AutoFuelCtrl(object):
    def __init__(self, robot, robot_eye, CV_tracker):
        self.robot = robot               # the robot to be controlled
        self.robot_eye = robot_eye       # the depth camera mounted on the fueling pole
        self.CV_tracker = CV_tracker     # the CV program to track the fueling hole
        self.window_name = "Auto fuel control"

        # init monitoring window
        self.screen_width, self.screen_height = self.__get_window_size()
        if robot_eye.depth_camera.resolution[0] / robot_eye.depth_camera.resolution[1] > self.screen_width / self.screen_height:
            self.window_width = self.screen_width
        else:
            self.window_width = int(self.screen_height * robot_eye.depth_camera.resolution[0] / robot_eye.depth_camera.resolution[1])

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow(self.window_name, self.screen_width, self.screen_height)
        cv2.setMouseCallback(self.window_name, on_mouse_event, self)

        self.__mouse_target = None   # the target specified by mouse click, mainly for test

    def __get_window_size(self):
        if platform.system() == "Windows":
            winapi = ctypes.windll.user32
            return winapi.GetSystemMetrics(0), winapi.GetSystemMetrics(1)
        elif platform.system() == "Linux":
            pass

    def __get_target_TCP(self):
        """
        get target TCP coordinate
        :return:
        """
        target_xy = None
        if self.CV_tracker is not None:
            pass
        else:
            target_xy = self.__mouse_target
            self.__mouse_target = None   # after the target is used, it shall be deleted to avoid repeated movement

        if target_xy is None:
            return None

        pos = self.robot_eye.get_eye_position(px=target_xy[0], py=target_xy[1])
        if pos is not None:
            # 在这里做滤波
            pass
        return pos

    def __calc_TCP_movement(self, target_TCP):
        """
        由目标的TCP坐标，计算机器人的运动增量。
        在加油杆控制中，加油杆的虚拟支点不在机器人的末端，因此需要由相机获得的目标TCP坐标，计算出机器人的位置和姿态给定变化
        :param target_TCP:
        :return:
        """
        return (-target_TCP[0], -target_TCP[1], -target_TCP[2], 0, 0, 0, 0, 10)

    def on_mouse_move(self, px, py):
        pass

    def on_mouse_down(self, px, py):
        resize_k = self.robot_eye.depth_camera.resolution[0] / self.window_width
        px = int(resize_k * px + 0.5)
        py = int(resize_k * py + 0.5)
        self.__mouse_target = (px, py)

    def on_mouse_up(self, x, y):
        pass

    def refresh(self):
        key = cv2.waitKeyEx(10)
        if key == ord('q') or key == ord('Q'):
            exit(0)

        self.robot_eye.refresh()
        img = self.robot_eye.depth_camera.get_RGBimage(width=self.window_width, mark_infeasible=True)

        target_pos = self.__get_target_TCP()
        if target_pos is not None:
            move = self.__calc_TCP_movement(target_pos)
            self.robot.move_by_TCP_coordinate([move])

        cv2.imshow(self.window_name, img)


if __name__ == '__main__':
    robot = RobotABB(host="192.168.125.1", port=8002)
    if not robot.connected:
        print("Failed to connect the robot")
        exit(0)

    robot_eye = None
    try:
        camera = ZEDCamera()
        robot_eye = RobotEye(depth_camera=camera, eye_name="CaliTest", on_hand=True)
    except Exception:
        print("Failed to open the camera!")
        exit(0)

    #tracker = PlaneTracker("加油口定位2021-11-20-17-51.pt")

    tracer = AutoFuelCtrl(robot=robot, robot_eye=robot_eye, CV_tracker=None)
    while True:
        tracer.refresh()
