import cv2
import math
import time
import numpy as np
import win32api, win32con
import threading
import pygame
import socket
import psutil
from enum import Enum

from ABB_EGM import EGM
from RobotABB import RobotABB
from ZEDCamera import ZEDCamera
from JoyStick import BoomJoystick
from JoyStick import TestJoystick_Boom
from PlaneTracker import PlaneTracker
from RobotEye import RobotEye
from CVMenu import CVMenu
from PID import PID
from ICCamera import ICCamera
from BiCameraCtrl import BiCameraCtrl
from Monitor3D import Monitor3D
from ImgTrans import ImgTrans

"""
2022.02.22: The program is slightly fined.
2022.02.26: 1. Remote control of the binocular camera is added but not tested.
            2. Kalman filter is added in tracker but not tuned and tested.
2022.06.16: 1. 3D monitor is tested
            2. Yolo is used for target tracking
2022.06.24: 1. Boom position indicator is redesigned
            2. Another ZED camera is used to replace the bi-camera 
            3. The tracker is improved
2022.07.12: The project acceptance check is accomplished. Further improvement shall be continued.
2022.07.14: In the previous version, joystick is refreshed and used in BoomCtrl class(main thread). Because the main
            thread is relatively slow(about 220ms each interval), there is a strong feeling of time delay when people
            operates the boom manually. Therefore joystick is moved into BoomCtrlThread, which is much faster(no more 
            than 25ms each interval). The modification has not yet been tested.
"""


"""
constant values
"""
ROBOT_IP = "192.168.1.103"
ROBOT_PORT = 8002

EGM_IP = ROBOT_IP
EGM_PORT = 6511

REMOTE_IP = "192.168.1.101"
REMOTE_PORT = 5678

BINOCULAR_IP = "192.168.1.111"
#BINOCULAR_IP = "127.0.0.1"
BINOCULAR_PORT = 2005
BINOCULAR_LOCAL_PORT = 20005

USCREEN_IP = "192.168.1.111"
USCREEN_PORT = 10000

# TRACK_MODEL = "加油口定位2021-11-20-17-51.pt"
TRACK_MODEL = "加油口定位2022-06-07-13-00.pt"

FIXED_POS_EULERS = {
            'EGM start': [(4000, 0, -1000, 0, 30, 0, 100, 0), (4000, 0, -1200, 0, 30, 0, 100, 0)],
            'rest': [(4880, 0, -320, 0, 0, 0, 100, 0)],
            'projector': [(4880, 0, 0, 0, 0, 0, 100, 0)],
            'maintenance': [(4000, -2240, -3000, -25, 28, 0, 100, 0)]
        }

BOOM_LENGTH = 2842           # in mm
BOOM_STRETCH_RANGE = 1000    # in mm
BOOM_STRETCH_SPEED = 0.5     # the boom stretching is controlled by a low-pass filter
BOOM_ANGLE_RANGE = 10        # in degree
BOOM_ANGLE_LINEAR = 8        # in degree
BOOM_ANGLE_SPD_LMT = 0.1     # Dead-zone speed, that is, the angle speed be zero if it's smaller than this value.
BOOM_ANGLE_SPD_MAX = 2.5     # The max angle speed in boom auto-tracking.
BOOM_CTRL_MIN_ERROR = 10     # The dead-zone of the position error in boom auto-tracking.
BOOM_CTRL_INTERVAL = 0.02
BOOM_CTRL_KP = 0.01
BOOM_CTRL_KD = 0

BICAMERA_LEFT = "DFK 33UX265 9124453"
BICAMERA_RIGHT = "DFK 33UX265 4124038"

ZED_MONITOR = 23441
ZED_ON_BOOM = 6468932

TEST_IN_LAB = True
NO_ROBOT = False
NO_BICAMERA = False
SHOW_MONITOR_INFO = True
SCREEN_NUMBER = 2
RIGHT_SCREEN_BLACK = False

GAUGE_EDGE = 10
GAUGE_WIDTH = 10
GAUGE_SCALE_CNT = 5

MENU_LEFT = 60
MENU_TOP = 60

SHOW_RAW_TRACKER = True


class BoomCtrlThread(threading.Thread):
    def __init__(self, lock, joystick):
        threading.Thread.__init__(self)
        self.lock = lock
        self.robot = None
        self.joystick = joystick
        self.auto_tracking = False
        self.terminated = False

        # set the initial boom state
        self.yaw_speed = 0
        self.pitch_speed = 0
        self.stretch_order = 0
        self.yaw = 0
        self.pitch = 0
        self.stretch = 0

        self.virtual_joint_pos = (0, 0, 0)
        self.init_euler = (0, 30, 0)

        # socket for sending status to the remote computer
        self.remote = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.__lst_time = time.time()
        self.loop_time = 0

    @staticmethod
    def __rotate_matrix(yaw, pitch, roll):
        # the input euler angle uses degree as unit, which shall be translated into rad before sin and cos calculation.
        yaw = yaw * np.pi / 180.0
        pitch = pitch * np.pi / 180.0
        roll = roll * np.pi / 180.0

        m_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])
        m_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])
        m_roll = np.array([[1, 0, 0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll), np.cos(roll)]])
        return m_yaw @ m_pitch @ m_roll

    def __calc_boom_offset(self, yaw, pitch, roll, L):
        """
        Calculate the vector from the virtual joint to the boom end.
        :param yaw:
        :param pitch:
        :param L:
        :return:
        """
        r_matrix = self.__rotate_matrix(yaw, pitch, roll)
        boom = np.array((L, 0, 0))
        pos = (r_matrix @ boom.T).T
        return pos[0], pos[1], pos[2]

    def __calc_virtual_joint_pos(self, robot_pos_euler):
        """
        Calculate the pos and euler of the virtual joint
        :param init_pos:
        :param init_euler:
        :param boom_len:
        :return:
        """
        x, y, z = self.__calc_boom_offset(robot_pos_euler[3], robot_pos_euler[4], robot_pos_euler[5], BOOM_LENGTH)
        return robot_pos_euler[0] - x, robot_pos_euler[1] - y, robot_pos_euler[2] - z

    @staticmethod
    def __sigmoid(x, lin_x, max_x):
        abs_x = math.fabs(x)
        if abs_x < lin_x:
            y = x
        else:
            K = max_x - lin_x
            T = 2.0 / K
            y = 1.0 + math.exp(-T * (abs_x - lin_x))
            y = K * (2.0 / y - 1.0) + lin_x
            if x < 0:
                y = -y
        return y

    def __speed_accumulate(self, speed, x, lin_x, max_x):
        if math.fabs(speed) > BOOM_ANGLE_SPD_LMT:
            x += speed * BOOM_CTRL_INTERVAL
            x = self.__sigmoid(x, lin_x, max_x)
        return x

    def __calc_pos_euler(self):
        # accumulate angle speed to obtain boom euler angle
        # boom length is directly controlled, so it is simplly limited within range
        self.yaw = self.__speed_accumulate(self.yaw_speed, self.yaw, BOOM_ANGLE_LINEAR, BOOM_ANGLE_RANGE)
        self.pitch = self.__speed_accumulate(self.pitch_speed, self.pitch, BOOM_ANGLE_LINEAR, BOOM_ANGLE_RANGE)
        self.stretch = BOOM_STRETCH_SPEED * self.stretch + (1.0 - BOOM_STRETCH_SPEED) * self.stretch_order

        robot_yaw = self.init_euler[0] + self.yaw
        robot_pitch = self.init_euler[1] + self.pitch
        robot_roll = self.init_euler[2]
        robot_stretch = BOOM_LENGTH + self.stretch * BOOM_STRETCH_RANGE
        x, y, z = self.__calc_boom_offset(robot_yaw, robot_pitch, robot_roll, robot_stretch)
        return (x + self.virtual_joint_pos[0], y + self.virtual_joint_pos[1], z + self.virtual_joint_pos[2],
                robot_yaw, robot_pitch, robot_roll)

    def start_ctrl(self):
        try:
            self.lock.acquire()
            robot = EGM(IP=EGM_IP, PortNum=EGM_PORT)
            init_pos_euler = robot.get_robot_pos_euler()
            if init_pos_euler is None:
                self.robot = None
                raise Exception("Failed to get position from robot")

            self.robot = robot
            self.auto_tracking = False
            self.virtual_joint_pos = self.__calc_virtual_joint_pos(init_pos_euler)
            self.init_euler = (init_pos_euler[3], init_pos_euler[4], init_pos_euler[5])
            self.yaw = self.pitch = self.stretch = 0
        finally:
            self.lock.release()

    def stop_ctrl(self):
        self.lock.acquire()
        self.robot = None
        self.lock.release()

    def run(self):
        while not self.terminated:
            time.sleep(BOOM_CTRL_INTERVAL)
            self.joystick.refresh()

            self.lock.acquire()
            if self.robot is not None:
                if not self.auto_tracking:
                    self.yaw_speed = -self.joystick.yaw * 4
                    self.pitch_speed = -self.joystick.pitch * 4
                self.stretch_order = self.joystick.stretch

                pos_euler = self.__calc_pos_euler()
                self.robot.set_robot_pos_euler(pos_euler)

                # send status to the computer that shows it on the screen hang on the tower
                str = "%1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f" % (pos_euler)
                self.remote.sendto(str.encode('utf-8'), (REMOTE_IP, REMOTE_PORT))
            self.lock.release()

            # calculate looping time
            self.loop_time = (time.time() - self.__lst_time) * 1000
            self.__lst_time = time.time()


class Interfaces(Enum):
    ROOT = 0,
    PARK = 1,
    MANUAL_CTRL = 2,
    AUTO_CTRL = 3,
    STEREO_CAMERA = 4


class FlyingBoomCtrl(object):
    def __init__(self):
        self.__log_string = list()

        self.__init_hardwares()
        self.__init_screen()
        self.tracker = PlaneTracker(TRACK_MODEL)
        self.menu = CVMenu(left=MENU_LEFT, top=MENU_TOP, width=250, height=25)

        self.__show_boom_eye = True    # Show images from ZED camera
        self.__auto_tracking = False
        self.__cur_interface = Interfaces.ROOT

        self.__lst_time = time.time()
        self.__target_pos = None
        self.__recording = False
        self.__video = None
        self.__take_picture = False

        self.lock = threading.RLock()
        self.boom_ctrl_thread = BoomCtrlThread(self.lock, self.joystick)
        self.boom_ctrl_thread.start()

        self.img_trans = ImgTrans((USCREEN_IP, USCREEN_PORT))

        self.__enter_root_interface()

    def __init_hardwares(self):
        """
        When the application startS, initialize all the necessary hardwares first.
        If the joystick is not connected, the application exists.
        If other hardwares are not connected, the associated functions will be canceled.
        :return: None
        """
        print("Openning joystick")
        try:
            pygame.init()
            self.joystick = BoomJoystick() if not TEST_IN_LAB else TestJoystick_Boom()
        except Exception as info:
            self.joystick = None
            print(info)
            exit(0)

        print("Openning ZED camera on the boom")
        try:
            depth_camera = ZEDCamera() if TEST_IN_LAB else ZEDCamera(ZED_ON_BOOM)
            self.boom_eye = RobotEye(depth_camera=depth_camera, eye_name="FlyingBoom", on_hand=True)
        except Exception:
            self.boom_eye = None
            self.__print("Failed to open the ZED camera")

        print("Openning ZED camera for monitoring")
        try:
            depth_camera = None if TEST_IN_LAB else ZEDCamera(ZED_MONITOR)
            self.monitor_zed = depth_camera
        except Exception:
            self.monitor_zed = None
            self.__print("Failed to open the ZED camera")

        if not TEST_IN_LAB and not NO_ROBOT:
            print("Openning robot")
            try:
                self.robot = RobotABB(host=ROBOT_IP, port=ROBOT_PORT)
                if not self.robot.connected:
                    raise Exception("Failed to connect robot!")

                if not self.robot.start():
                    raise Exception("Failed to start robot!")

                self.robot.clear()
            except Exception as info:
                self.robot = None
                self.__print(info)
        else:
            self.robot = None

        """
        self.__bicamera_ctrl = BiCameraCtrl(local_port=BINOCULAR_LOCAL_PORT,
                                            remote_ip=BINOCULAR_IP, remote_port=BINOCULAR_PORT, timeout=0.1)
        """

    def __init_screen(self):
        """
        Obtain the screen size, and initialize the cv window.
        :return: None
        """
        screens = (0, 2) if TEST_IN_LAB else None
        cali_file = "monitor_cali_lab.mat" if TEST_IN_LAB else "monitor_cali.mat"
        self.monitor = Monitor3D(trans_mat_file=cali_file, screens=screens)
        self.screen_size = self.monitor.screen_size

    def __print(self, string, color=(0, 0, 255)):
        self.__log_string.append((string, color))
        if len(self.__log_string) > 6:
            del self.__log_string[0]

    def __menu_operate(self):
        if self.joystick.menu_option == 1:
            self.menu.move_up()
        elif self.joystick.menu_option == -1:
            self.menu.move_down()
        elif self.joystick.menu_option == 2:
            self.__menu_select()
        self.joystick.menu_option = 0

    def __menu_select(self):
        menu_item = self.menu.selected()
        if menu_item is None:
            return

        # switch bewteen ZED and stereo-camera occurs in several different interfaces,
        # so the operation is carried out independently
        caption = menu_item['caption']
        if caption == "Switch to stereo-camera":
            self.__show_boom_eye = False
            self.menu.items[0]["caption"] = "Switch to depth-camera"
        elif caption == "Switch to depth-camera":
            self.__show_boom_eye = True
            self.menu.items[0]["caption"] = "Switch to stereo-camera"

        if self.__cur_interface == Interfaces.ROOT:
            self.__menu_select_ROOT(menu_item)
        elif self.__cur_interface == Interfaces.PARK:
            self.__menu_select_PARK(menu_item)
        elif self.__cur_interface == Interfaces.MANUAL_CTRL:
            self.__menu_select_MANU(menu_item)
        elif self.__cur_interface == Interfaces.AUTO_CTRL:
            self.__menu_select_AUTO(menu_item)
        elif self.__cur_interface == Interfaces.STEREO_CAMERA:
            self.__menu_select_STEREO_CAMERA(menu_item)

    def __menu_select_ROOT(self, menu_item):
        caption = menu_item['caption']

        if caption == "Park robot":
            self.__enter_park_interface()
        elif caption == "Stereo-camera adjust":
            self.__enter_camera_interface()
        elif caption == "Start test":
            self.__enter_manu_interface()
        elif caption == "Start recording":
            menu_item['caption'] = "Stop recording"
            self.__recording = True
        elif caption == "Stop recording":
            menu_item['caption'] = "Start recording"
            self.__recording = False
        elif caption == "Take a picture":
            self.__take_picture = True
        elif caption == "Exit":
            self.boom_ctrl_thread.terminated = True
            self.boom_ctrl_thread.join()
            exit(0)

    def __menu_select_PARK(self, menu_item):
        caption = menu_item['caption']

        if caption == "Exit":
            self.__enter_root_interface()
        elif caption == "Stop robot":
            self.robot.stop_moving()
        else:
            pos_euler = None
            if caption == "Park for rest":
                pos_euler = FIXED_POS_EULERS['rest']
            elif caption == "Park for projector":
                pos_euler = FIXED_POS_EULERS['projector']
            elif caption == "Park for maintenance":
                pos_euler = FIXED_POS_EULERS['maintenance']

            if pos_euler is not None:
                if not self.robot.move_by_ground_coordinate(pos_euler):
                    self.__print("Failed to send robot order")

    def __menu_select_MANU(self, menu_item):
        caption = menu_item['caption']
        if caption == "Start auto-control":
            self.__enter_auto_interface()
        elif caption == "Stop test":
            self.__enter_root_interface()

    def __menu_select_AUTO(self, menu_item):
        caption = menu_item['caption']
        if caption == "Stop auto-control":
            self.__enter_manu_interface()

    def __menu_select_STEREO_CAMERA(self, menu_item):
        caption = menu_item['caption']
        if caption == "Exit":
            self.__enter_root_interface()

    def __enter_root_interface(self):
        # When the interface switches from manual test control to root, the test shall be stopped.
        # There are two possible cases:
        # 1. The robot already in EGM mode, quit EGM mode and testing thread.
        # 2. The robot not in EGM mode, still moving to the EGM starting position, then simply stop moving.
        if self.__cur_interface == Interfaces.MANUAL_CTRL and self.robot is not None:
            if self.robot.EGM:
                self.boom_ctrl_thread.stop_ctrl()
                self.robot.stop_EGM()
            else:
                self.robot.stop_moving()

        self.menu.clear()
        self.menu.add_item("Switch to stereo-camera" if self.__show_boom_eye else "Switch to depth-camera")
        self.menu.add_item("Park robot")
        self.menu.add_item("Start test")
        self.menu.add_item("Start recording")
        self.menu.add_item("Take a picture")
        self.menu.add_item("Exit")
        self.__cur_interface = Interfaces.ROOT

    def __interface_ROOT_operate(self):
        pass

    def __enter_park_interface(self):
        self.menu.clear()
        self.menu.add_item("Switch to stereo-camera" if self.__show_boom_eye else "Switch to depth-camera")
        self.menu.add_item("Park for rest")
        self.menu.add_item("Park for projector")
        self.menu.add_item("Park for maintenance")
        self.menu.add_item("Stop robot")
        self.menu.add_item("Exit")
        self.__cur_interface = Interfaces.PARK

    def __interface_park_operate(self):
        """
        if self.joystick.joystick.button[0]:
            self.robot.stop_moving()
            self.__print("Robot stopped!")
        """
        pass

    def __enter_camera_interface(self):
        self.menu.clear()
        self.menu.add_item("Switch to stereo-camera" if self.__show_boom_eye else "Switch to depth-camera")
        self.menu.add_item("Exit")

        self.__cur_interface = Interfaces.STEREO_CAMERA

    def __enter_manu_interface(self):
        # When entering into manual test interface, it shall start EGM mode first.
        # The robot is order to move to the EGM starting position before EGM starts, which costs some time.
        # Therefore it only starts moving the robot here, and checks its position repeatedly in __interface_manu_operate
        if self.__cur_interface == Interfaces.ROOT:
            if self.robot is not None and not self.robot.EGM:
                pos_euler = FIXED_POS_EULERS['EGM start']
                self.robot.move_by_ground_coordinate(pos_euler)

        self.menu.clear()
        self.menu.add_item("Switch to stereo-camera" if self.__show_boom_eye else "Switch to depth-camera")
        self.menu.add_item("Start auto-control")
        self.menu.add_item("Stop test")
        self.__cur_interface = Interfaces.MANUAL_CTRL

    def __robot_moving_to_EGM(self):
        diff = 0.0
        dest = FIXED_POS_EULERS['EGM start'][-1][0:6]
        curr = self.robot.pos_euler
        for i in range(len(dest)):
            diff += (dest[i] - curr[i]) * (dest[i] - curr[i])

        if math.sqrt(diff) < 1.0 and self.robot.start_EGM():
            time.sleep(0.5)
            self.boom_ctrl_thread.start_ctrl()

    def __robot_manual_control(self):
        # the actual robot motion control is by the boom control thread
        # here it just calculates motion speeds and send them to the thread
        spd_yaw = -self.joystick.yaw * 4
        spd_pitch = -self.joystick.pitch * 4

        self.boom_ctrl_thread.yaw_speed = spd_yaw
        self.boom_ctrl_thread.pitch_speed = spd_pitch
        self.boom_ctrl_thread.stretch_order = self.joystick.stretch

    def __coordinate_screen_to_ZED(self, sx, sy):
        resize_k = self.boom_eye.depth_camera.resolution[0] / self.screen_size[0]
        return int(sx * resize_k), int(sy * resize_k)

    def __coordinate_ZED_to_screen(self, cx, cy):
        resize_k = self.screen_size[0] / self.boom_eye.depth_camera.resolution[0]
        return int(cx * resize_k), int(cy * resize_k)

    def __interface_manu_operate(self):
        if self.robot is not None and not self.robot.EGM:
            self.__robot_moving_to_EGM()

        self.tracker.track(self.boom_eye.depth_camera.get_RGBimage())
        if self.boom_eye is not None and self.tracker.target is not None:
            self.__target_pos = self.boom_eye.get_hand_position(self.tracker.target[0], self.tracker.target[1])

    def __enter_auto_interface(self):
        self.menu.clear()
        self.menu.add_item("Switch to stereo-camera" if self.__show_boom_eye else "Switch to depth-camera")
        self.menu.add_item("Stop auto-control")

        self.__cur_interface = Interfaces.AUTO_CTRL

    def __interface_auto_operate(self):
        self.tracker.track(self.boom_eye.depth_camera.get_RGBimage())
        if self.tracker.target is not None:
            self.__target_pos = self.boom_eye.get_hand_position(self.tracker.target[0], self.tracker.target[1])

        # automatically control the boom pointing to the target
        if self.__target_pos is not None:
            dy = self.__target_pos[1] if math.fabs(self.__target_pos[1]) > BOOM_CTRL_MIN_ERROR else 0.0
            dz = self.__target_pos[2] if math.fabs(self.__target_pos[2]) > BOOM_CTRL_MIN_ERROR else 0.0

            spd_yaw = min(BOOM_ANGLE_SPD_MAX, max(-BOOM_ANGLE_SPD_MAX, dy * BOOM_CTRL_KP))
            spd_pitch = -min(BOOM_ANGLE_SPD_MAX, max(-BOOM_ANGLE_SPD_MAX, dz * BOOM_CTRL_KP))

            self.boom_ctrl_thread.yaw_speed = spd_yaw
            self.boom_ctrl_thread.pitch_speed = spd_pitch

    def __interface_operate(self):
        if self.__cur_interface == Interfaces.ROOT:
            self.__interface_ROOT_operate()
        elif self.__cur_interface == Interfaces.PARK:
            self.__interface_park_operate()
        elif self.__cur_interface == Interfaces.MANUAL_CTRL:
            self.__interface_manu_operate()
        elif self.__cur_interface == Interfaces.AUTO_CTRL:
            self.__interface_auto_operate()

    def __draw_tracker(self, img, target, color):
        if target is None:
            return

        target = self.__coordinate_ZED_to_screen(target[0], target[1])

        lt = (target[0] - 50, target[1] - 50)
        rb = (target[0] + 50, target[1] + 50)
        cv2.rectangle(img, lt, rb, color, 2)

        cv2.circle(img, target, 5, color, 1)
        cv2.line(img, (target[0] - 5, target[1]), (target[0] - 25, target[1]), color, 1)
        cv2.line(img, (target[0] + 5, target[1]), (target[0] + 25, target[1]), color, 1)
        cv2.line(img, (target[0], target[1] - 5), (target[0], target[1] - 25), color, 1)
        cv2.line(img, (target[0], target[1] + 5), (target[0], target[1] + 25), color, 1)

    def __draw_info(self, img, y_pos):
        if SHOW_MONITOR_INFO:
            cv2.putText(img, "Yaw = %1.1f, pitch = %1.1f, stretch = %1.0f" %
                        (self.boom_ctrl_thread.yaw, self.boom_ctrl_thread.pitch, self.boom_ctrl_thread.stretch),
                        (MENU_LEFT, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
            y_pos += 20

            cv2.putText(img, "main thread time = %1.3f, boom thread time = %1.3fms" %
                        ((time.time() - self.__lst_time) * 1000, self.boom_ctrl_thread.loop_time), (MENU_LEFT, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
            self.__lst_time = time.time()
            y_pos += 20

        if self.__target_pos is not None:
            cv2.putText(img, "Target position = (%4.0f, %4.0f, %4.0f)" % (self.__target_pos[0], self.__target_pos[1],
                                                                          self.__target_pos[2]),
                        (MENU_LEFT, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
            y_pos += 20

        return y_pos

    def __draw_gauge(self, img):
        color = (0, 255, 0)
        guage_h_half = int(self.screen_size[0] * 3 / 8)
        guage_v_half = int(self.screen_size[1] * 3 / 8)

        # the horizontal gauge
        left = int(self.screen_size[0] / 2) - guage_h_half
        right = int(self.screen_size[0] / 2) + guage_h_half
        count = GAUGE_SCALE_CNT * 2 + 1
        step = int((right - left) / (count - 1))

        for i in range(count):
            x = left + i * step
            cv2.line(img, (x, GAUGE_EDGE), (x, GAUGE_EDGE + GAUGE_WIDTH * 2), (0, 255, 255), 2)
            cv2.line(img, (x, self.screen_size[1] - GAUGE_EDGE), (x, self.screen_size[1] - GAUGE_EDGE - GAUGE_WIDTH * 2),
                     (0, 255, 255), 2)

        center = int((left + right) / 2)
        cv2.line(img, (center, GAUGE_EDGE), (center, GAUGE_EDGE + GAUGE_WIDTH * 2), (0, 0, 255), 2)
        cv2.line(img, (center, self.screen_size[1] - GAUGE_EDGE),
                 (center, self.screen_size[1] - GAUGE_EDGE - GAUGE_WIDTH * 2), (0, 0, 255), 2)

        cv2.rectangle(img, (left, GAUGE_EDGE), (right, GAUGE_EDGE + GAUGE_WIDTH), color, -1)
        cv2.rectangle(img, (left, self.screen_size[1] - GAUGE_EDGE),
                      (right, self.screen_size[1] - GAUGE_EDGE - GAUGE_WIDTH), color, -1)


        # the vertical gauge
        top = int(self.screen_size[1] / 2) - guage_v_half
        btm = int(self.screen_size[1] / 2) + guage_v_half
        left = GAUGE_EDGE

        step = int((btm - top) / (count - 1))
        for i in range(count):
            y = top + i * step
            cv2.line(img, (GAUGE_EDGE, y), (GAUGE_EDGE + GAUGE_WIDTH * 2, y), (0, 255, 255), 2)
            cv2.line(img, (self.screen_size[0] - GAUGE_EDGE, y), (self.screen_size[0] - GAUGE_EDGE - GAUGE_WIDTH * 2, y),
                     (0, 255, 255), 2)

        center = int((top + btm) / 2)
        cv2.line(img, (GAUGE_EDGE, center), (GAUGE_EDGE + GAUGE_WIDTH * 2, center), (0, 0, 255), 2)
        cv2.line(img, (self.screen_size[0] - GAUGE_EDGE, center),
                 (self.screen_size[0] - GAUGE_EDGE - GAUGE_WIDTH * 2, center),
                 (0, 0, 255), 2)

        cv2.rectangle(img, (10, top), (20, btm), color, -1)
        cv2.rectangle(img, (self.screen_size[0] - left, top),
                      (self.screen_size[0] - left - GAUGE_WIDTH, btm), color, -1)

    def __draw_indicator_h(self, img, x, color):
        top = GAUGE_EDGE + GAUGE_WIDTH
        guage_h_half = int(self.screen_size[0] * 3 / 8)
        x = int(guage_h_half * x + self.screen_size[0] / 2)
        w = 15
        t = int(w * 1.732)

        pts = np.array(((x, top), (x - w, top + t), (x + w, top + t)))
        cv2.fillPoly(img, [pts], color)

        pts = np.array(((x, self.screen_size[1] - top), (x - w, self.screen_size[1] - top - t),
                        (x + w, self.screen_size[1] - top - t)))
        cv2.fillPoly(img, [pts], color)

    def __draw_indicator_v(self, img, y, color):
        left = GAUGE_EDGE + GAUGE_WIDTH
        guage_v_half = int(self.screen_size[1] * 3 / 8)
        y = int(guage_v_half * y + self.screen_size[1] / 2)
        h = 15
        w = int(h * 1.732)

        pts = np.array(((left, y), (left + w, y - h), (left + w, y + h)))
        cv2.fillPoly(img, [pts], color)

        pts = np.array(((left, self.screen_size[0] - y), (left + w, self.screen_size[0] - y - h),
                        (left + w, self.screen_size[0] - y + h)))
        cv2.fillPoly(img, [pts], color)

    def __draw_position_cross(self, img, cx, cy, length, width, x, y, color):
        cv2.rectangle(img, (cx - length - width, cy - width), (cx - width, cy + width), color, 1)
        cv2.rectangle(img, (cx + length + width, cy - width), (cx + width, cy + width), color, 1)
        cv2.rectangle(img, (cx - width, cy - length - width), (cx + width, cy - width), color, 1)
        cv2.rectangle(img, (cx - width, cy + length + width), (cx + width, cy + width), color, 1)
        cv2.rectangle(img, (cx - width, cy - width), (cx + width, cy + width), (50, 50, 255), -1)

        if x < 0:
            cv2.rectangle(img, (cx + int(x * length) - width, cy - width), (cx - width, cy + width), color, -1)
        else:
            cv2.rectangle(img, (cx + int(x * length) + width, cy - width), (cx + width, cy + width), color, -1)

        if y < 0:
            cv2.rectangle(img, (cx - width, cy + int(y * length) - width), (cx + width, cy - width), color, -1)
        else:
            cv2.rectangle(img, (cx - width, cy + int(y * length) + width), (cx + width, cy + width), color, -1)

    def __draw_position_bar(self, img, bx, by, length, width, y, color):
        cv2.rectangle(img, (bx - width, by), (bx + width, by - length), color, 1)
        cv2.rectangle(img, (bx - width, by), (bx + width, by - int(length * y)), color, -1)

    def __draw_boom_status(self, img, yaw, pitch, stretch):
        cx = 200
        cy = 500
        max_length = 100
        line_width = 10
        color = (50, 255, 255)

        self.__draw_position_cross(img, cx, cy, max_length, line_width, -yaw, pitch, color)
        self.__draw_position_bar(img, cx - max_length - line_width * 4, cy + max_length, max_length * 2, line_width,
                                 stretch, color)
        self.__draw_position_bar(img, cx + max_length + line_width * 4, cy + max_length, max_length * 2, line_width,
                                 stretch, color)

        return img

    def __draw_info_strings(self, img, row_begin):
        for i, row in enumerate(self.__log_string):
            cv2.putText(img, row[0], (MENU_LEFT, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, row[1], 1)
            row_begin += 20
        return row_begin

    def __get_boom_eye_frame(self):
        if self.boom_eye is None:
            return None, None

        img_left = self.boom_eye.depth_camera.get_RGBimage()
        img_left = cv2.resize(img_left, self.screen_size)
        img_right = self.boom_eye.depth_camera.get_RGBimage_right()
        img_right = cv2.resize(img_right, self.screen_size)

        return img_left, img_right

    def __get_stereo_frame(self):
        if self.monitor_zed is None:
            return None, None

        self.monitor_zed.refresh()
        img_left = self.monitor_zed.get_RGBimage()
        img_left = cv2.resize(img_left, self.screen_size)
        img_right = self.monitor_zed.get_RGBimage_right()
        img_right = cv2.resize(img_right, self.screen_size)

        return img_left, img_right

    def __show_robot_status(self, img, row_begin):
        if self.robot is None:
            return row_begin

        if self.robot.EGM:
            cv2.putText(img, "Robot in EGM mode", (MENU_LEFT, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif self.robot.refresh_status():
            if self.robot.status < 0:
                cv2.putText(img, "Robot error: " + self.robot.error, (MENU_LEFT, row_begin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(img, "Robot status = (%4.0f, %4.0f, %4.0f, %3.0f, %3.0f, %3.0f)" % self.robot.pos_euler,
                        (MENU_LEFT, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        else:
            cv2.putText(img, "Robot is not connected!", (MENU_LEFT, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return row_begin + 20

    def __show_tracking_status(self, img, row_begin):
        if self.robot is not None and self.robot.EGM and self.__target_pos is not None:
            cv2.putText(img, "Target position = (%6.1f, %6.1f, %6.1f)" %
                        (self.__target_pos[0], self.__target_pos[1], self.__target_pos[2]),
                        (MENU_LEFT, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
            row_begin += 20
        return row_begin

    def __show_interface(self, img, left):
        if img is None:
            width = self.screen_size[0] * SCREEN_NUMBER
            img = np.zeros((self.screen_size[1], width, 3), dtype=np.uint8)

        self.__draw_gauge(img)
        if self.robot is not None and self.robot.EGM:
            # indicate boom angle
            self.__draw_indicator_h(img, x=-self.boom_ctrl_thread.yaw / 10.0, color=(50, 255, 255))
            self.__draw_indicator_v(img, y=self.boom_ctrl_thread.pitch / 10.0, color=(50, 255, 255))

            # indicate target position
            if left and self.__show_boom_eye:
                if SHOW_RAW_TRACKER:
                    self.__draw_tracker(img, target=self.tracker.raw_target, color=(20, 255, 255))
                self.__draw_tracker(img, target=self.tracker.target, color=(20, 255, 20))

            # indicate boom position against the target
            if self.__target_pos is not None:
                dist_x = min(1.0, max(-1.0, self.__target_pos[0] / 200.0))
                dist_y = min(1.0, max(-1.0, self.__target_pos[1] / 200.0))
                dist_z = min(1.0, max(-1.0, self.__target_pos[2] / 1000.0))
                self.__draw_indicator_h(img, dist_y, color=(50, 50, 255))
                self.__draw_indicator_v(img, dist_z, color=(50, 50, 255))
                #self.__draw_target_dist(img, dist_y, dist_z, dist_x)

        self.menu.refresh(img)
        row = self.__draw_info(img, 300)
        row = self.__show_robot_status(img, row)
        row = self.__show_tracking_status(img, row)
        row = self.__draw_info_strings(img, row)

        """
        self.__draw_boom_status(img, self.boom_ctrl_thread.yaw / 10.0,
                                self.boom_ctrl_thread.pitch / 10.0, self.boom_ctrl_thread.stretch / 1000.0)
        """
        return img

    def __video_record(self, img):
        if self.__recording:
            if self.__video is None:
                self.__video = cv2.VideoWriter(filename=time.strftime("videos\\%Y_%m_%d_%H_%M_%S.avi", time.localtime()),
                                               fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                                               fps=5,
                                               frameSize=self.screen_size)
            self.__video.write(img)
        elif self.__video is not None:
            self.__video.release()
            self.__video = None

    def __picture_record(self, img):
        if self.__take_picture:
            cv2.imwrite(filename=time.strftime("images\\%Y_%m_%d_%H_%M_%S.jpg", time.localtime()), img=img)
            self.__take_picture = False

    def __send_img_to_Uscreen(self, img):
        pass

    def refresh(self):
        time.sleep(0.01)

        # refresh ZED camera
        if self.boom_eye is not None:
            self.boom_eye.refresh()

        # refresh operations
        self.__menu_operate()
        self.__interface_operate()

        # show interface
        img_left, img_right = self.__get_boom_eye_frame() if self.__show_boom_eye else self.__get_stereo_frame()
        if img_left is not None:
            img_left = self.__show_interface(img=img_left, left=True)
            self.__video_record(img_left)
            self.img_trans.send(img_left)
        if img_right is not None:
            img_right = self.__show_interface(img=img_right, left=False)
        self.monitor.show(img_left, img_right)
        self.__send_img_to_Uscreen(img_left)

        if self.boom_eye is not None:
            self.__picture_record(self.boom_eye.depth_camera.get_RGBimage())


if __name__ == '__main__':
    proc_cnt = 0
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['name'])
        except psutil.NoSuchProcess:
            pass
        else:
            if pinfo['name'] == 'python':
                proc_cnt += 1

    if proc_cnt <= 1:
        boom = FlyingBoomCtrl()
        while True:
            boom.refresh()

