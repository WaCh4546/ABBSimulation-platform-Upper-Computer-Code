import cv2
import math
import time
import numpy as np
import win32api, win32con
import threading
import pygame
import socket
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

"""
2022.02.22: The program is slightly fined.
2022.02.26: 1. Remote control of the binocular camera is added but not tested.
            2. Kalman filter is added in tracker but not tuned and tested.
"""


"""
constant values
"""
ROBOT_IP = "192.168.1.103"
ROBOT_PORT = 8002

EGM_IP = ROBOT_IP
EGM_PORT = 6511

REMOTE_IP = "192.168.1.2"
REMOTE_PORT = 5678

#BINOCULAR_IP = "192.168.1.5"
BINOCULAR_IP = "127.0.0.1"
BINOCULAR_PORT = 4196
BINOCULAR_LOCAL_PORT = 1234

TRACK_MODEL = "加油口定位2021-11-20-17-51.pt"

FIXED_POS_EULERS = {
            'EGM start': [(4000, 0, -1200, 0, 30, 0, 100, 0)],
            'rest': [(4880, 0, -320, 0, 0, 0, 100, 0)],
            'projector': [(4880, 0, -320, 0, 0, 0, 100, 0)],
            'maintenance': [(4000, -2240, -3000, -25, 28, 0, 100, 0)]
        }

BOOM_LENGTH = 2000
BOOM_STRETCH_RANGE = 1000
BOOM_ANGLE_RANGE = 10   # degree
BOOM_ANGLE_LINEAR = 8
BOOM_ANGLE_SPD_LMT = 0.1   # the minimum angle speed of the boom (deg/s)
BOOM_CTRL_INTERVAL = 0.02

BICAMERA_LEFT = ""
BICAMERA_RIGHT = ""

TEST_IN_LAB = True
NO_ROBOT = False
SHOW_MONITOR_INFO = True
SCREEN_NUMBER = 1


class BoomCtrlThread(threading.Thread):
    def __init__(self, lock):
        threading.Thread.__init__(self)
        self.lock = lock
        self.robot = None
        self.terminated = False

        # set the initial boom state
        self.yaw_speed = 0
        self.pitch_speed = 0
        self.stretch_speed = 0
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
        self.stretch = max(0, min(BOOM_STRETCH_RANGE, self.stretch + self.stretch_speed * BOOM_CTRL_INTERVAL))

        robot_yaw = self.init_euler[0] + self.yaw
        robot_pitch = self.init_euler[1] + self.pitch
        robot_roll = self.init_euler[2]
        robot_stretch = BOOM_LENGTH + self.stretch
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

            self.lock.acquire()
            if self.robot is not None:
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
        self.menu = CVMenu(left=20, top=20, width=250, height=25)

        self.__show_boom_eye = True    # Show images from ZED camera
        self.__auto_tracking = False
        self.__cur_interface = Interfaces.ROOT

        self.__lst_time = time.time()
        self.__target_pos = None
        self.__recording = False
        self.__video = None
        self.__take_picture = False

        self.__yaw_ctrl = PID(kp=0.01, ki=0, kd=0.01, T=0.15, K=0.1, type_inc=True)
        self.__pitch_ctrl = PID(kp=0.01, ki=0, kd=0.01, T=0.15, K=0.1, type_inc=True)

        self.lock = threading.RLock()
        self.boom_ctrl_thread = BoomCtrlThread(self.lock)
        self.boom_ctrl_thread.start()

        self.__enter_root_interface()

    def __init_hardwares(self):
        """
        When the application startS, initialize all the necessary hardwares first.
        If the joystick is not connected, the application exists.
        If other hardwares are not connected, the associated functions will be canceled.
        :return: None
        """
        try:
            pygame.init()
            self.joystick = BoomJoystick() if not TEST_IN_LAB else TestJoystick_Boom()
        except Exception as info:
            self.joystick = None
            print(info)
            exit(0)

        try:
            depth_camera = ZEDCamera()
            self.boom_eye = RobotEye(depth_camera=depth_camera, eye_name="FlyingBoom", on_hand=True)
        except Exception:
            self.boom_eye = None
            self.__print("Failed to open ZED camera")

        try:
            self.camera_l = ICCamera(BICAMERA_LEFT)
            self.camera_r = ICCamera(BICAMERA_RIGHT)
        except Exception as info:
            self.camera_l = None
            self.camera_r = None
            self.__print(info.args[0])

        if not TEST_IN_LAB and not NO_ROBOT:
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

        self.__bicamera_ctrl = BiCameraCtrl(local_port=BINOCULAR_LOCAL_PORT,
                                            remote_ip=BINOCULAR_IP, remote_port=BINOCULAR_PORT, timeout=0.1)

    def __init_screen(self):
        """
        Obtain the screen size, and initialize the cv window.
        :return: None
        """
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

        screen_width = int(screen_width / SCREEN_NUMBER)
        self.screen_size = (screen_width, screen_height)

        cv2.namedWindow('window', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('window', 0, 0)

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
        self.menu.add_item("Stereo-camera adjust")
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
        self.menu.add_item("Park for projector ")
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

    def __interface_camera_operate(self):
        trans = self.joystick.track_x
        rotate = self.joystick.track_y
        try:
            self.__bicamera_ctrl.send_order(trans, rotate)
        except socket.timeout:
            self.__print("Bi-camera PLC timeout")

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
        spd_stretch = self.joystick.stretch * 80

        self.boom_ctrl_thread.yaw_speed = spd_yaw
        self.boom_ctrl_thread.pitch_speed = spd_pitch
        self.boom_ctrl_thread.stretch_speed = spd_stretch

    def __coordinate_screen_to_ZED(self, sx, sy):
        resize_k = self.boom_eye.depth_camera.resolution[0] / self.screen_size[0]
        return int(sx * resize_k), int(sy * resize_k)

    def __coordinate_ZED_to_screen(self, cx, cy):
        resize_k = self.screen_size[0] / self.boom_eye.depth_camera.resolution[0]
        return int(cx * resize_k), int(cy * resize_k)

    def __interface_manu_operate(self):
        if self.robot is not None:
            self.__robot_manual_control() if self.robot.EGM else self.__robot_moving_to_EGM()

        # Switch between manu- and auto-tracking the target
        # When the tracking button on the joystick is clicked, if
        # 1. the target is not being tracked, then start tracking;
        # 2. the target is already being tracked, stop tracking.
        if self.joystick.track_click:
            self.__auto_tracking = True if not self.__auto_tracking and self.boom_eye is not None else False

        # Test cannot be stopped in auto-tracking.
        if self.__auto_tracking:
            self.menu.items[self.menu.get_item_idx("Start auto-control")]['enabled'] = True
            self.menu.items[self.menu.get_item_idx("Stop test")]['enabled'] = False
        else:
            self.menu.items[self.menu.get_item_idx("Start auto-control")]['enabled'] = False
            self.menu.items[self.menu.get_item_idx("Stop test")]['enabled'] = True

        # when auto-tracking the target, take the image from ZED camera and transport it to the tracker.
        # otherwise the target position is given by joystick.
        if self.__auto_tracking:
            img_track = cv2.cvtColor(self.boom_eye.depth_camera.RGBimage, cv2.COLOR_BGR2GRAY)
            self.tracker.track(img_track)
        else:
            self.tracker.target = (self.tracker.target[0] + int(self.joystick.track_x * 8),
                                   self.tracker.target[1] - int(self.joystick.track_y * 8))

        if self.boom_eye is not None:
            self.__target_pos = self.boom_eye.get_hand_position(self.tracker.target[0], self.tracker.target[1])

    def __enter_auto_interface(self):
        self.menu.clear()
        self.menu.add_item("Switch to stereo-camera" if self.__show_boom_eye else "Switch to depth-camera")
        self.menu.add_item("Stop auto-control")

        self.__cur_interface = Interfaces.AUTO_CTRL

    def __interface_auto_operate(self):
        # detect target position
        img_track = cv2.cvtColor(self.boom_eye.depth_camera.RGBimage, cv2.COLOR_BGR2GRAY)
        px, py = self.tracker.track(img_track)
        self.__target_pos = self.boom_eye.get_hand_position(px, py)

        # automatically control the boom pointing to the target
        if self.__target_pos is not None:
            #d_yaw = self.__yaw_ctrl(self.__target_pos[0])
            #d_pitch = self.__pitch_ctrl(self.__target_pos[1])

            #spd_yaw = -d_yaw / self.__yaw_ctrl.T
            #spd_pitch = -d_pitch / self.__pitch_ctrl.T

            dy = self.__target_pos[1] if math.fabs(self.__target_pos[1]) > 10.0 else 0.0
            dz = self.__target_pos[2] if math.fabs(self.__target_pos[2]) > 10.0 else 0.0

            spd_yaw = min(5, max(-5, dy * 0.05))
            spd_pitch = 0 #min(5, max(-5, dz * 0.01))
            spd_stretch = self.joystick.stretch * 80

            self.boom_ctrl_thread.yaw_speed = spd_yaw
            self.boom_ctrl_thread.pitch_speed = spd_pitch
            self.boom_ctrl_thread.stretch_speed = spd_stretch

    def __interface_operate(self):
        if self.__cur_interface == Interfaces.ROOT:
            self.__interface_ROOT_operate()
        elif self.__cur_interface == Interfaces.PARK:
            self.__interface_park_operate()
        elif self.__cur_interface == Interfaces.MANUAL_CTRL:
            self.__interface_manu_operate()
        elif self.__cur_interface == Interfaces.AUTO_CTRL:
            self.__interface_auto_operate()
        elif self.__cur_interface == Interfaces.STEREO_CAMERA:
            self.__interface_camera_operate()

    def __draw_tracker(self, img, color):
        target = self.__coordinate_ZED_to_screen(self.tracker.target[0], self.tracker.target[1])

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
                        (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
            y_pos += 20

            cv2.putText(img, "main thread time = %1.3f, boom thread time = %1.3fms" %
                        ((time.time() - self.__lst_time) * 1000, self.boom_ctrl_thread.loop_time), (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
            self.__lst_time = time.time()
            y_pos += 20

        if self.__target_pos is not None:
            cv2.putText(img, "Target position = (%4.0f, %4.0f, %4.0f)" % (self.__target_pos[0], self.__target_pos[1],
                                                                          self.__target_pos[2]),
                        (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
            y_pos += 20

        return y_pos

    def __draw_info_strings(self, img, row_begin):
        for i, string in enumerate(self.__log_string):
            cv2.putText(img, self.__log_string[i][0], (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, self.__log_string[i][1], 1)
            row_begin += 20
        return row_begin

    def __get_boom_eye_frame(self):
        if self.boom_eye is None:
            return None

        img_left = self.boom_eye.depth_camera.RGBimage
        img_left = cv2.resize(img_left, self.screen_size)

        if SCREEN_NUMBER > 1:
            img_right = self.boom_eye.depth_camera.RGBimage_right
            img_right = cv2.resize(img_right, self.screen_size)
            img_right = cv2.flip(img_right, 1)
            img = cv2.hconcat((img_left, img_right))
        else:
            img = img_left

        return img

    def __get_stereo_frame(self):
        img = None
        if self.camera_l is not None:
            img_l = self.camera_l.snap()
            img_r = self.camera_r.snap()
            if all((img_l, img_r)):
                img_l = cv2.resize(img_l, self.screen_size)
                img_r = cv2.resize(img_r, self.screen_size)
                img = cv2.hconcat((img_l, img_r))
        return img

    def __show_robot_status(self, img, row_begin):
        if self.robot is None:
            return row_begin

        if self.robot.EGM:
            cv2.putText(img, "Robot in EGM mode", (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        elif self.robot.refresh_status():
            if self.robot.status < 0:
                cv2.putText(img, "Robot error: " + self.robot.error, (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
            else:
                cv2.putText(img, "Robot status = (%4.0f, %4.0f, %4.0f, %3.0f, %3.0f, %3.0f)" % self.robot.pos_euler,
                        (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        else:
            cv2.putText(img, "Robot is not connected!", (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return row_begin + 20

    def __show_tracking_status(self, img, row_begin):
        if self.robot is not None and self.robot.EGM and self.__target_pos is not None:
            cv2.putText(img, "Target position = (%6.1f, %6.1f, %6.1f)" %
                        (self.__target_pos[0], self.__target_pos[1], self.__target_pos[2]),
                        (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
            row_begin += 20
        return row_begin

    def __show_interface(self, img):
        if img is None:
            width = self.screen_size[0] * SCREEN_NUMBER
            img = np.zeros((self.screen_size[1], width, 3), dtype=np.uint8)

        if self.robot is not None and self.robot.EGM and self.__show_boom_eye:
            self.__draw_tracker(img, color=(20, 255, 20))

        self.menu.refresh(img)
        row = self.__draw_info(img, 220)
        row = self.__show_robot_status(img, row)
        row = self.__show_tracking_status(img, row)
        row = self.__draw_info_strings(img, row)

        cv2.imshow("window", img)
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

    def refresh(self):
        time.sleep(0.01)

        # refresh ZED camera
        if self.boom_eye is not None:
            self.boom_eye.refresh()

        # refresh joystick and operations
        self.joystick.refresh()
        self.__menu_operate()
        self.__interface_operate()

        # show interface
        frame = self.__get_boom_eye_frame() if self.__show_boom_eye else self.__get_stereo_frame()
        frame = self.__show_interface(frame)
        self.__video_record(frame)

        if self.boom_eye is not None:
            self.__picture_record(self.boom_eye.depth_camera.RGBimage)


if __name__ == '__main__':
    boom = FlyingBoomCtrl()
    while True:
        boom.refresh()

