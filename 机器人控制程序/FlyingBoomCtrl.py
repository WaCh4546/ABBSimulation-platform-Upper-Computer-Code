import cv2
import math
import time
import numpy as np
import win32api,win32con
import threading
import pygame
import socket

from ABB_EGM import EGM
from RobotABB import RobotABB
from ZEDCamera import ZEDCamera
from JoyStick import BoomJoystick
from PlaneTracker import PlaneTracker
from RobotEye import RobotEye

# robot IP = 192.168.1.103

ROBOT_IP = "192.168.1.103"
ROBOT_PORT = 8002
REMOTE_IP = "192.168.1.101"
REMOTE_PORT = 5678
TWIN_SCREEN = True


class BoomCtrlThread(threading.Thread):
    def __init__(self, robot, joystick, lock):
        threading.Thread.__init__(self)
        self.robot = robot
        self.joystick = joystick
        self.lock = lock
        self.boom_len = 2000

        self.auto_tracking = False
        self.track_button = (0, 0)
        self.auto_track_speed = (0, 0, 0)

        # obtain the current position and gesture of the robot, calculate the virtual joint of the boom accordingly
        self.set_robot(robot)

        # set the initial boom state
        self.yaw = 0
        self.pitch = 0
        self.L = 0
        self.speed = (0, 0, 0)
        self.terminated = False

        # control range
        self.angle_range = 10
        self.angle_linear = 8
        self.L_range = 1000
        self.ctrl_interval = 0.02

        self.__lst_time = time.time()
        self.loop_time = 0

        self.remote = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
        x, y, z = self.__calc_boom_offset(robot_pos_euler[3], robot_pos_euler[4], robot_pos_euler[5], self.boom_len)
        return robot_pos_euler[0] - x, robot_pos_euler[1] - y, robot_pos_euler[2] - z

    def __sigmoid(self, x, lin_x, max_x):
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
        x += speed * self.ctrl_interval
        x = self.__sigmoid(x, lin_x, max_x)
        return x

    def __calc_pos_euler(self, speed):
        # accumulate angle speed to obtain boom euler angle
        # boom length is directly controlled, so it is simplly limited within range
        self.yaw = self.__speed_accumulate(speed[0], self.yaw, self.angle_linear, self.angle_range)
        self.pitch = self.__speed_accumulate(speed[1], self.pitch, self.angle_linear, self.angle_range)
        self.L = max(0, min(self.L_range, self.L + speed[2] * self.ctrl_interval))

        yaw = self.init_euler[0] + self.yaw
        pitch = self.init_euler[1] + self.pitch
        roll = self.init_euler[2]
        L = self.boom_len + self.L
        x, y, z = self.__calc_boom_offset(yaw, pitch, roll, L)
        return (x + self.virtual_joint_pos[0], y + self.virtual_joint_pos[1], z + self.virtual_joint_pos[2],
                yaw, pitch, roll)

    def set_robot(self, robot):
        self.lock.acquire()

        self.virtual_joint_pos = (0, 0, 0)
        self.init_euler = (0, 30, 0)
        try:
            if robot is not None:
                init_pos_euler = robot.get_robot_pos_euler()
                if init_pos_euler is None:
                    raise Exception("Failed to get position from robot")
                self.robot = robot
                self.virtual_joint_pos = self.__calc_virtual_joint_pos(init_pos_euler)
                self.init_euler = (init_pos_euler[3], init_pos_euler[4], init_pos_euler[5])
        except Exception as info:
            self.robot = None
            print(info)
        finally:
            self.lock.release()

        return self.robot is not None

    def __joystick_to_speed(self, yaw, pitch, L):
        spd_yaw = -yaw * 2
        spd_pitch = -pitch * 2
        spd_L = L * 80

        return spd_yaw, spd_pitch, spd_L

    def run(self):
        pygame.init()
        while not self.terminated:
            pygame.time.wait(int(self.ctrl_interval * 1000))

            # get joystick status
            manu_yaw, manu_pitch, manu_L, track_button, tracking = self.joystick.refresh()
            self.auto_tracking = tracking
            self.track_button = track_button

            # boom control
            if not self.auto_tracking:
                speed = self.__joystick_to_speed(manu_yaw, manu_pitch, manu_L)
            else:
                speed = self.auto_track_speed
            pos_euler = self.__calc_pos_euler(speed)

            self.lock.acquire()
            if self.robot is not None:
                self.robot.set_robot_pos_euler(pos_euler)
            str = "%1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f" % (pos_euler)
            self.remote.sendto(str.encode('utf-8'), (REMOTE_IP, REMOTE_PORT))
            self.lock.release()

            self.loop_time = (time.time() - self.__lst_time) * 1000
            self.__lst_time = time.time()


class FlyingBoomCtrl(object):
    def __init__(self):
        self.__init_hardwares()
        self.__init_screen()
        self.tracker = PlaneTracker("加油口定位2021-11-20-17-51.pt")

        self.manu_ctrl = True       # control boom by joystick
        self.show_boom_eye = True   # the image is captured by ZED on the boom, otherwise by the cameras on the platform

        self.__lst_time = time.time()
        self.__EGM_enabled = False

        self.lock = threading.RLock()
        self.boom_ctrl_thread = BoomCtrlThread(None, self.joystick, self.lock)
        self.boom_ctrl_thread.start()

    def __init_hardwares(self):
        try:
            self.robot = RobotABB(host=ROBOT_IP, port=ROBOT_PORT)
            if not self.robot.connected:
                raise Exception("Failed to connect robot!")

            if not self.robot.start():
                raise Exception("Failed to start robot!")
        except Exception as info:
            self.robot = None
            print(info)

        try:
            depth_camera = ZEDCamera()
            self.boom_eye = RobotEye(depth_camera=depth_camera, eye_name="FlyingBoom", on_hand=True)
        except Exception as info:
            self.boom_eye = None
            print("Failed to open ZED camera")

        try:
            self.joystick = BoomJoystick()
        except Exception as info:
            self.joystick = None
            print(info)

    def __init_screen(self):
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        if TWIN_SCREEN:
            screen_width = int(screen_width / 2)
        self.screen_size = (screen_width, screen_height)

        cv2.namedWindow('window', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('window', 0, 0)

    def __get_track_input(self):
        return 0, 0, 0

    def __draw_tracker(self, img, color):
        target = self.tracker.target
        lt = (target[0] - 50, target[1] - 50)
        rb = (target[0] + 50, target[1] + 50)
        cv2.rectangle(img, lt, rb, color, 2)

        cv2.circle(img, target, 5, color, 1)
        cv2.line(img, (target[0] - 5, target[1]), (target[0] - 25, target[1]), color, 1)
        cv2.line(img, (target[0] + 5, target[1]), (target[0] + 25, target[1]), color, 1)
        cv2.line(img, (target[0], target[1] - 5), (target[0], target[1] - 25), color, 1)
        cv2.line(img, (target[0], target[1] + 5), (target[0], target[1] + 25), color, 1)

    def __draw_menu(self, img, y_pos, color):
        if not self.__EGM_enabled:
            cv2.putText(img, "S: Start robot test", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, "Q: Quit", (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(img, "E: Stop robot test", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 40

        return y_pos

    def __draw_info(self, img, y_pos, color):
        cv2.putText(img, "Yaw = %1.1f, pitch = %1.1f, L = %1.0f" % (self.boom_ctrl_thread.yaw, self.boom_ctrl_thread.pitch, self.boom_ctrl_thread.L), (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 20

        cv2.putText(img, "main thread time = %1.3f, boom thread time = %1.3fms" % ((time.time() - self.__lst_time) * 1000, self.boom_ctrl_thread.loop_time), (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        self.__lst_time = time.time()
        y_pos += 20

        if self.robot is None:
            cv2.putText(img, "The robot is not connected", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_pos += 20
        if self.boom_eye is None:
            cv2.putText(img, "The depth camera is not connected", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_pos += 20
        if self.joystick is None:
            cv2.putText(img, "The joystick is not connected", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
            y_pos += 20

        return y_pos

    def __start_ctrl(self):
        if self.robot is None:
            print("Robot is not connected!")
            return

        if self.robot.start_EGM_at(pos=(4000, 0, -1200), euler=(0, 30, 0)):
            robot_EGM = EGM(PortNum=6511)
            self.__EGM_enabled = self.boom_ctrl_thread.set_robot(robot=robot_EGM)
        else:
            self.__EGM_enabled = False
            print("Failed to start robot for EGM ctrl")

    def __stop_ctrl(self):
        self.boom_ctrl_thread.terminated = True
        self.boom_ctrl_thread.join()
        self.robot.stop_EGM_at(pos=(4000, 0, -1200), euler=(0, 30, 0))

    def __key_select(self, key):
        if not self.__EGM_enabled:
            if key == ord('s') or key == ord('S'):
                self.__start_ctrl()
            elif key == ord('q') or key == ord('Q'):
                if self.boom_ctrl_thread is not None:
                    self.boom_ctrl_thread.terminated = True
                    self.boom_ctrl_thread.join()
                exit(0)
        else:
            if key == ord('e') or key == ord('E'):
                self.__stop_ctrl()

    def __get_boom_eye_frame(self):
        img_left = self.boom_eye.depth_camera.RGBimage
        img_left = cv2.resize(img_left, self.screen_size)

        if TWIN_SCREEN:
            img_right = self.boom_eye.depth_camera.RGBimage_right  # cv2.cvtColor(self.boom_eye.RGBimage_right, cv2.COLOR_BGR2GRAY)
            img_right = cv2.resize(img_right, self.screen_size)
            img_right = cv2.flip(img_right, 1)
            img = cv2.hconcat((img_left, img_right))
        else:
            img = img_left

        return img

    def __get_stereo_frame(self):
        return None

    def __show_interface(self, img):
        if img is None:
            if TWIN_SCREEN:
                img = np.zeros((self.screen_size[1], self.screen_size[0] * 2, 3))
            else:
                img = np.zeros((self.screen_size[1], self.screen_size[0], 3))

        self.__draw_tracker(img, color=(20, 255, 20))
        y_pos = self.__draw_menu(img, 20, (20, 255, 20))
        self.__draw_info(img, y_pos, (20, 255, 20))

        cv2.imshow("window", img)

    def refresh(self):
        # delay a short time and check if it is required to quit
        key = cv2.waitKeyEx(10)
        self.__key_select(key)

        # get image from ZED
        img_track = None
        if self.boom_eye is not None:
            self.boom_eye.refresh()
            img_track = cv2.cvtColor(self.boom_eye.depth_camera.RGBimage, cv2.COLOR_BGR2GRAY)

        # auto tracking plane
        if self.boom_ctrl_thread.auto_tracking:
            self.tracker.track(img_track)
        else:
            btn = self.boom_ctrl_thread.track_button
            self.tracker.target = (self.tracker.target[0] + int(btn[0] * 16), self.tracker.target[1] - int(btn[1] * 16))

        frame = None
        if self.show_boom_eye and self.boom_eye is not None:
            frame = self.__get_boom_eye_frame()
        else:
            frame = self.__get_stereo_frame()
        self.__show_interface(frame)


if __name__ == '__main__':
    boom = FlyingBoomCtrl()
    while True:
        boom.refresh()

