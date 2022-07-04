import cv2
import math
import time
import win32api,win32con
import threading
import pygame
import psutil
import numpy as np
from enum import Enum

from ABB_EGM import EGM
from RobotABB import RobotABB
from JoyStick import PlaneJoystick
from JoyStick import TestJoystick_Plane
from CVMenu import CVMenu
from ICCamera import ICCamera
from Axis7 import *
from Gyro import gyro_indicator


"""
2022.02.27: 1. CVMenu used
            2. Axis7 control added but not tested.
"""


ROBOT_IP = "192.168.1.104"
ROBOT_PORT = 8002

EGM_IP = ROBOT_IP
EGM_PORT = 6510

AXIS7_IP = "192.168.1.6"
#AXIS7_IP = "127.0.0.1"
AXIS7_PORT = 4196
AXIS7_LOCAL_PORT = 1234

TEST_IN_LAB = False
NO_ROBOT = False

euler_K = 0.6        # euler angle low-pass filter parameter. A smaller euler_K leads to faster rotation.
yaw_max = 10.0
pitch_max = 10.0
roll_max = 15.0

z_max = 250
y_max = 500

pitch_to_spd = 16 #8
roll_to_spd = 16 #8

FIXED_POS_EULERS = {
            'EGM start': [(2250, 0, 1250, 0, 0, 0, 100, 0)],
            'rest': [(2000, 0, 1200, 0, 0, 0, 100, 0)],
            'projector': [(2400, 0, -260, 0, 35, 0, 100, 0)],
            'maintenance': [(2300, 0, 280, 0, 35, 0, 100, 0)]
        }


class PlaneCtrlThread(threading.Thread):
    def __init__(self, lock):
        threading.Thread.__init__(self)
        self.robot = None
        self.lock = lock
        self.terminated = False

        # control range
        # the x movement is carried out by axis7, so it is left 0 here.
        self.max_range = (0, 500, 300, 10, 15, 15)
        self.lin_range = (0, 450, 250, 8, 13, 13)
        self.ctrl_interval = 0.01

        self.pos_euler_neu = self.pos_euler = (0, 0, 0, 0, 0, 0)
        self.input = (0, 0, 0)

        self.__lst_time = time.time()
        self.loop_time = 0

        self.euler_K = 0.6
        self.euler_to_spd = 16

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

    @staticmethod
    def __lowpass(x, lst_y, k):
        return lst_y * k + x * (1.0 - k)

    def __speed_accumulate(self, speed, cur_x, lin_x, max_x):
        cur_x += speed * self.ctrl_interval
        cur_x = self.__sigmoid(cur_x, lin_x, max_x)
        return cur_x

    def __calc_pos_euler(self, input):
        # translate joystick input into euler angles
        yaw = input[0] * yaw_max
        pitch = -input[1] * pitch_max
        roll = input[2] * roll_max

        # the euler angles are low-pass filtered such that they don't change too fast
        yaw = self.__lowpass(yaw, self.pos_euler[3], self.euler_K)
        pitch = self.__lowpass(pitch, self.pos_euler[4], self.euler_K)
        roll = self.__lowpass(roll, self.pos_euler[5], self.euler_K)

        # calculate y and z movement
        y_spd = -roll * self.euler_to_spd
        z_spd = -pitch * self.euler_to_spd

        y = self.__speed_accumulate(y_spd, self.pos_euler[1], y_max * 0.9, y_max)
        z = self.__speed_accumulate(z_spd, self.pos_euler[2], z_max * 0.9, z_max)

        return (0.0, y, z, yaw, pitch, roll)

    def start_ctrl(self):
        try:
            self.lock.acquire()

            robot = EGM(IP=EGM_IP, PortNum=EGM_PORT)
            init_pos_euler = robot.get_robot_pos_euler()
            if init_pos_euler is None:
                self.robot = None
                raise Exception("Failed to get position from robot")

            self.robot = robot
            self.pos_euler_neu = init_pos_euler
            self.pos_euler = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        finally:
            self.lock.release()

    def stop_ctrl(self):
        self.lock.acquire()
        self.robot = None
        self.lock.release()

    def run(self):
        while not self.terminated:
            time.sleep(self.ctrl_interval)

            self.lock.acquire()
            self.pos_euler = self.__calc_pos_euler(self.input)
            if self.robot is not None:
                pos_euler = (self.pos_euler[0] + self.pos_euler_neu[0],
                             self.pos_euler[1] + self.pos_euler_neu[1],
                             self.pos_euler[2] + self.pos_euler_neu[2],
                             self.pos_euler[3] + self.pos_euler_neu[3],
                             self.pos_euler[4] - self.pos_euler_neu[4],
                             self.pos_euler[5] + self.pos_euler_neu[5])
                self.robot.set_robot_pos_euler(pos_euler)
            self.lock.release()

            self.loop_time = (time.time() - self.__lst_time) * 1000
            self.__lst_time = time.time()


class Interfaces(Enum):
    ROOT = 0,
    PARK = 1,
    TEST = 2,
    AXIS7 = 3,
    SPEED = 4


class PlaneCtrl(object):
    def __init__(self):
        self.menu = CVMenu(left=20, top=20, width=250, height=25)
        self.__log_string = list()

        self.__init_hardwares()
        self.__init_screen()

        self.__EGM_enabled = False
        self.__recording = False
        self.__video = None
        self.__lst_time = time.time()

        self.lock = threading.RLock()
        self.plane_ctrl_thread = PlaneCtrlThread(self.lock)
        self.plane_ctrl_thread.start()

        self.__cur_interface = Interfaces.ROOT
        self.__enter_root_interface()

        self.__gyro = gyro_indicator(width=300)
        self.__recording = False

        try:
            spd = np.loadtxt("PlaneSpeed.npy")
            self.rotation_spd = spd[0]
            self.movement_spd = spd[1]
        except:
            self.rotation_spd = 0.6
            self.movement_spd = 10

    def __init_hardwares(self):
        # Initialize the joystick. Exit if no joystick exists
        try:
            pygame.init()
            self.joystick = PlaneJoystick() if not TEST_IN_LAB else TestJoystick_Plane()
        except Exception as info:
            self.joystick = None
            print(info)
            exit(0)

        # Initialize the front camera.
        self.camera = None
        try:
            dev_names = ICCamera.enum_names()
            if len(dev_names) == 0:
                print("The camera is not mounted!")
            else:
                self.camera = ICCamera(device_name=dev_names[0], dev_format="RGB32 (2048x1536)")
        except Exception as info:
            self.camera = None
            self.__print(info.args[0])

        # Initialize the robot
        if not TEST_IN_LAB and not NO_ROBOT:
            try:
                self.robot = RobotABB(host=ROBOT_IP, port=ROBOT_PORT)
                if not self.robot.connected:
                    raise Exception("Failed to connect robot!")
                if not self.robot.start():
                    raise Exception("Failed to start robot!")

                # clear error status
                time.sleep(1)
                self.robot.clear()
                time.sleep(1)
            except Exception as info:
                self.robot = None
                self.__print(info)
        else:
            self.robot = None

        # Initialize axis7
        self.axis7 = Axis7(local_port=AXIS7_LOCAL_PORT, remote_ip=AXIS7_IP, remote_port=AXIS7_PORT)

    def __init_screen(self):
        """
        Obtain the screen size, and initialize the cv window.
        :return: None
        """
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
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

        if self.__cur_interface == Interfaces.ROOT:
            self.__menu_select_ROOT(menu_item)
        elif self.__cur_interface == Interfaces.PARK:
            self.__menu_select_PARK(menu_item)
        elif self.__cur_interface == Interfaces.TEST:
            self.__menu_select_TEST(menu_item)
        elif self.__cur_interface == Interfaces.AXIS7:
            self.__menu_select_AXIS7(menu_item)
        elif self.__cur_interface == Interfaces.SPEED:
            self.__menu_select_SPEED(menu_item)

    def __menu_select_ROOT(self, menu_item):
        caption = menu_item['caption']

        if caption == "Park robot":
            self.__enter_park_interface()
        elif caption == "Start test":
            self.__enter_test_interface()
        elif caption == "Axis7 manual control":
            self.__enter_axis7_interface()
        elif caption == "Start recording":
            menu_item['caption'] = "Stop recording"
            self.__recording = True
        elif caption == "Stop recording":
            menu_item['caption'] = "Start recording"
            self.__recording = False
        elif caption == "Set moving speed":
            self.__enter_speed_interface()
        elif caption == "Exit":
            self.plane_ctrl_thread.terminated = True
            self.plane_ctrl_thread.join()
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
                print("projector")
            elif caption == "Park for maintenance":
                pos_euler = FIXED_POS_EULERS['maintenance']

            if pos_euler is not None:
                if not self.robot.move_by_ground_coordinate(pos_euler):
                    self.__print("Failed to send robot order")

    def __menu_select_TEST(self, menu_item):
        caption = menu_item['caption']
        if caption == "Exit":
            self.__enter_root_interface()

    def __menu_select_AXIS7(self, menu_item):
        caption = menu_item['caption']

        if caption == "Go zero":
            self.axis7.cmd = AXIS7_CMD.GO_ZERO
        elif caption == "Move to far position":
            self.axis7.cmd = AXIS7_CMD.MOVE_FWD
        elif caption == "Move to near position":
            self.axis7.cmd = AXIS7_CMD.MOVE_BKWD
        elif caption == "Stop":
            self.axis7.cmd = AXIS7_CMD.STOP
        elif caption == "Exit":
            self.axis7.cmd = AXIS7_CMD.STOP
            self.__enter_root_interface()

    def __menu_select_SPEED(self, menu_item):
        caption = menu_item['caption']

        if caption == "Increase rotation":
            self.rotation_spd = min(0.9, self.rotation_spd + 0.1)
            self.__print("Rotation speed = %1.1f" % self.rotation_spd, color=(0, 255, 0))
        elif caption == "Decrease rotation":
            self.rotation_spd = max(0.3, self.rotation_spd - 0.1)
            self.__print("Rotation speed = %1.1f" % self.rotation_spd, color=(0, 255, 0))
        elif caption == "Increase movement":
            self.movement_spd = min(24, self.movement_spd + 2)
            self.__print("Movement speed = %1.1f" % self.movement_spd, color=(0, 255, 0))
        elif caption == "Decrease movement":
            self.movement_spd = max(8, self.movement_spd - 2)
            self.__print("Movement speed = %1.1f" % self.movement_spd, color=(0, 255, 0))
        elif caption == "Exit":
            self.__enter_root_interface()

        np.savetxt("PlaneSpeed.npy", (self.rotation_spd, self.movement_spd))

    def __enter_root_interface(self):
        # When the interface switches from manual test control to root, the test shall be stopped.
        # There are two possible cases:
        # 1. The robot already in EGM mode, quit EGM mode and testing thread.
        # 2. The robot not in EGM mode, still moving to the EGM starting position, then simply stop moving.
        if self.__cur_interface == Interfaces.TEST and self.robot is not None:
            if self.robot.EGM:
                self.plane_ctrl_thread.stop_ctrl()
                self.robot.stop_EGM()
            else:
                self.robot.stop_moving()
        self.axis7.cmd = AXIS7_CMD.STOP

        self.menu.clear()
        self.menu.add_item("Park robot")
        self.menu.add_item("Axis7 manual control")
        self.menu.add_item("Start test")
        self.menu.add_item("Start recording") if not self.__recording else self.menu.add_item("Stop recording")
        self.menu.add_item("Set moving speed")
        self.menu.add_item("Exit")
        self.__cur_interface = Interfaces.ROOT

    def __enter_park_interface(self):
        self.menu.clear()
        self.menu.add_item("Park for rest")
        self.menu.add_item("Park for projector")
        self.menu.add_item("Park for maintenance")
        self.menu.add_item("Stop robot")
        self.menu.add_item("Exit")
        self.__cur_interface = Interfaces.PARK

    def __enter_axis7_interface(self):
        self.menu.clear()
        self.menu.add_item("Go zero")
        self.menu.add_item("Move to far position")
        self.menu.add_item("Move to near position")
        self.menu.add_item("Stop")
        self.menu.add_item("Exit")
        self.__cur_interface = Interfaces.AXIS7

    def __enter_test_interface(self):
        if self.__cur_interface == Interfaces.ROOT:
            if self.robot is not None and not self.robot.EGM:
                pos_euler = FIXED_POS_EULERS['EGM start']
                self.robot.move_by_ground_coordinate(pos_euler)

        self.menu.clear()
        self.menu.add_item("Exit")
        self.__cur_interface = Interfaces.TEST

    def __enter_speed_interface(self):
        self.menu.clear()
        self.menu.add_item("Increase rotation")
        self.menu.add_item("Decrease rotation")
        self.menu.add_item("Increase movement")
        self.menu.add_item("Decrease movement")
        self.menu.add_item("Exit")
        self.__cur_interface = Interfaces.SPEED

    def __interface_operate(self):
        if self.__cur_interface == Interfaces.ROOT:
            self.__interface_ROOT_operate()
        elif self.__cur_interface == Interfaces.PARK:
            self.__interface_park_operate()
        elif self.__cur_interface == Interfaces.TEST:
            self.__interface_test_operate()

    def __interface_ROOT_operate(self):
        pass

    def __interface_park_operate(self):
        pass

    def __interface_test_operate(self):
        if self.robot is not None:
            self.__robot_manual_control() if self.robot.EGM else self.__robot_moving_to_EGM()

    def __robot_moving_to_EGM(self):
        diff = 0.0
        dest = FIXED_POS_EULERS['EGM start'][-1][0:6]
        curr = self.robot.pos_euler
        for i in range(len(dest)):
            diff += (dest[i] - curr[i]) * (dest[i] - curr[i])

        if math.sqrt(diff) < 1.0 and self.robot.start_EGM():
            time.sleep(0.5)
            self.plane_ctrl_thread.start_ctrl()

    def __robot_manual_control(self):
        """
        speed = self.__joystick_input_to_speed(self.joystick.yaw, self.joystick.pitch,
                                               self.joystick.roll, self.joystick.throttle)
        self.plane_ctrl_thread.speed = speed
        """
        self.plane_ctrl_thread.euler_K = self.rotation_spd
        self.plane_ctrl_thread.euler_to_spd = self.movement_spd
        self.plane_ctrl_thread.input = (self.joystick.yaw, self.joystick.pitch, self.joystick.roll)

        if self.axis7.zeroed:
            self.axis7.cmd = AXIS7_CMD.NORMAL_CTRL
            self.axis7.speed = speed[0]
        else:
            self.axis7.cmd = AXIS7_CMD.STOP

    def __joystick_input_to_speed(self, yaw, pitch, roll, throttle):
        """
        So far we only use a very simple model here. It is necessary to replace with a better model.
        """
        spd_x = throttle * 5
        spd_yaw = yaw * 8
        spd_pitch = pitch * 5  #5
        spd_roll = roll * 5    #5
        spd_y = self.plane_ctrl_thread.pos_euler[5] * 10   #2
        spd_z = self.plane_ctrl_thread.pos_euler[4] * 10   #2
        return spd_x, spd_y, spd_z, spd_yaw, spd_pitch, spd_roll

    def __show_ctrl_status(self, img, row_begin):
        cv2.putText(img, "X = %1.1f, Y = %1.1f, Z = %1.0f" % (self.axis7.position,
                                                              self.plane_ctrl_thread.pos_euler[1],
                                                              self.plane_ctrl_thread.pos_euler[2]),
                    (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        row_begin += 20

        cv2.putText(img, "Yaw = %1.2f, Pitch = %1.2f, Roll = %1.2f" % (self.plane_ctrl_thread.pos_euler[3],
                                                                       self.plane_ctrl_thread.pos_euler[4],
                                                                       self.plane_ctrl_thread.pos_euler[5]),
                    (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        row_begin += 20

        """
        cv2.putText(img, "main thread time = %1.3fms, plane thread time = %1.3fms" %
                    ((time.time() - self.__lst_time) * 1000, self.plane_ctrl_thread.loop_time),
                    (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        self.__lst_time = time.time()
        row_begin += 20
        """

        return row_begin

    def __show_robot_status(self, img, row_begin):
        if self.robot is None:
            return row_begin

        if self.robot.EGM:
            cv2.putText(img, "Robot in EGM mode", (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
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

    def __show_axis7_status(self, img, row_begin):
        if self.axis7.timeout:
            cv2.putText(img, "Axis7 timeout!", (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            row_begin += 20
        elif self.axis7.zeroed:
            cv2.putText(img, "Axis7 not zeroed yet!", (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            row_begin += 20
        return row_begin

    def __draw_info_strings(self, img, row_begin):
        for i, row in enumerate(self.__log_string):
            cv2.putText(img, row[0], (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, row[1], 1)
            row_begin += 20
        return row_begin

    def __show_interface(self, img):
        img = cv2.resize(img, self.screen_size)

        self.menu.refresh(img)
        row = self.__show_ctrl_status(img, row_begin=220)
        row = self.__show_robot_status(img, row_begin=row)
        row = self.__show_axis7_status(img, row_begin=row)
        row = self.__draw_info_strings(img, row_begin=row)

        self.__gyro.draw_on_screen(pitch=-self.plane_ctrl_thread.pos_euler[4],
                                   roll=-self.plane_ctrl_thread.pos_euler[5],
                                   screen=img,
                                   left=img.shape[1] - 350,
                                   top=50)

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

    def refresh(self):
        time.sleep(0.05)

        # refresh joystick and operations
        self.joystick.refresh()
        #self.axis7.refresh()
        self.__menu_operate()
        self.__interface_operate()

        if self.camera is not None:
            frame = self.camera.snap()
            if frame is None:
                return
        else:
            frame = np.zeros((self.screen_size[1], self.screen_size[0], 3), dtype=np.uint8)
        frame = self.__show_interface(frame)

        if self.__recording:
            self.__video_record(frame)

        self.__video_record(frame)


def check_process_exist(proc_name):
    return False

    proc_cnt = 0
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=["pid", "name"])
        except psutil.NoSuchProcess:
            pass
        else:
            if pinfo["name"] == proc_name:
                proc_cnt += 1
    return proc_cnt > 1


if __name__ == '__main__':
    if not check_process_exist("python.exe"):
        plane = PlaneCtrl()
        while True:
            plane.refresh()
    else:
        print("PlaneCtrl cannot be opened repeatedly")

