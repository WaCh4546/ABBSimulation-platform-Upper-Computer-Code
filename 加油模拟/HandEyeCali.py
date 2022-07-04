import cv2
import configparser  # see https://docs.python.org/3/library/configparser.html for detailed info
import ctypes
import csv
import numpy as np
import platform
import pyswarms as ps
import random
import math

from RobotEye import RobotEye
from ZEDCamera import ZEDCamera
from RobotABB import RobotABB

# configurations for calibration
NO_ROBOT = False
INIT_POS_EULER = (-400, 10, 0, -90, 0, -90)
ROBOT_IP = "192.168.1.103"
ROBOT_PORT = 8002
EYE_NAME = "FlyingBoom"
EYE_ON_HAND = True

"""
There are two kinds of calibration: eye off the hand, eye on the hand.

A. Eye Off the Hand
Calibrate eye-hand transformation when the eye is not fixed on the robot hand.
The calibration works as follows:
1. A cali-mark is fixed on the hand. The hand is in a gesture such that the mark is clearly visible for the camera.
2. Move the hand to a variety of positions. At each position, mouse-click the cali-mark on the screen to mark its 
   location on the image. Press key 'A' to obtain its coordinate in camera system, and the coordinate of the hand in
    robot system. The pair of coordinates are stored.
3. After enough samples are stored, press 'C' key to calculate (x, y, z, yaw, pitch, roll).

B. Eye On the Hand
Calibrate eye-hand transformation when the eye is fixed on the robot hand. 
The calibration works as follows:
1. A cali-mark is placed in space. 
2. Move the robot manually until the hand touches the cali-mark. Press key 'H' to obtain its coordinate in 
   robot system(denoted as P0), and its gesture(denoted as E).
3. Move the robot to other positions. The gesture of the hand must keep unchanged in moving. At each position,
   press key 'E' to obtain its coordinate in camera system(denoted as Pc) and in robot system (denoted as P1).
   Then press key 'A' to add (Pc, P0 - P1, E) into storage.
4. After enough samples are stored, press 'C' key to calculate (x, y, z, yaw, pitch, roll).

The program is finished in 2022.02.21.
Eye-on-hand calibration is tested in 2022.02.21.
Eye-off-hand calibration is not tested yet. 
"""


def on_mouse_event(event, x, y, flags, interface):
    if event == cv2.EVENT_LBUTTONDOWN:
        interface.on_mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        interface.on_mouse_up(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        interface.on_mouse_move(x, y)


def rotate_matrix(yaw, pitch, roll):
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


class PSO_Solver(object):
    def __init__(self, init_pos_euler, eye_samples, hand_samples):
        self.__eye_samples = np.array(eye_samples)
        self.__hand_samples = np.array(hand_samples)

        # generate initial particles
        particle_cnt = 100
        pos_range = 100
        euler_range = 30
        init = [init_pos_euler]
        for i in range(particle_cnt - 1):
            init.append(
                (init_pos_euler[0] + random.uniform(-pos_range, pos_range),
                 init_pos_euler[1] + random.uniform(-pos_range, pos_range),
                 init_pos_euler[2] + random.uniform(-pos_range, pos_range),
                 init_pos_euler[3] + random.uniform(-euler_range, euler_range),
                 init_pos_euler[4] + random.uniform(-euler_range, euler_range),
                 init_pos_euler[5] + random.uniform(-euler_range, euler_range))
            )
        init_pos = np.array(init)

        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
        optimizer = ps.single.GlobalBestPSO(n_particles=particle_cnt,
                                            dimensions=6,
                                            options=options,
                                            init_pos=init_pos)
        cost, pos_euler = optimizer.optimize(self.__opt_func, iters=500)

        self.cost = cost
        self.pos_euler = pos_euler

    @staticmethod
    def __calc_hand(pos_euler, eye_samples):
        rm = rotate_matrix(pos_euler[3], pos_euler[4], pos_euler[5])
        t = np.array((pos_euler[0], pos_euler[1], pos_euler[2]))
        return (rm @ eye_samples.T).T + t

    @staticmethod
    def __calc_cost(pos_euler, eye_samples, hand_samples):
        calc_hand = PSO_Solver.__calc_hand(pos_euler, eye_samples)
        diff = (hand_samples - calc_hand) ** 2
        cost = math.sqrt(np.sum(diff) / diff.shape[0])
        return cost

    def __opt_func(self, X):
        n_particles = X.shape[0]  # number of particles
        dist = [PSO_Solver.__calc_cost(X[i], self.__eye_samples, self.__hand_samples) for i in range(n_particles)]
        return np.array(dist)


class HandEyeCali(object):
    def __init__(self, robot_eye, robot):
        self.robot_eye = robot_eye
        self.robot = robot
        self.cali_name = robot_eye.eye_name if robot_eye is not None else "CaliTest"

        self.screen_width, self.screen_height = self.__get_window_size()
        self.screen_width = int(self.screen_width / 2)
        self.window_width = self.screen_width

        cv2.namedWindow(self.cali_name, cv2.WND_PROP_FULLSCREEN)
        cv2.resizeWindow(self.cali_name, self.screen_width, self.screen_height)
        cv2.setMouseCallback(self.cali_name, on_mouse_event, self)

        # the initial position and gesture of the eye, which will be used in calibration calculating
        # the calculating result is stored in eye_pos and eye_euler
        self.init_pos_euler = None
        self.eye_pos = None
        self.eye_euler = None

        self.eye_pos = np.array((-524.965, 60.6973, 91.0749))
        self.eye_euler = np.array((-90.7107, -2.38861, -92.3569))

        # samples of eye and hand stored for calibration calculating
        self.cali_samples = list()
        self.__eye_sample = None
        self.__hand_sample1 = None
        self.__hand_sample2 = None
        self.__hand_sample = None

        # the position in pixel of the marked calibrating point
        self.eye_cali_pos = (0, 0)
        self.cur_robot_axis = 0
        self.robot_step_mm = 10
        self.robot_step_deg = 1

        # log string
        self.__log_string = list()

    @staticmethod
    def __get_window_size():
        if platform.system() == "Windows":
            winapi = ctypes.windll.user32
            return winapi.GetSystemMetrics(0), winapi.GetSystemMetrics(1)
        elif platform.system() == "Linux":
            pass

    def __print(self, string):
        self.__log_string.append(string)
        if len(self.__log_string) > 6:
            del self.__log_string[0]

    def __restart_cali(self):
        self.cali_samples.clear()
        self.__print("Restart sampling")

    def __get_eye_sample(self, repeat=1):
        eye_pos = None
        if self.robot_eye.depth_camera is not None:
            resize_k = self.robot_eye.depth_camera.resolution[0] / self.window_width
            px = int(resize_k * self.eye_cali_pos[0] + 0.5)
            py = int(resize_k * self.eye_cali_pos[1] + 0.5)
            eye_pos = self.robot_eye.get_eye_position(px, py, repeat=repeat)
        return eye_pos

    def __get_hand_sample(self):
        hand_pos_euler = None
        if self.robot is not None and self.robot.refresh_status():
            hand_pos_euler = self.robot.pos_euler
        return hand_pos_euler

    def __check_calibration(self, eye_samples, hand_samples, pos, euler):
        eye_samples = np.array(eye_samples)
        hand_samples = np.array(hand_samples)

        hand_calculation = self.robot_eye.eye_to_hand(eye_samples, pos, euler)
        diff = hand_calculation - hand_samples

        norm = np.linalg.norm(diff, axis=1)
        return np.max(norm), np.min(norm), np.average(norm)

    def __calibrate(self):
        if len(self.cali_samples) < 6:
            self.__print("Short of samples, add more!")
            return

        if self.init_pos_euler is None:
            self.__print("Init position and gesture better be specified!")
            init_pos_euler = (0, 0, 0, 0, 0, 0)
        else:
            init_pos_euler = self.init_pos_euler

        # prepare the calibrating samples
        eye_samples = list()
        hand_samples = list()
        for _, sample in enumerate(self.cali_samples):
            eye_samples.append(sample[0])
            hand_samples.append(sample[1])
            if self.robot_eye.on_hand:
                r_matrix = rotate_matrix(sample[2][0], sample[2][1], sample[2][2])
                hand_samples[-1] = r_matrix.T @ hand_samples[-1]

        # calibrate
        cali = PSO_Solver(init_pos_euler, eye_samples, hand_samples)
        self.eye_pos = (cali.pos_euler[0], cali.pos_euler[1], cali.pos_euler[2])
        self.eye_euler = (cali.pos_euler[3], cali.pos_euler[4], cali.pos_euler[5])
        self.__print("Calibrated result: %1.1f, %1.1f, %1.1f, %1.1f, %1.1f, %1.1f" %
                     (self.eye_pos[0], self.eye_pos[1], self.eye_pos[2],
                      self.eye_euler[0], self.eye_euler[1], self.eye_euler[2]))

        # check calibration
        max_diff, min_diff, avg_diff = \
            self.__check_calibration(eye_samples, hand_samples, self.eye_pos, self.eye_euler)
        self.__print("Calibration error: max = %1.2f, min = %1.2f, avg = %1.2f" % (max_diff, min_diff, avg_diff))

        self.save_cali_result()

    def save_cali_result(self):
        cf = configparser.ConfigParser()

        cf.add_section("TransParams")
        cf["TransParams"]["trans_x"] = str(self.eye_pos[0])
        cf["TransParams"]["trans_y"] = str(self.eye_pos[1])
        cf["TransParams"]["trans_z"] = str(self.eye_pos[2])
        cf["TransParams"]["trans_yaw"] = str(self.eye_euler[0])
        cf["TransParams"]["trans_pitch"] = str(self.eye_euler[1])
        cf["TransParams"]["trans_roll"] = str(self.eye_euler[2])

        with open(self.cali_name + ".ini", 'w') as configfile:
            cf.write(configfile)

        self.__print("Calibration result saved")

    def save_cali_samples(self, file_name):
        file = open(file_name, 'w', newline='')
        writer = csv.writer(file)
        for i in range(len(self.cali_samples)):
            row = (str(self.cali_samples[i][0][0]), str(self.cali_samples[i][0][1]), str(self.cali_samples[i][0][2]),
                   str(self.cali_samples[i][1][0]), str(self.cali_samples[i][1][1]), str(self.cali_samples[i][1][2]),
                   str(self.cali_samples[i][2][0]), str(self.cali_samples[i][2][1]), str(self.cali_samples[i][2][2]))
            writer.writerow(row)
        self.__print("Calibration samples saved in " + file_name)

    def load_cali_samples(self, file_name):
        file = open(file_name, 'r')
        reader = csv.reader(file)

        self.cali_samples.clear()
        for row in reader:
            eye_point = (float(row[0]), float(row[1]), float(row[2]))
            hand_pos = (float(row[3]), float(row[4]), float(row[5]))
            hand_euler = (float(row[6]), float(row[7]), float(row[8]))
            self.cali_samples.append((eye_point, hand_pos, hand_euler))
        self.__print("Calibration samples loaded from " + file_name)

    def __robot_ctrl(self, key):
        # select an axis
        if ord('0') <= key <= ord('6'):
            self.cur_robot_axis = int(key - ord('0'))
        elif key == ord('x') or key == ord('X'):
            self.cur_robot_axis = 7
        elif key == ord('y') or key == ord('Y'):
            self.cur_robot_axis = 8
        elif key == ord('z') or key == ord('Z'):
            self.cur_robot_axis = 9

        # increase or decrease movement step
        elif key == ord('+'):
            if 1 <= self.cur_robot_axis <= 6 or 10 <= self.cur_robot_axis <= 12:
                self.robot_step_deg = min(self.robot_step_deg + 1, 5)
            else:
                self.robot_step_mm = min(self.robot_step_mm + 1, 100)
        elif key == ord('-'):
            if 1 <= self.cur_robot_axis <= 6 or 10 <= self.cur_robot_axis <= 12:
                self.robot_step_deg = max(self.robot_step_deg - 1, 1)
            else:
                self.robot_step_mm = max(self.robot_step_mm - 1, 1)

        # control robot position
        if self.robot is None or not self.robot.connected:
            return

        # control robot joints
        if 1 <= self.cur_robot_axis <= 6:
            joint = [0, 0, 0, 0, 0, 0, 100, 0.5]
            if key == 2490368:
                joint[self.cur_robot_axis - 1] = 1
                self.robot.move_by_joint([joint])
            elif key == 2621440:
                joint[self.cur_robot_axis - 1] = -1
                self.robot.move_by_joint([joint])
        # control robot position
        elif 7 <= self.cur_robot_axis <= 9:
            move = [0, 0, 0, 0, 0, 0, 100, 0.5]
            if key == 2490368:
                move[self.cur_robot_axis - 7] = self.robot_step_mm
                self.robot.move_by_TCP_coordinate([move])
            elif key == 2621440:
                move[self.cur_robot_axis - 7] = -self.robot_step_mm
                self.robot.move_by_TCP_coordinate([move])
        # control robot gesture
        elif 10 <= self.cur_robot_axis <= 12:
            move = [0, 0, 0, 0, 0, 0, 0, 0.1]
            if key == 2490368:
                move[self.cur_robot_axis - 1] = 1
                self.robot.move_by_TCP_coordinate([move])
            elif key == 2621440:
                move[self.cur_robot_axis - 1] = -1
                self.robot.move_by_TCP_coordinate([move])

    def __key_act_eye_on_hand(self, key):
        # In eye-on-hand mode
        # 1. The robot touches a cali-target first. Click 'h' to obtain the pos and euler
        #    of the robot, and store it in __hand_sample2.
        # 2. Then the robot moves to a positions randomly. At each position, click left key of the mouse on the target
        #    to mark the target. Click 'e' to obtain the pos and euler of the robot, and the pos of the target
        #    in the camera. They are stored in __hand_sample1 and __eye_sample.
        # 3. After that, click 'a' to add a cali-sample
        #           (__eye_sample, __hand_sample2.pos - __hand_sample1.pos, __hand_sample1.euler),
        #    which is appended to eye_samples and hand_samples.
        # 4. Repeat step 2 for more samples.
        if key == ord('e') or key == ord('E'):
            if self.robot_eye.depth_camera is not None:
                self.__eye_sample = self.__get_eye_sample(repeat=11)
                if self.__eye_sample is None:
                    self.__print("Failed to get eye coordinate!")
                self.__hand_sample1 = self.__get_hand_sample()
                if self.__hand_sample1 is None:
                    self.__print("Failed to get robot coordinate!")

        # get hand position from robot
        elif key == ord('h') or key == ord('H'):
            self.__hand_sample2 = self.__get_hand_sample()
            if self.__hand_sample2 is None:
                self.__print("Failed to get robot coordinate!")

        # add a pair of hand and eye samples
        elif key == ord('a') or key == ord('A'):
            if self.__eye_sample is not None and self.__hand_sample1 is not None and self.__hand_sample2 is not None:
                v_hand = (self.__hand_sample2[0] - self.__hand_sample1[0],
                          self.__hand_sample2[1] - self.__hand_sample1[1],
                          self.__hand_sample2[2] - self.__hand_sample1[2])
                euler_hand = (self.__hand_sample1[3], self.__hand_sample1[4], self.__hand_sample1[5])
                self.cali_samples.append((self.__eye_sample, v_hand, euler_hand))
                self.__hand_sample1 = self.__eye_sample = None
            else:
                self.__print("No sufficient eye and hand coordinates!")

    def __key_act_eye_off_hand(self, key):
        # In eye-off-hand mode, install a cali-target on the robot, move the robot to various positions randomly.
        # At each position, click left key of the mouse on the target. Click 'a' to get pos of the robot and the camera
        # and store them in cali_samples.
        if key == ord('a') or key == ord('A'):
            self.__eye_sample = self.__get_eye_sample(repeat=11)
            if self.__eye_sample is None:
                self.__print("Failed to get eye coordinate!")

            self.__hand_sample1 = self.__get_hand_sample()
            if self.__hand_sample1 is None:
                self.__print("Failed to get robot coordinate!")

            if self.__eye_sample is not None and self.__hand_sample1 is not None:
                self.cali_samples.append((self.__eye_sample,
                                          (self.__hand_sample1[0], self.__hand_sample1[1], self.__hand_sample1[2]),
                                          (self.__hand_sample1[3], self.__hand_sample1[4], self.__hand_sample1[5])))
            else:
                self.__print("No sufficient eye and hand coordinates!")

    def __key_act(self, key):
        self.__robot_ctrl(key)
        self.__key_act_eye_on_hand(key) if self.robot_eye.on_hand else self.__key_act_eye_off_hand(key)

        if key == ord('n') or key == ord('N'):
            self.__restart_cali()
        elif key == ord('c') or key == ord('C'):
            self.__calibrate()
        elif key == ord('s') or key == ord('S'):
            self.save_cali_samples(self.cali_name + ".csv")
        elif key == ord('l') or key == ord('L'):
            self.load_cali_samples(self.cali_name + ".csv")
        elif key == ord('q') or key == ord('Q'):
            exit(0)

    def __draw_basic_menu(self, img, row_begin, color):
        menu = ("C: Calibrate and save result",
                "S: Save calibrating samples",
                "L: Load calibrating samples",
                "Q: Quit")

        for _, row in enumerate(menu):
            cv2.putText(img, row, (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            row_begin += 20

        return row_begin

    def __draw_eye_off_hand_menu(self, img, row_begin, color):
        menu = ("X, Y, Z: Robot X, Y, Z coordinate control;",
                "1~6: Robot joint 1~6 control;",
                "Up/Dn: Inc/Dec robot movement;"
                "+/-: Inc/Dec movement step",
                "N: New calibration",
                "A: Add a hand-eye sample pair"
                )

        for _, row in enumerate(menu):
            cv2.putText(img, row, (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            row_begin += 20
        row_begin += 20

        return row_begin

    @staticmethod
    def __draw_eye_on_hand_menu(img, row_begin, color):
        menu = ("X, Y, Z: Robot X, Y, Z coordinate control; 1~6: Robot joint 1~6 control",
                "Up/Dn: Inc/Dec robot movement; +/-: Inc/Dec movement step",
                "N: New calibration",
                "E: Eye coordinate",
                "H: Hand coordinate",
                "A: Add a hand-eye sample pair"
                )
        for _, row in enumerate(menu):
            cv2.putText(img, row, (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            row_begin += 20

        return row_begin

    def __draw_menu(self, img, row_begin, color):
        row_begin = self.__draw_basic_menu(img, row_begin, color)

        if self.robot_eye.depth_camera is not None:
            if self.robot_eye.on_hand:
                row_begin = self.__draw_eye_on_hand_menu(img, row_begin, color)
            else:
                row_begin = self.__draw_eye_off_hand_menu(img, row_begin, color)

        return row_begin

    def __draw_robot_info(self, img, row_begin, color):
        if self.robot is None or not self.robot.connected:
            cv2.putText(img, "Robot is not connected!", (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return row_begin + 20

        axis = ("None", "J1", "J2", "J3", "J4", "J5", "J6", "X", "Y", "Z", "Yaw", "Pitch", "Roll")
        cv2.putText(img, "Controlling robot " + axis[self.cur_robot_axis] + " axis", (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)
        row_begin += 20

        cv2.putText(img, "Movement step = (%1.1fmm, %1.1fdeg)" % (self.robot_step_mm, self.robot_step_deg),
                    (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        row_begin += 20

        self.robot.refresh_status()
        pos_euler = self.robot.pos_euler
        if pos_euler is not None:
            cv2.putText(img, "Robot position & gesture = (%1.1f, %1.1f, %1.1f, %1.1f, %1.1f, %1.1f)" %
                        pos_euler, (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(img, "Current hand position and gesture is None", (20, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        row_begin += 20

        return row_begin

    def __draw_eye_info(self, img, row_begin, color):
        if self.robot_eye.depth_camera is None:
            cv2.putText(img, "Robot eye is not opened", (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return row_begin + 20

        # show cali position
        resize_k = self.robot_eye.depth_camera.resolution[0] / self.window_width
        px = int(resize_k * self.eye_cali_pos[0] + 0.5)
        py = int(resize_k * self.eye_cali_pos[1] + 0.5)
        cali_pos = self.robot_eye.get_eye_position(px, py)
        if cali_pos is None:
            cv2.putText(img, "Current eye position is None", (20, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cali_pos = (cali_pos[0], cali_pos[1], cali_pos[2])
            cv2.putText(img, "Current eye position: %3.0f, %3.0f, %3.0f" % cali_pos, (20, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        row_begin += 20

        if cali_pos is not None and self.eye_pos is not None and self.eye_euler is not None:
            cali_pos = np.array(cali_pos)
            hand_pos = self.robot_eye.eye_to_hand(points=cali_pos, camera_pos=self.eye_pos, camera_euler=self.eye_euler)
            cv2.putText(img, "Translated robot position: %3.0f, %3.0f, %3.0f" % (hand_pos[0], hand_pos[1], hand_pos[2]), (20, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        row_begin += 20

        # draw sample point
        cv2.circle(img, self.eye_cali_pos, 5, (0, 255, 0), 1)
        cv2.line(img, (self.eye_cali_pos[0] - 5, self.eye_cali_pos[1]), (self.eye_cali_pos[0] - 25, self.eye_cali_pos[1]),
                 (0, 255, 0), 1)
        cv2.line(img, (self.eye_cali_pos[0] + 5, self.eye_cali_pos[1]), (self.eye_cali_pos[0] + 25, self.eye_cali_pos[1]),
                 (0, 255, 0), 1)
        cv2.line(img, (self.eye_cali_pos[0], self.eye_cali_pos[1] - 5), (self.eye_cali_pos[0], self.eye_cali_pos[1] - 25),
                 (0, 255, 0), 1)
        cv2.line(img, (self.eye_cali_pos[0], self.eye_cali_pos[1] + 5), (self.eye_cali_pos[0], self.eye_cali_pos[1] + 25),
                 (0, 255, 0), 1)

        return row_begin

    def __draw_cali_info(self, img, row_begin, color):
        #if any([self.robot_eye.depth_camera, self.robot_eye.on_hand]):
        #    return row_begin

        if self.__eye_sample is None:
            cv2.putText(img, "Current eye sample is None", (20, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            eye = (self.__eye_sample[0], self.__eye_sample[1], self.__eye_sample[2])
            cv2.putText(img, "Current eye sample is (%1.1f, %1.1f, %1.1f)" % eye, (20, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        row_begin += 20

        if self.__hand_sample1 is None:
            cv2.putText(img, "Current hand sample1 is None", (20, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(img, "Current hand sample1 is (%1.1f, %1.1f, %1.1f, %1.1f, %1.1f, %1.1f)"
                        % self.__hand_sample1, (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        row_begin += 20

        if self.__hand_sample2 is None and self.robot_eye.on_hand:
            cv2.putText(img, "Current hand sample2 is None", (20, row_begin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(img, "Current hand sample2 is (%1.1f, %1.1f, %1.1f, %1.1f, %1.1f, %1.1f)" %
                        self.__hand_sample2, (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        row_begin += 40

        return row_begin

    def __draw_info_strings(self, img, row_begin, color):
        for i, string in enumerate(self.__log_string):
            cv2.putText(img, self.__log_string[i], (20, row_begin), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)
            row_begin += 20
        return row_begin

    def __draw_interface(self, img):
        row_begin = 20
        row_begin = self.__draw_menu(img, row_begin, (0, 255, 255))
        row_begin = self.__draw_robot_info(img, row_begin, (0, 255, 0))
        row_begin = self.__draw_eye_info(img, row_begin, (0, 255, 0))
        row_begin = self.__draw_cali_info(img, row_begin, (255, 255, 0))
        row_begin = self.__draw_info_strings(img, row_begin, (0, 0, 255))
        return row_begin

    def refresh(self):
        if self.robot_eye.depth_camera is not None:
            self.robot_eye.refresh()
            img = self.robot_eye.depth_camera.get_RGBimage(width=self.window_width, mark_infeasible=True)
        else:
            img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.int8)

        cv2.rectangle(img, (0, 0), (600, 350), (0, 0, 0), thickness=-1)
        self.__draw_interface(img)
        cv2.imshow(self.cali_name, img)

        key = cv2.waitKeyEx(50)
        self.__key_act(key)

    def on_mouse_move(self, px, py):
        pass

    def on_mouse_down(self, px, py):
        self.eye_cali_pos = (px, py)

    def on_mouse_up(self, x, y):
        pass


if __name__ == '__main__':
    camera = None
    try:
        camera = ZEDCamera()
    except Exception:
        print("Failed to open the camera!")
    robot_eye = RobotEye(depth_camera=camera, eye_name=EYE_NAME, on_hand=EYE_ON_HAND)

    robot = None
    if not NO_ROBOT:
        robot = RobotABB(host=ROBOT_IP, port=ROBOT_PORT)
        robot.start()

    cali = HandEyeCali(robot_eye=robot_eye, robot=robot)
    cali.init_pos_euler = INIT_POS_EULER

    while True:
        cali.refresh()