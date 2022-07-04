from ctypes import *   # for detail of ctypes visit this: https://docs.python.org/3/library/ctypes.html
import numpy as np
import sys


class HandEyeCaliFunc(object):
    def __init__(self):
        self.dll = windll.LoadLibrary(sys.path[0] + "\\HandEyeCali.dll")

    def __rotate_matrix_yaw_pitch_roll(self, euler):
        # the input euler angle uses degree as unit, which shall be translated into rad before sin and cos calculation.
        yaw = euler[0] * np.pi / 180.0
        pitch = euler[1] * np.pi / 180.0
        roll = euler[2] * np.pi / 180.0

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

    def __eye_to_hand(self, points, camera_pos, camera_euler):
        r_matrix = self.__rotate_matrix_yaw_pitch_roll(camera_euler)
        return (r_matrix @ points.T).T + camera_pos

    def __call__(self, init_pos_euler, eye_positions, hand_positions):
        # generate arrays to store pos_euler, eye_points and hand_points
        point_cnt = len(eye_positions)
        POS_EULER = c_double * 6
        EYE_POINTS = c_double * (3 * point_cnt)
        HAND_POINTS = c_double * (3 * point_cnt)

        # initialize arrays
        pos_euler = POS_EULER()
        eye_points = EYE_POINTS()
        hand_points = HAND_POINTS()

        for i in range(6):
            pos_euler[i] = init_pos_euler[i]

        for i in range(point_cnt):
            eye_points[i * 3] = eye_positions[i][0]
            eye_points[i * 3 + 1] = eye_positions[i][1]
            eye_points[i * 3 + 2] = eye_positions[i][2]

            hand_points[i * 3] = hand_positions[i][0]
            hand_points[i * 3 + 1] = hand_positions[i][1]
            hand_points[i * 3 + 2] = hand_positions[i][2]

        cali_func = self.dll.HandEyeCalibrate
        cali_func.restype = c_bool
        if cali_func(pointer(pos_euler), pointer(eye_points), pointer(hand_points), c_int(point_cnt)):
            return np.array((pos_euler[0], pos_euler[1], pos_euler[2])), \
                   np.array((pos_euler[3], pos_euler[4], pos_euler[5]))
        else:
            return None, None

    def test(self):
        print("given pos euler is")
        pos = np.random.rand(3) - 0.5
        euler = np.random.rand(3) * 20 - 10
        print(pos)
        print(euler)

        eye_points = np.random.rand(8, 3) * 100
        hand_points = self.__eye_to_hand(eye_points, pos, euler)

        print(eye_points)

        init_pos_euler = np.zeros(6)
        cali_pos, cali_euler = self.__call__(init_pos_euler, eye_points, hand_points)
        if cali_pos is not None:
            print("calculated pos euler is")
            print(cali_pos)
            print(cali_euler)

            cali_hand_points = self.__eye_to_hand(eye_points, cali_pos, cali_euler)

            for i in range(hand_points.shape[0]):
                print("real point in hand is")
                print(hand_points[i])

                print("transformed point in hand is")
                print(cali_hand_points[i])
        else:
            print("calibration failed")

if __name__ == '__main__':
    cali_func = HandEyeCaliFunc()
    cali_func.test()
