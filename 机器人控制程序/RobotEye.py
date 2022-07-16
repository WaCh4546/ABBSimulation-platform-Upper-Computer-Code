import numpy as np
import cv2
import configparser  # see https://docs.python.org/3/library/configparser.html for detailed info
from ZEDCamera import ZEDCamera


def on_mouse_event(event, x, y, flags, interface):
    if event == cv2.EVENT_LBUTTONDOWN:
        interface.on_mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        interface.on_mouse_up(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        interface.on_mouse_move(x, y)


class RobotEye(object):
    def __init__(self, depth_camera, eye_name, on_hand):
        self.eye_name = eye_name
        self.depth_camera = depth_camera
        self.on_hand = on_hand

        if eye_name is None:
            self.camera_pos = np.zeros(3)
            self.camera_euler = np.zeros(3)
        else:
            self.camera_pos, self.camera_euler = self.__load_trans_params(eye_name)

    @staticmethod
    def __load_trans_params(eye_name):
        cf = configparser.ConfigParser()
        cf.read(eye_name + ".ini")

        trans_x = cf.getfloat("TransParams", "trans_x", fallback=0.0)
        trans_y = cf.getfloat("TransParams", "trans_y", fallback=0.0)
        trans_z = cf.getfloat("TransParams", "trans_z", fallback=0.0)
        trans_yaw = cf.getfloat("TransParams", "trans_yaw", fallback=0.0)
        trans_pitch = cf.getfloat("TransParams", "trans_pitch", fallback=0.0)
        trans_roll = cf.getfloat("TransParams", "trans_roll", fallback=0.0)

        return np.array((trans_x, trans_y, trans_z)), np.array((trans_yaw, trans_pitch, trans_roll))

    @staticmethod
    def __rotate_matrix(euler):
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

    def eye_to_hand(self, points, camera_pos=None, camera_euler=None):
        if camera_pos is None:
            camera_pos = self.camera_pos
            camera_euler = self.camera_euler

        r_matrix = self.__rotate_matrix(camera_euler)
        return (r_matrix @ points.T).T + camera_pos

    def get_eye_position(self, px, py, radius=5, repeat=0):
        ROI = (px - radius, py - radius, px + radius, py + radius)

        # the first sample
        # if no repeat is needed, the depth camera need not be refreshed.
        X, Y, Z = self.depth_camera.get_XYZ_ROI(ROI)
        nans = np.isinf(Z) | np.isnan(Z)
        X[nans], Y[nans], Z[nans] = 0, 0, 0
        acc_x, acc_y, acc_z = X.sum(), Y.sum(), Z.sum()
        count = Z.shape[0] * Z.shape[1] - nans.sum()

        for i in range(repeat):
            self.depth_camera.refresh()
            X, Y, Z = self.depth_camera.get_XYZ_ROI(ROI)
            nans = np.isinf(Z) | np.isnan(Z)
            X[nans], Y[nans], Z[nans] = 0, 0, 0

            acc_x += X.sum()
            acc_y += Y.sum()
            acc_z += Z.sum()

            count += Z.shape[0] * Z.shape[1] - nans.sum()

        if count > Z.shape[0] * Z.shape[1] * 0.6:
            return np.array((acc_x / count, acc_y / count, acc_z / count))
        else:
            return None

    def get_hand_position(self, px, py, radius=5, repeat=0):
        pos = self.get_eye_position(px, py, radius, repeat)
        if pos is not None:
            pos = self.eye_to_hand(pos)
        return pos

    def get_eye_hand_position(self, px, py, radius=5, repeat=0):
        eye_pos = self.get_eye_position(px, py, radius, repeat)
        hand_pos = self.eye_to_hand(eye_pos)
        return eye_pos, hand_pos

    def refresh(self):
        self.depth_camera.refresh()


class TestInterface(object):
    def __init__(self, depth_camera, window_width):
        self.depth_camera = depth_camera
        self.eye = RobotEye(depth_camera, "测试相机")
        self.window_width = window_width
        self.hand_position = (0.0, 0.0, 0.0)

    def show_interface(self):
        self.eye.refresh()
        img = self.depth_camera.get_RGBimage(width=self.window_width, mark_infeasible=True)

        # show mouse position
        if self.hand_position is None:
            cv2.putText(img, "Current hand position is None", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(img, "Current hand position: %3.0f, %3.0f, %3.0f" % self.hand_position, (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Calibrating", img)
        cv2.setMouseCallback("Calibrating", on_mouse_event, self)

    def on_mouse_move(self, px, py):
        resize_k = self.depth_camera.resolution[0] / self.window_width
        px = int(resize_k * px + 0.5)
        py = int(resize_k * py + 0.5)
        pos = self.eye.get_hand_position(px, py)
        self.hand_position = (pos[0], pos[1], pos[2]) if pos is not None else None

    def on_mouse_down(self, px, py):
        resize_k = self.depth_camera.resolution[0] / self.window_width
        px = int(resize_k * px + 0.5)
        py = int(resize_k * py + 0.5)
        pos = self.eye.get_hand_position(px, py, repeat=5)
        self.hand_position = (pos[0], pos[1], pos[2]) if pos is not None else None

    def on_mouse_up(self, x, y):
        pass


if __name__ == '__main__':
    try:
        camera = ZEDCamera()
    except Exception:
        print("Failed to open the camera!")
        exit(0)

    test = TestInterface(depth_camera=camera, window_width=1400)

    key = cv2.waitKey(50) & 0xFF
    while key != ord('e') and key != ord('E'):
        test.show_interface()
        key = cv2.waitKey(50) & 0xFF
