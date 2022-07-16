import cv2
import numpy as np
import random
import torch
import sys
import math
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from enum import Enum
from CoilNet import CoilNet
from PIL import Image
from yolo import YOLO
from ZEDCamera import ZEDCamera

"""
Training tracking model
1. Sample images: BoomCtrl interface has a menu 'Take a picture', by clicking the menu an image from the ZED is stored. 
2. Label the samples: 
3. Build training set: LabelImages.py is used to cut labeled images into 256*256 sub-images for training. 
4. Train the model: Training is performed in CoilNet.py. After the training, change the model filename in BoomCtrl.py
"""


def on_mouse_event(event, x, y, flags, interface):
    if event == cv2.EVENT_LBUTTONDOWN:
        interface.on_mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        interface.on_mouse_up(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        interface.on_mouse_move(x, y)


class FilterType(Enum):
    NO_FILTER = 0
    KALMAN = 1
    LOW_PASS = 2


class PlaneTracker(object):
    def __init__(self, model_file):
        net = CoilNet(img_size=256, output_size=2)
        net.init_model()
        net.load_model(sys.path[0] + "\\" + model_file)
        self.tracker = net
        self.tracking = False

        # A set of sub-images are generated around the target, their distribution and count are decided here.
        self.rand_offset = 32
        self.rand_count = 4

        # YOLO V4
        self.yolo = YOLO()

        # A kalman-filter is used to track the target by default.
        # A low-pass filter is also defined
        self.__init_Kalman()
        self.low_pass_alpha = 0.3
        self.filter_type = FilterType.KALMAN

        self.target = None               # filtered target position
        self.raw_target = None     # target position without filtering

    def __init_Kalman(self):
        self.kalman_x = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman_x.F = np.array([[1.0, 1.0],
                                    [0.0, 1.0]])
        self.kalman_x.H = np.array([[1.0, 0.0]])
        self.kalman_x.P *= 10
        self.kalman_x.R *= 1
        self.kalman_x.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.1)

        self.kalman_y = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman_y.F = np.array([[1.0, 1.0],
                                    [0.0, 1.0]])
        self.kalman_y.H = np.array([[1.0, 0.0]])
        self.kalman_y.P *= 10
        self.kalman_y.R *= 1
        self.kalman_y.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.1)

    @staticmethod
    def __trans_windows(init_center, x_range, y_range, step):
        w_center = list()
        for x in range(int((x_range[1] - x_range[0]) / step)):
            for y in range(int((y_range[1] - y_range[0]) / step)):
                w_center.append([init_center[0] + int(x * step + x_range[0]), init_center[1] + int(y * step + y_range[0])])
        return w_center

    @staticmethod
    def __trans_windows_rand(init_center, x_range, y_range, count):
        w_center = list()
        for i in range(count):
            x = random.randint(-x_range, x_range)
            y = random.randint(-y_range, y_range)
            w_center.append([init_center[0] + x, init_center[1] + y])
        return w_center

    @staticmethod
    def __norm_image(src):
        mean = src.mean()
        std = src.std()
        img = (src - mean) / std
        return img

    @staticmethod
    def __img_window(center, img_size):
        left = int(int(center[0]) - img_size / 2)
        top = int(int(center[1]) - img_size / 2)
        right = int(int(center[0]) + img_size / 2)
        bottom = int(int(center[1]) + img_size / 2)

        return left, top, right, bottom

    @staticmethod
    def __distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))

    def __get_sub_image(self, img, center, img_size):
        left, top, right, bottom = self.__img_window(center, img_size)

        if left < 0:
            right += -left
            left = 0
        elif right > img.shape[1]:
            left -= right - img.shape[1]
            right = img.shape[1]

        if top < 0:
            bottom += -top
            top = 0
        elif bottom > img.shape[0]:
            top -= bottom - img.shape[0]
            bottom = img.shape[0]

        sub_img = img[top: bottom, left: right]
        return sub_img

    def __kalman_filter(self, x, y):
        """
        self.kalman_filter.predict()
        self.kalman_filter.update([[x, y]])
        return self.kalman_filter.x[0], self.kalman_filter.x[1]
        """
        self.kalman_x.predict()
        self.kalman_x.update(x)

        self.kalman_y.predict()
        self.kalman_y.update(y)

        return self.kalman_x.x[0], self.kalman_y.x[0]

    def __filter(self, x, y):
        if self.filter_type == FilterType.KALMAN:
            # note: the belief of the result should be checked
            x, y = self.__kalman_filter(x, y)
        elif self.filter_type == FilterType.LOW_PASS:
            if self.target is not None:
                if self.__distance(self.target, (x, y)) > 100:
                    x, y = None, None
                else:
                    x = self.low_pass_alpha * self.target[0] + (1.0 - self.low_pass_alpha) * x
                    y = self.low_pass_alpha * self.target[1] + (1.0 - self.low_pass_alpha) * y
        return None if x is None else (int(x), int(y))

    def __detect_by_yolo(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        inform, _ = self.yolo.detect_image(frame)

        return inform

    def __detect_by_cnn(self, frame, init_center):
        # a set of images are clipped around the init center, each one is used to obtain one center location
        w_center = self.__trans_windows_rand(init_center, self.rand_offset, self.rand_offset, self.rand_count)

        sub_imgs = list()
        for _, center in enumerate(w_center):
            img = self.__get_sub_image(frame, center, self.tracker.img_size)
            if img is not None:
                sub_imgs.append((self.__norm_image(img), center))
        if len(sub_imgs) == 0:
            return None

        # write images into a batch and predict the result
        batch_img = torch.zeros(len(sub_imgs), self.tracker.img_size, self.tracker.img_size, dtype=torch.float32)
        for i, img in enumerate(sub_imgs):
            batch_img[i] = torch.from_numpy(img[0])
        params = self.tracker.predict(batch_img)

        # average obtained results
        values = np.zeros(2, np.float32)
        for i, p in enumerate(params):
            values[0] += p[0] - self.tracker.img_size / 2 + sub_imgs[i][1][0]
            values[1] += p[1] - self.tracker.img_size / 2 + sub_imgs[i][1][1]
        x = values[0] / len(params)
        y = values[1] / len(params)

        return x, y

    def start_tracking(self):
        self.tracking = True
        self.first_scan = True

    def stop_tracking(self):
        self.tracking = False

    def track(self, frame):
        # detect the target by yolo first
        # if the target cannot be detected when tracking starts, the target will be set None
        yolo_rst = self.__detect_by_yolo(frame)
        if yolo_rst is None or yolo_rst[2] < 0.4:
            return None
        else:
            target = (yolo_rst[0], yolo_rst[1])

        # find the accurate position by cnn fitting
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y = self.__detect_by_cnn(frame, target)
        self.raw_target = (int(x), int(y))

        # filter the result
        # if the target is likely to be wrong, the filter returns None
        self.target = self.__filter(x, y)
        return self.target

    def test(self, frame, key, window_width):
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        resize_k = frame.shape[1] / window_width
        if not self.tracking:
            x = int(resize_k * self.mouse_pos[0] + 0.5)
            y = int(resize_k * self.mouse_pos[1] + 0.5)
            target_center = (x, y)
            self.target = (x, y)
        else:
            x, y = self.track(grey_frame)
            target_center = (x, y)

        if key == ord('s') or key == ord('S'):
            x = int(resize_k * self.target[0] + 0.5)
            y = int(resize_k * self.target[1] + 0.5)
            self.start_tracking()
        elif key == ord('e') or key == ord('E'):
            self.stop_tracking()
        elif key == ord('q') or key == ord('Q'):
            exit(0)

        # draw tracking result
        cv2.putText(grey_frame, "S: Start tracking", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        cv2.putText(grey_frame, "E: End tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        cv2.putText(grey_frame, "Q: Quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        cv2.putText(grey_frame, "Tracking" if self.tracking else "Idle", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

        cv2.circle(grey_frame, target_center, 5, 0, 2)
        cv2.rectangle(grey_frame, (target_center[0]  - 25, target_center[1] - 25), (target_center[0] + 25, target_center[1] + 25), 0, 2)
        cv2.line(grey_frame, (target_center[0] - 5, target_center[1]), (target_center[0] - 25, target_center[1]), 0, 2)
        cv2.line(grey_frame, (target_center[0] + 5, target_center[1]), (target_center[0] + 25, target_center[1]), 0, 2)
        cv2.line(grey_frame, (target_center[0], target_center[1] - 5), (target_center[0], target_center[1] - 25), 0, 2)
        cv2.line(grey_frame, (target_center[0], target_center[1] + 5), (target_center[0], target_center[1] + 25), 0, 2)

        #cv2.putText(frame, "S: Start tracking", (int(20), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        #cv2.imshow("grey", grey_frame)
        """
        cv2.circle(frame, target_center, 5, (0, 255, 0), 1)
        cv2.line(frame, (target_center[0] - 5, target_center[1]), (target_center[0] - 25, target_center[1]),
                 (0, 255, 0), 1)
        cv2.line(frame, (target_center[0] + 5, target_center[1]), (target_center[0] + 25, target_center[1]),
                 (0, 255, 0), 1)
        cv2.line(frame, (target_center[0], target_center[1] - 5), (target_center[0], target_center[1] - 25),
                 (0, 255, 0), 1)
        cv2.line(frame, (target_center[0], target_center[1] + 5), (target_center[0], target_center[1] + 25),
                 (0, 255, 0), 1)
        """

        frame = cv2.resize(grey_frame, (window_width, int(frame.shape[0] / resize_k)))
        cv2.imshow("track test", frame)
        cv2.setMouseCallback("track test", on_mouse_event, self)

    def on_mouse_move(self, px, py):
        pass

    def on_mouse_down(self, px, py):
        self.mouse_pos = (px, py)

    def on_mouse_up(self, x, y):
        pass


if __name__ == '__main__':
    tracker = PlaneTracker("加油口定位2021-11-20-17-51.pt")
    camera = ZEDCamera()
    while True:
        camera.refresh()
        frame = camera.get_RGBimage(mark_infeasible=True)
        #frame = cv2.imread("test.jpg", cv2.IMREAD_COLOR)
        key = cv2.waitKey(10)
        tracker.test(frame, key, 1920)