import cv2
import torch
import glob
import os
import math
import numpy as np
import time
import random
import matplotlib.pyplot as plt

from CoilNet import CoilNet
from ImageSet import LabelType


class CoilTracer(object):
    def __init__(self, image_size, output_size):
        self.WINDOW_SIZE = image_size
        self.output_size = output_size
        self.coil_net = CoilNet(img_size=image_size, output_size=output_size)
        self.coil_net.init_model()

        # 使用CNN确定中心位置时，在初始位置附近，随机产生一系列窗口，对窗口中的图像检测中心，然后求平均
        self.net_x_range = (-20, 20)          # 随机窗口x方向范围
        self.net_y_range = (-20, 20)          # 随机窗口y方向范围
        self.net_rand_sample_cnt = 16          # 随机窗口数量

        # 保存路径检测记录数据
        self.save_begin_x = 1800
        self.save_end_x = 1250
        self.trace_saving = False
        self.trace_file = None
        self.trace_dir = ""

        self.record_trace = True
        self.repeat_trace = True
        self.cnn_trace = list()

        self.saving_video = False
        self.video_save = None

    def load_model(self, model_file):
        self.coil_net.load_model(model_file)

    def init_one_cycle(self, init_center):
        self.init_center = init_center
        self.lst_frame = None
        self.keytime = 10

    def trans_windows(self, init_center, x_range, y_range, step):
        w_center = list()
        for x in range(int((x_range[1] - x_range[0]) / step)):
            for y in range(int((y_range[1] - y_range[0]) / step)):
                w_center.append([init_center[0] + int(x * step + x_range[0]), init_center[1] + int(y * step + y_range[0])])
        return w_center

    def trans_windows_rand(self, init_center, x_range, y_range, count):
        w_center = list()
        for i in range(count):
            x = random.randint(x_range[0], x_range[1])
            y = random.randint(y_range[0], y_range[1])
            w_center.append([init_center[0] + x, init_center[1] + y])
        return w_center

    def norm_image(self, src):
        mean = src.mean()
        std = src.std()
        img = (src - mean) / std
        return img

    def img_window(self, center, img_size):
        left = int(int(center[0]) - img_size / 2)
        top = int(int(center[1]) - img_size / 2)
        right = int(int(center[0]) + img_size / 2)
        bottom = int(int(center[1]) + img_size / 2)

        return left, top, right, bottom

    def get_sub_image(self, img, center, img_size):
        left, top, right, bottom = self.img_window(center, img_size)

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

    def center_by_net(self, frame, img_size, init_center):
        # a set of images are clipped around the init center, each one is used to obtain one center location
        w_center = self.trans_windows_rand(init_center, self.net_x_range, self.net_y_range, self.net_rand_sample_cnt)

        sub_imgs = list()
        for _, center in enumerate(w_center):
            img = self.get_sub_image(frame, center, img_size)
            if img is not None:
                sub_imgs.append((self.norm_image(img), center))

        rst_cnt = len(sub_imgs)
        if rst_cnt == 0:
            return None

        # write images into a batch and predict the result
        batch_img = torch.zeros(len(sub_imgs), int(img_size), int(img_size), dtype=torch.float32)
        for i, img in enumerate(sub_imgs):
            batch_img[i] = torch.from_numpy(img[0])
        params = self.coil_net.predict(batch_img)

        # average obtained results
        values = np.zeros(LabelType.InnerOuter, np.int32)
        for i, p in enumerate(params):
            values[0] += p[0] - img_size / 2 + sub_imgs[i][1][0]
            values[1] += p[1] - img_size / 2 + sub_imgs[i][1][1]

            if self.output_size > LabelType.CenterOnly:
                values[2] += p[2]
                values[3] += p[3]
                values[4] += p[4]

            if self.output_size > LabelType.InnerOnly:
                values[5] += p[5] - img_size / 2 + sub_imgs[i][1][0]
                values[6] += p[6] - img_size / 2 + sub_imgs[i][1][1]
                values[7] += p[7]
                values[8] += p[8]
                values[9] += p[9]

        for i in range(10):
            values[i] = int(values[i] / rst_cnt)

        return values

    def analyze_one_frame(self, frame):
        # locate coil center by cnn
        center_cnn = self.center_by_net(frame, img_size=self.WINDOW_SIZE, init_center=self.init_center)
        self.cnn_trace.append(center_cnn)

        return center_cnn

    def show_trace(self, frame, center_cnn):
        # show coil center obtained by CNN and kalman filter
        cv2.circle(frame, (center_cnn[0], center_cnn[1]), 2, (255, 0, 0), 2)

        # show center moving trace
        lst_center = self.cnn_trace[0]
        for _, center in enumerate(self.cnn_trace):
            if lst_center[0] == -1:
                if center[0] != -1:
                    lst_center = center
            else:
                cv2.line(frame, lst_center, center, (0, 255, 0), 2)
                lst_center = center

    def __open_trace_save_file(self):
        file_name = time.strftime(self.trace_dir + "\\%Y-%m-%d-%H-%M-%S.cvs", time.localtime())
        file = open(file_name, 'w')
        return file

    def __within(self, begin, end, x, tolerance=1.0):
        return abs(x - begin) + abs(x - end) < abs(end - begin) + tolerance

    def save_trace(self, x, y):
        if self.__within(self.save_begin_x, self.save_end_x, x, 1.0):
            if not self.trace_saving:
                self.trace_saving = True
                self.trace_file = self.__open_trace_save_file()
        elif not self.__within(self.save_begin_x, self.save_end_x, x, 5.0):
            if self.trace_saving:
                self.trace_file.close()
                self.trace_saving = False

        if self.trace_saving:
            self.trace_file.write('%d,%d\n' % (x, y))

    def test_center_by_net(self, dir):
        os.chdir(dir)

        dist_range = 25
        distribution = np.zeros(dist_range)
        dist_density = np.zeros(dist_range)
        error_list = list()
        total_cnt = 0
        for file_name in glob.glob("*.jpg"):
            params = file_name.split('.')[0].split('-')
            if len(params) <= self.output_size:
                continue

            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            cx = int(params[1])
            cy = int(params[2])

            center_cnn = self.center_by_net(img, img_size=self.WINDOW_SIZE, init_center=(cx, cy))
            ex = math.fabs(cx - center_cnn[0])
            ey = math.fabs(cy - center_cnn[1])
            err = math.sqrt(ex * ex + ey * ey)
            if int(err) >= dist_range - 1:
                dist_density[dist_range - 1] += 1
            else:
                dist_density[int(err)] += 1
            total_cnt += 1

            error_list.append((err, file_name, cx, cy, center_cnn[0], center_cnn[1], ex, ey))

        # 由概率密度计算概率分布
        dist_density *= 100.0
        dist_density /= total_cnt
        for i in range(1, dist_range):
            distribution[i] = distribution[i - 1] + dist_density[i]

        # 显示统计曲线
        plt.plot(dist_density)
        plt.plot(distribution)
        plt.show()

        # 显示实际检测结果
        error_list = sorted(error_list, key=lambda error: error[0], reverse=True)
        for sample in error_list:
            err = sample[0]
            img = cv2.imread(sample[1], cv2.IMREAD_GRAYSCALE)
            cx = sample[2]
            cy = sample[3]
            cnn_x = sample[4]
            cnn_y = sample[5]
            ex = sample[6]
            ey = sample[7]

            cv2.circle(img, (cnn_x, cnn_y), 6, 0, 3)
            cv2.circle(img, (cx, cy), 6, 255, 3)

            cv2.putText(img, "ex = %2.0f, ey = %2.0f, err = %5.2f" % (ex, ey, err),
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)

            resize_width = 1200
            img = cv2.resize(img, (resize_width, int(resize_width * img.shape[0] / img.shape[1])))
            cv2.imshow("", img)
            if cv2.waitKey(0) == ord('q'):
                break

    def test(self, video, resize_k=1):
        while True:
            start_time = time.perf_counter()

            # 从摄像头或文件中读取帧，调整尺寸并灰度化
            ret, frame = video.read()
            if not ret:
                break
            if resize_k > 1:
                frame = cv2.resize(frame, (int(frame.shape[1] / resize_k), int(frame.shape[0] / resize_k)))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 计算获取中心点位置(卡尔曼滤波后的位置，卷积网络计算位置)
            center_kalman, center_cnn = self.analyze_one_frame(frame_gray)   # obtain the center

            # 显示计算耗时
            elapsed = (time.perf_counter() - start_time) * 1000
            cv2.putText(frame, "computing time = %6.2f ms" % elapsed, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)

            # 显示操作按键
            cv2.putText(frame, "S(s): start tracing and recording", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)
            cv2.putText(frame, "E(e): stop tracing and recording", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)

            # 显示运动轨迹
            if self.record_trace == True:
                self.show_trace(frame, center_kalman, center_cnn)

            # 视频显示
            frame = cv2.resize(frame, (1200, int(1200 * frame.shape[0] / frame.shape[1])))
            cv2.imshow("video", frame)

            # key operation
            key = cv2.waitKey(self.keytime) & 0xFF
            if key == ord('p') or key == ord('Q'):
                self.keytime = 10 if self.keytime == 0 else 0
            elif key == ord('S') or key == ord('s'):
                self.saving_video = True
            elif key == ord('E') or key == ord('e'):
                self.saving_video = False
            elif key == ord('q') or key == ord('Q'):
                break

if __name__ == '__main__':
    tracer = CoilTracer(512, 2)
    tracer.net_rand_sample_cnt = 32
    tracer.coil_net.load_model("D:\\01 自动上卷\\Python\\带卷定位2021-06-12-03-06.pt")

    tracer.test_center_by_net("D:\\01 自动上卷\\对准时刻附近图片")