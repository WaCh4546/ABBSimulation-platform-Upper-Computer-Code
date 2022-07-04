import cv2
import glob
import os
import time
import datetime

from CoilTracer import CoilTracer
from ImageSet import LabelType
from CoilStatus import CoilOpenStatus
from CoilStatus import CoilPosStatus
from CoilInnerOuterSize import CoilInnerOuterSize

class AutoLoader():
    def __init__(self):
        self.coil_tracer = CoilTracer(image_size=512, output_size=LabelType.CenterOnly)
        self.coil_tracer.load_model("D:\\01 自动上卷\\Python\\带卷定位2021-05-27-02-35.pt")    #("带卷定位2021-04-28-23-07.pt")
        self.coil_tracer.repeat_trace = False
        self.coil_tracer.trace_dir = "D:\\01 自动上卷"

        self.coil_pos_status = CoilPosStatus(src_size=1200, src_position=(0, 800))
        self.coil_pos_status.init_model()
        self.coil_pos_status.load_model("上卷检测2020-10-24-16-30.pt")

        self.coil_open_status = CoilOpenStatus(src_size=1200, src_position=(0, 800))
        self.coil_open_status.init_model()
        self.coil_open_status.load_model("开卷检测2020-10-24-12-12.pt")

        self.coil_size = CoilInnerOuterSize(src_size=1400, dst_size=512)
        self.coil_size.load_model("带卷定位2021-10-22-00-55.pt")

        # the trace that the coil is expected to follow
        # it has five elements: (type, a, b, var, begin, end).
        # type = 0: x as variable; type = 1: y as variable
        # a, b: coefficient of the trace. for type == 0, y = ax + b; for type == 1, x = ay + b
        # var: tolerance of the error
        # begin, end: the range of the trace
        self.exp_traces = list()
        self.exp_traces.append((1, -1.017699, 2961.818164, 7, 1510, 1690))
        #self.exp_traces.append((0, -1 / 1.017699, -2961.818164 / 1.017699, 7, 1260, 1420))

    def init_one_cycle(self):
        self.coil_tracer.init_one_cycle(init_center=(2048, 438))
        self.loading_stage = 0
        self.saving_video = False
        self.video_save = None
        self.keytime = 10

    def __within(self, begin, end, x, tolerance=1.0):
        return abs(x - begin) + abs(x - end) <= abs(end - begin) + tolerance

    def draw_exp_trace(self, frame):
        for i, line in enumerate(self.exp_traces):
            if line[0] == 0:
                x1 = line[4]
                y1 = int(x1 * line[1] + line[2])
                x2 = line[5]
                y2 = int(x2 * line[1] + line[2])
            else:
                y1 = line[4]
                x1 = int(y1 * line[1] + line[2])
                y2 = line[5]
                x2 = int(y2 * line[1] + line[2])

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(frame, (x1, y1 + line[3]), (x2, y2 + line[3]), (255, 0, 0), 2)
            cv2.line(frame, (x1, y1 - line[3]), (x2, y2 - line[3]), (255, 0, 0), 2)

    def check_trace(self, x, y):
        for i, line in enumerate(self.exp_traces):
            if line[0] == 0:
                if self.__within(line[4], line[5], x):
                    exp_y = x * line[0] + line[1]
                    return y - exp_y
            else:
                if self.__within(line[4], line[5], y):
                    exp_y = x / line[1] - line[2] / line[1]
                    return y - exp_y
        return 0

    def save_video(self, video, frame):
        if self.saving_video:
            if self.video_save is None:
                now = datetime.datetime.now()
                datetime_str = "%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)

                fps = video.get(cv2.CAP_PROP_FPS)
                size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

                self.video_save = cv2.VideoWriter("d:/" + datetime_str + ".avi", cv2.VideoWriter_fourcc(*'XVID'), fps, size)
            self.video_save.write(frame)

            cv2.putText(frame, "recording video", (600, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2)
        else:
            self.video_save = None


    def test(self, video):
        while True:
            start_time = time.perf_counter()

            # 从摄像头或文件中读取帧，调整尺寸并灰度化
            ret, frame = video.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 计算获取中心点位置(卡尔曼滤波后的位置，卷积网络计算位置)
            if self.loading_stage == 0:
                center_kalman, center_cnn = self.coil_tracer.analyze_one_frame(frame_gray)   # obtain the center
                self.coil_tracer.show_trace(frame, center_kalman, center_cnn)

                self.coil_size.analyze_one_frame(frame, center_kalman)
                self.coil_size.show_ellipses(frame)

                error = self.check_trace(center_kalman[0], center_kalman[1])
                cv2.putText(frame, "x = %6.1f, y = %6.1f, err = %6.1f" % (center_kalman[0], center_kalman[1], error), (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            elif self.loading_stage == 1:
                status = self.coil_pos_status.analyze_one_frame(frame_gray)
                cv2.putText(frame, "coil loaded" if status else "coil not loaded", (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)
            elif self.loading_stage == 2:
                status = self.coil_open_status.analyze_one_frame(frame_gray)
                cv2.putText(frame, "coil opened" if status else "coil not opened", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)

            # 保存视频
            self.save_video(video, frame)

            # 显示期望轨迹线
            self.draw_exp_trace(frame)

            # 显示计算耗时
            elapsed = (time.perf_counter() - start_time) * 1000
            cv2.putText(frame, "computing time = %6.2f ms" % elapsed, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)

            # 显示操作按键
            cv2.putText(frame, "S(s): start tracing and recording", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)
            cv2.putText(frame, "E(e): stop tracing and recording", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)
            cv2.putText(frame, "L(l): detect if the coil is loaded", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)
            cv2.putText(frame, "O(o): detect if the coil is opened", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)

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
            elif key == ord('L') or key == ord('l'):
                self.loading_stage = 1
            elif key == ord('o') or key == ord('O'):
                self.loading_stage = 2
            elif key == ord('q') or key == ord('Q'):
                break

if __name__ == '__main__':
    auto_loader = AutoLoader()

    #os.chdir("D:/2020.9.7-15帧视频数据/3072-2048/下午3-5 曝光时间10ms 增益20.03")
    os.chdir("D:\\2021.1现场视频剪")
    #os.chdir("D:/阴影问题视频")
    for file_name in glob.glob("*.avi"):
        video = cv2.VideoCapture(file_name)

        auto_loader.init_one_cycle()
        auto_loader.test(video)