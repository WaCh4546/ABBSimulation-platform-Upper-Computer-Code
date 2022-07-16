import cv2
import socket
import math
import numpy as np
import time
import threading


class ImgTrans(object):
    def __init__(self, remote_addr, local_port=None):
        self.remote_addr = remote_addr
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if remote_addr is None:
            self.udp.bind(('localhost', local_port))

    def send(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img_encode = cv2.imencode('.jpg', img, encode_param)
        data = np.array(img_encode)

        seg_size_max = 16384
        img_size = len(data)
        seg_count = math.ceil(img_size / seg_size_max)
        seg_index = 0
        while seg_index < seg_count:
            seg_begin = seg_index * seg_size_max
            seg_size = seg_size_max if seg_index < seg_count - 1 else len(data) - seg_begin
            seg_head = np.array((img_size >> 24, img_size >> 16, img_size >> 8, img_size, seg_count, seg_index),
                                dtype=np.uint8)
            seg_data = data[seg_begin: seg_begin + seg_size].flatten()
            seg = np.hstack((seg_head, seg_data))
            self.udp.sendto(seg, self.remote_addr)

            seg_index += 1

    def receive(self):
        seg_size_max = 16384
        img_data = None

        self.udp.settimeout(0.2)
        while True:
            try:
                data, addr = self.udp.recvfrom(seg_size_max + 6)
                data = np.frombuffer(data, dtype=np.uint8)
                img_size = data[0] << 24 | data[1] << 16 | data[2] << 8 | data[3]
                seg_count = data[4]
                seg_index = data[5]
                seg_begin = seg_index * seg_size_max
                seg_size = len(data) - 6

                if img_data is None:
                    img_data = np.zeros(img_size, dtype=np.uint8)
                img_data[seg_begin: seg_begin + seg_size] = data[6: len(data)]

                if seg_index >= seg_count - 1:
                    img = cv2.imdecode(img_data, 1)
                    return img
            except socket.timeout:
                return None


class Receiver_Test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.rcver = ImgTrans(remote_addr=None, local_port=6002)

    def run(self):
        while True:
            cv2.waitKey(20)
            img = self.rcver.receive()
            if img is not None:
                #img = cv2.resize(img, (800, 600))
                cv2.imshow('client', img)
            else:
                print("timeout")


if __name__ == '__main__':
    #rcver = Receiver_Test()
    #rcver.start()

    sender = ImgTrans(remote_addr=('localhost', 10000))
    raw_img = cv2.imread("G:\\IMG_6104.jpg")
    raw_img = cv2.resize(raw_img, (2208, 1242))
    while True:
        time.sleep(0.3)
        img = raw_img.copy()
        cv2.putText(img, time.strftime("images\\%Y_%m_%d_%H_%M_%S.jpg", time.localtime()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        sender.send(img)