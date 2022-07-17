import cv2
import socket
import math
import numpy as np
import time
import threading


class ImgSender(threading.Thread):
    def __init__(self, remote_addr):
        threading.Thread.__init__(self)
        self.remote_addr = remote_addr
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.lock = threading.RLock()
        self.image = None
        self.connected = False
        self.terminated = False
        self.error = False
        self.err_string = None
        self.start()

    def __init_connection(self):
        try_times = 3
        while try_times:
            try:
                try_times -= 1
                self.tcp.settimeout(5.0)
                self.tcp.connect(self.remote_addr)
                self.error = False
                return True
            except socket.timeout:
                self.error = True
                self.err_string = "Timeout in connecting U-screen"
            except WindowsError as e:
                self.error = True
                self.err_string = e
        return False

    def __send(self, img):
        # transform image into jpeg
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img_encode = cv2.imencode('.jpg', img, encode_param)
        img_data = np.array(img_encode).flatten()
        img_size = len(img_data)

        stream_head = np.array((img_size >> 24, img_size >> 16, img_size >> 8, img_size), dtype=np.uint8)
        stream_data = np.hstack((stream_head, img_data))

        try:
            self.tcp.settimeout(1.0)
            self.tcp.send(stream_data)
            self.error = False
        except socket.timeout:
            self.error = True
            self.err_string = "Timeout in sending image"
        except WindowsError as e:
            self.error = True
            self.err_string = e
        return self.error

    def run(self):
        while not self.terminated:
            time.sleep(0.02)

            if not self.connected:
                if not self.__init_connection():
                    self.terminated = True
                    continue
                else:
                    self.connected = True

            self.lock.acquire()
            if self.image is not None:
                if self.__send(self.image):
                    self.connected = False
                self.image = None
            self.lock.release()
        self.tcp.close()

    def send(self, image):
        self.lock.acquire()
        self.image = image.copy()
        self.lock.release()
        return True

    def terminate(self):
        self.terminated = True
        self.join()


class ImgTrans(object):
    def __init__(self, remote_addr, local_port=None):
        self.remote_addr = remote_addr
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.settimeout(5.0)

        if remote_addr is None:
            self.tcp.bind(('localhost', local_port))
            self.tcp.listen()
        else:
            self.tcp.connect(remote_addr)

    def send(self, img):
        # transform image into jpeg
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, img_encode = cv2.imencode('.jpg', img, encode_param)
        img_data = np.array(img_encode).flatten()
        img_size = len(img_data)
        print("img size = %d" % img_size)

        stream_head = np.array((img_size >> 24, img_size >> 16, img_size >> 8, img_size), dtype=np.uint8)
        stream_data = np.hstack((stream_head, img_data))

        try:
            self.tcp.settimeout(1.0)
            self.tcp.send(stream_data)
            return True
        except socket.timeout:
            print("Timeout in sending image")
            return True
        except WindowsError as e:
            print("Failed to send image")
            print(e)
            return False


if __name__ == '__main__':
    #rcver = Receiver_Test()
    #rcver.start()

    try:
        sender = ImgSender(remote_addr=('192.168.1.5', 10000))
    except WindowsError as e:
        print(e)
        exit(0)

    raw_img = cv2.imread("d:\\img.jpg")
    raw_img = cv2.resize(raw_img, (2208, 1242))
    cnt = 0
    while True:
        time.sleep(0.3)
        img = raw_img.copy()
        cv2.putText(img, time.strftime("images\\%Y_%m_%d_%H_%M_%S.jpg", time.localtime()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print("Start trans")
        sender.send(img)
        if sender.error:
            print(sender.err_string)
        cnt += 1
        print("send %d" % cnt)