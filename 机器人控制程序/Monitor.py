import numpy as np
import cv2
import time
import win32api,win32con
from ICCamera import ICCamera
from ImgStitch import Stitch


class Monitor(object):
    def __init__(self, dev_format):
        self.__init_devices(dev_format)

        """
        self.screen_num, self.screen_size = self.__init_screen_old()
        print("检测到%d个显示器" % self.screen_num)
        if self.screen_num < 3:
            print("显示器数量不足，仅有%d个显示器" % self.screen_num)

        self.img_left = np.zeros((self.screen_size[1], self.screen_size[0], 3), dtype=np.uint8)
        self.img_right = np.zeros((self.screen_size[1], self.screen_size[0], 3), dtype=np.uint8)
        self.img_center = np.zeros((self.screen_size[1], self.screen_size[0], 3), dtype=np.uint8)
        """

        self.__init_screen()
        self.img = np.zeros((self.screen_size[1], self.screen_size[0], 3), dtype=np.uint8)

    def __init_devices(self, dev_format):
        print("Initializing cameras")

        dev_names = ICCamera.enum_names()
        self.dev_list = [None, None, None]
        for i, name in enumerate(dev_names):
            if name == 'DFK 33UX250 12125088':
                self.dev_list[0] = ICCamera(device_name=name, dev_format=dev_format)
            elif name == 'DFK 33UX250 12125086':
                self.dev_list[2] = ICCamera(device_name=name, dev_format=dev_format)
            elif name == 'DFK 33UX250 12125089':
                self.dev_list[1] = ICCamera(device_name=name, dev_format=dev_format)
            print("Device %s initialized" % name)

    def __init_screen_old(self):
        """
        Obtain the screen size, and initialize the cv window.
        :return: None
        """
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        self.screen_size = (screen_width, screen_height)
        self.screen_num = win32api.GetSystemMetrics(win32con.SM_CMONITORS)

        cv2.namedWindow('window_left', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("window_left", 1920, 1080)
        cv2.moveWindow('window_left', x=-1920, y=0)
        cv2.setWindowProperty('window_left', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.namedWindow('window_right', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("window_right", 1920, 1080)
        cv2.moveWindow('window_right', x=0, y=0)
        cv2.setWindowProperty('window_right', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.namedWindow('window_center', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("window_center", 1920, 1080)
        cv2.moveWindow('window_center', x=1920, y=0)
        cv2.setWindowProperty('window_center', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        return self.screen_num, self.screen_size

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

    def __show_none_camera(self, screen_width, screen_height):
        img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.putText(img, "The camera is lost", (600, int(screen_height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (20, 255, 20), 1)
        return img

    def refresh(self):
        screen_width = int(self.screen_size[0] / 3)
        screen_height = self.screen_size[1]

        if self.dev_list[0] is not None:
            img_left = self.dev_list[0].snap()
            if img_left is not None:
                img_left = cv2.resize(img_left, (screen_width, screen_height))
                img_left = cv2.flip(img_left, 0)
                img_left = cv2.flip(img_left, 1)
                self.img[:, 0: screen_width, :] = img_left[:, :, :]
        else:
            img_left = self.__show_none_camera(screen_width, screen_height)
            self.img[:, 0: screen_width, :] = img_left[:, :, :]

        if self.dev_list[1] is not None:
            img_center = self.dev_list[1].snap()
            if img_center is not None:
                img_center = cv2.resize(img_center, (screen_width, screen_height))
                self.img[:, screen_width: screen_width * 2, :] = img_center[:, :, :]
        else:
            img_center = self.__show_none_camera(screen_width, screen_height)
            self.img[:, screen_width: screen_width * 2, :] = img_center[:, :, :]

        if self.dev_list[2] is not None:
            img_right = self.dev_list[2].snap()
            if img_right is not None:
                img_right = cv2.resize(img_right, (screen_width, screen_height))
                self.img[:, screen_width * 2: screen_width * 3, :] = img_right[:, :, :]
        else:
            img_right = self.__show_none_camera(screen_width, screen_height)
            self.img[:, screen_width * 2: screen_width * 3, :] = img_right[:, :, :]

        cv2.imshow("window", self.img)

        """
        if img_left is not None:
            cv2.imshow("window_left", img_left)

        if img_center is not None:
            cv2.imshow("window_center", img_center)

        if img_right is not None:
            cv2.imshow("window_right", img_right)
        """

    def get_images(self):
        images = list()
        for i, device in enumerate(self.dev_list):
            img = device.snap()
            if img is not None:
                #img = cv2.resize(img, (640, 480))
                images.append(img)
            time.sleep(0.1)
        return images

    def stitch_test(self):
        images = monitor.get_images()
        s = Stitch(images)

        s.leftshift()
        s.showImage('left')
        s.rightshift()
        print("done")
        cv2.imshow("stitched image", s.leftImage)
        cv2.waitKey()


if __name__ == '__main__':
    monitor = Monitor(dev_format="RGB24(2448x2048)")
    #monitor.stitch_test()

    while True:
        monitor.refresh()
        if cv2.waitKey(50) == 'q':
            break