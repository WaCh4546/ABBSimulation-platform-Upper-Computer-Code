import numpy as np
import cv2
import time
import win32api,win32con
from ICCamera import ICCamera
from ImgStitch import Stitch


class Monitor(object):
    def __init__(self, dev_format):
        self.__init_devices(dev_format)

        self.screen_num, self.screen_size = self.__init_screen()
        if self.screen_num == 1:
            self.screen_size = (int(self.screen_size[0] / 3), int(self.screen_size[1] / 3))
        self.screen_img = np.zeros((self.screen_size[1], self.screen_size[0] * 3, 3), dtype=np.uint8)

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

    def __init_screen(self):
        """
        Obtain the screen size, and initialize the cv window.
        :return: None
        """
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        self.screen_size = (screen_width, screen_height)
        self.screen_num = win32api.GetSystemMetrics(win32con.SM_CMONITORS)

        cv2.namedWindow('window', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('window', 0, 0)

        return self.screen_num, self.screen_size

    def refresh(self):
        for i, device in enumerate(self.dev_list):
            img = device.snap()
            if img is not None:
                img = cv2.resize(img, self.screen_size)
                self.screen_img[0: self.screen_size[1], i * self.screen_size[0]: (i+1) * self.screen_size[0], :] \
                    = img[0: self.screen_size[1], 0: self.screen_size[0], :]
            time.sleep(0.1)   # Note: this shall be removed when all the cameras are connected by USB3.0
        cv2.imshow("window", self.screen_img)

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
    monitor = Monitor(dev_format="RGB32 (1024x768)")
    #monitor.stitch_test()

    while True:
        monitor.refresh()
        if cv2.waitKey(50) == 'q':
            break