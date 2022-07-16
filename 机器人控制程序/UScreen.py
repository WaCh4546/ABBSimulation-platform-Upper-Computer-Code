import cv2
import win32api, win32con
from ImgTrans import ImgTrans


class UScreen(object):
    def __init__(self):
        self.img_boom = ImgTrans(remote_addr=None, local_port=10000)
        self.img_plane = ImgTrans(remote_addr=None, local_port=10001)

        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        self.screen_size = (screen_width, screen_height)
        self.show_boom = True

        cv2.namedWindow('window', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow('window', 0, 0)

    def refresh(self):
        key = cv2.waitKey(20)
        if key == ord('b') or key == ord('B'):
            self.show_boom = True
        elif key == ord('p') or key == ord('P'):
            self.show_boom = False

        img = self.img_boom.receive() if self.show_boom else self.img_plane.receive()
        if img is not None:
            img = cv2.resize(img, self.screen_size)
            cv2.imshow("window", img)


if __name__ == '__main__':
    screen = UScreen()
    while True:
        screen.refresh()