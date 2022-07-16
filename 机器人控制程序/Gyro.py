import cv2
import numpy as np
import math


class gyro_indicator(object):
    def __init__(self, width):
        self.__gyro = cv2.imread("GyroIndicator.png")
        self.__shape = (self.__gyro.shape[0], self.__gyro.shape[1])
        self.__width = width

        # inner mask
        mask = np.logical_and(np.logical_and(self.__gyro[:, :, 0] == 0x0, self.__gyro[:, :, 1] == 0xFF),
                              self.__gyro[:, :, 2] == 0xFF)
        self.__mask_inner = np.zeros((self.__shape[0], self.__shape[1]), dtype=np.uint8)
        self.__mask_inner[np.where(mask)] = 0xFF

        # outer mask
        resized_gyro = cv2.resize(self.__gyro, (width, width))
        mask = np.logical_and(np.logical_and(resized_gyro[:, :, 0] == 0x80, resized_gyro[:, :, 1] == 0x80),
                              resized_gyro[:, :, 2] == 0x80)
        self.__mask_outer = np.zeros((width, width), dtype=np.uint8)
        self.__mask_outer[np.where(mask)] = 0xFF

        self.__gyro = cv2.bitwise_and(self.__gyro, self.__gyro, mask=cv2.bitwise_not(self.__mask_inner))

    def get_image(self, pitch, roll, width=None):
        roll *= 1.65

        sky_ground = np.zeros((self.__shape[0], self.__shape[1], 3), dtype=np.uint8)
        sky_ground[:, :, 0] = 196
        sky_ground[:, :, 1] = 147
        sky_ground[:, :, 2] = 91

        raw_w = self.__shape[1]
        raw_h = self.__shape[0]
        center = (int(raw_w / 2), int(raw_h / 2 - (1200 - 1085) * pitch / 10.0))

        x = int(raw_w / 2)
        y = int(x * math.sin(roll * math.pi / 180.0))
        p1 = (center[0] - x, center[1] + y)
        p2 = (center[0] + x - 1, center[1] - y)
        poly = np.array((p1, p2, (raw_w - 1, raw_h - 1), (0, raw_h - 1)))
        cv2.fillConvexPoly(sky_ground, poly, color=(50, 82, 125))

        gyro = cv2.bitwise_and(self.__gyro, self.__gyro, mask=cv2.bitwise_not(self.__mask_inner))
        sky_ground = cv2.bitwise_and(sky_ground, sky_ground, mask=self.__mask_inner)
        image = cv2.add(gyro, sky_ground)

        return image if width is None else cv2.resize(image, (width, width))

    def draw_on_screen(self, pitch, roll, screen, left, top):
        gyro = self.get_image(pitch, roll, self.__width)

        bk_img = screen[top: top + self.__width, left: left + self.__width, :]
        bk_img = cv2.bitwise_and(bk_img, bk_img, mask=self.__mask_outer)
        gyro = cv2.bitwise_and(gyro, gyro, mask=cv2.bitwise_not(self.__mask_outer))

        img = cv2.add(bk_img, gyro)
        screen[top: top + self.__width, left: left + self.__width, :] = img[:, :, :]


if __name__ == '__main__':
    gyro = gyro_indicator(300)
    bkimg = cv2.imread("IMG_6140.jpg")
    bkimg = cv2.resize(bkimg, (2048, int(bkimg.shape[0] * 2048 / bkimg.shape[1])))
    gyro.draw_on_screen(pitch=5, roll=20, screen=bkimg, left=100, top=100)

    cv2.imshow("", bkimg)
    cv2.waitKey()