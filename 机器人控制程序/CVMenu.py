# CVMenu is designed to provide simple menu on the opencv window

import cv2
import platform
import ctypes
import numpy as np


class CVMenu(object):
    def __init__(self, left, top, width, height):
        self.items = list()
        self.item_idx = 0
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.color_normal = (20, 255, 20)
        self.color_selected = (255, 20, 20)
        self.color_disabled = (80, 80, 80)

    def add_item(self, caption):
        item = {
            'caption': caption,
            'index': len(self.items),
            'selected': False,
            'enabled': True
        }
        self.items.append(item)

    def clear(self):
        self.items.clear()
        self.item_idx = -1

    def get_item_idx(self, caption):
        for i in range(len(self.items)):
            if self.items[i]['caption'] == caption:
                return i
        return -1

    def _draw_item(self, img, item_idx):
        item = self.items[item_idx]
        top = self.top + item_idx * self.height
        color = self.color_normal
        if item["selected"]:
            color = self.color_selected
        elif not item["enabled"]:
            color = self.color_disabled

        cv2.rectangle(img, (self.left, top), (self.left + self.width, top + self.height), (100, 100, 100), thickness=-1)
        cv2.rectangle(img, (self.left, top), (self.left + self.width, top + self.height), color, thickness=1)

        txt_size = cv2.getTextSize(item["caption"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, item["caption"], (self.left + 10, top + int((self.height + txt_size[0][1]) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def move_up(self):
        idx = self.item_idx - 1
        while idx >= 0:
            if self.items[idx]['enabled']:
                break
            idx -= 1

        if idx >= 0:
            self.items[self.item_idx]['selected'] = False
            self.item_idx = idx
            self.items[self.item_idx]['selected'] = True

    def move_down(self):
        idx = self.item_idx + 1
        while idx < len(self.items):
            if self.items[idx]['enabled']:
                break
            idx += 1

        if idx < len(self.items):
            self.items[self.item_idx]['selected'] = False
            self.item_idx = idx
            self.items[self.item_idx]['selected'] = True

    def click(self, x, y):
        if 0 <= self.item_idx < len(self.items):
            self.items[self.item_idx]['selected'] = False

        if self.left <= x < self.left + self.width:
            idx = int((y - self.top) / self.height)
            if idx < len(self.items) and self.items[idx]['enabled']:
                self.items[idx]['selected'] = True
                self.item_idx = idx
            else:
                self.item_idx = -1
        else:
            self.item_idx = -1

    def selected(self):
        if -1 < self.item_idx < len(self.items):
            return self.items[self.item_idx]
        else:
            return None

    def refresh(self, img):
        for i in range(len(self.items)):
            self._draw_item(img, i)


def on_mouse_event(event, x, y, flags, menu):
    if event == cv2.EVENT_LBUTTONDOWN:
        menu.click(x, y)


if __name__ == '__main__':
    if platform.system() == "Windows":
        winapi = ctypes.windll.user32
        screen_width = winapi.GetSystemMetrics(0)
        screen_height = winapi.GetSystemMetrics(1)
    elif platform.system() == "Linux":
        screen_width = 100
        screen_height = 100
    else:
        screen_width = 100
        screen_height = 100

    menu = CVMenu(20, 20, 400, 25)
    menu.add_item("menu item1")
    menu.add_item("menu item2")
    menu.add_item("menu item3")
    menu.add_item("menu item4")
    menu.items[2]['enabled'] = False

    cv2.namedWindow("Test", cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow("Test", screen_width, screen_height)
    cv2.setMouseCallback("Test", on_mouse_event, menu)

    key = 0
    while key != ord('q') and key != ord('Q'):
        if key == 2490368:
            menu.move_up()
        elif key == 2621440:
            menu.move_down()

        img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        menu.refresh(img)
        cv2.imshow("Test", img)
        key = cv2.waitKeyEx(0)

