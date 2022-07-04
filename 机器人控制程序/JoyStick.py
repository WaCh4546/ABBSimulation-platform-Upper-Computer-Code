import pygame
import time
import numpy as np
import math


class JoyStick(object):
    def __init__(self, id):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise Exception("Joystick not found!")

        self.joystick = pygame.joystick.Joystick(id)
        self.joystick.init()

        self.axis = np.zeros(self.joystick.get_numaxes())
        self.__alpha = 0.0

        self.button = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]
        self.hat = [self.joystick.get_hat(i) for i in range(self.joystick.get_numhats())]

        self.lst_button = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]
        self.lst_hat = [self.joystick.get_hat(i) for i in range(self.joystick.get_numhats())]

        self.button_click = self._check_button_click(self.lst_button, self.button)
        self.hat_click = self._check_hat_click(self.lst_hat, self.hat)

        self.dead_zone = 0.1

    @staticmethod
    def _check_button_click(lst, cur):
        rise = [0] * len(lst)
        for i in range(len(lst)):
            if not lst[i] and cur[i]:
                rise[i] = 1
        return rise

    @staticmethod
    def _check_hat_click(lst, cur):
        rise = [(0, 0)] * len(lst)
        for i in range(len(lst)):
            h0 = 1 if not lst[i][0] and cur[i][0] else 0
            h1 = 1 if not lst[i][1] and cur[i][1] else 0
            rise[i] = (h0, h1)
        return rise

    @staticmethod
    def count():
        pygame.init()
        return pygame.joystick.get_count()

    @staticmethod
    def device_name(id):
        if id < JoyStick.count():
            return pygame.joystick.Joystick(id).get_name()
        else:
            return None

    def name(self):
        return self.joystick.get_name()

    def refresh(self):
        pygame.event.pump()

        for i in range(self.joystick.get_numaxes()):
            axis = self.joystick.get_axis(i)
            self.axis[i] = self.axis[i] * self.__alpha + axis * (1.0 - self.__alpha)
            if -self.dead_zone < self.axis[i] < self.dead_zone:
                self.axis[i] = 0

        self.lst_button = self.button
        self.lst_hat = self.hat

        self.button = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]
        self.hat = [self.joystick.get_hat(i) for i in range(self.joystick.get_numhats())]

        self.button_click = self._check_button_click(self.lst_button, self.button)
        self.hat_click = self._check_hat_click(self.lst_hat, self.hat)


class PlaneJoystick(object):
    def __init__(self):
        self.joystick = None
        self.throttle = None
        self.rudder = None
        for i in range(JoyStick.count()):
            stick = JoyStick(i)
            print(stick.name())
            if "Joystick" in stick.name():
                self.joystick = stick
            elif "Throttle" in stick.name():
                self.throttle_stick = stick
            elif "Rudder" in stick.name():
                self.rudder = stick

        if self.joystick is None:
            raise Exception("joystick is not connected")
        if self.throttle_stick is None:
            raise Exception("throttle is not connected")
        if self.rudder is None:
            raise Exception("ruddle is not connected")

        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.throttle = 0
        self.menu_option = 0

    def refresh(self):
        self.joystick.refresh()
        self.throttle_stick.refresh()
        self.rudder.refresh()

        self.yaw = self.rudder.axis[2]
        self.pitch = self.joystick.axis[1]
        self.roll = self.joystick.axis[0]
        self.throttle = self.throttle_stick.axis[2]

        if self.joystick.hat_click[0][1]:
            self.menu_option = self.joystick.hat[0][1]
        elif self.joystick.button_click[1]:
            self.menu_option = 2
        else:
            self.menu_option = 0


class TestJoystick_Plane(object):
    def __init__(self):
        if JoyStick.count() < 1:
            raise Exception("Joystick is not connected!")
        self.joystick = JoyStick(0)

        self.yaw = 0
        self.pitch = 0
        self.roll = 0
        self.throttle = 0
        self.menu_option = 0

    def refresh(self):
        self.joystick.refresh()

        self.roll = self.joystick.axis[0]
        self.pitch = self.joystick.axis[1]
        self.yaw = self.joystick.axis[5]
        self.throttle = self.joystick.axis[2]

        if self.joystick.hat_click[0][1]:
            self.menu_option = self.joystick.hat[0][1]
        elif self.joystick.button_click[1]:
            self.menu_option = 2
        else:
            self.menu_option = 0


class BoomJoystick(object):
    def __init__(self):
        self.joystick = None
        self.throttle = None
        for i in range(JoyStick.count()):
            stick = JoyStick(i)
            print(stick.name())
            if "Joystick" in stick.name():
                self.joystick = stick
            elif "Throttle" in stick.name():
                self.throttle = stick

        if self.joystick is None:
            raise Exception("Joystick is not connected!")
        if self.throttle is None:
            raise Exception("Throttle is not connected!")

        self.yaw = 0             # the yaw angle in [-1, 1]
        self.pitch = 0           # the pitch angle in [-1, 1]
        self.stretch = 0         # the stretch between -1, 0, 1
        self.menu_option = 0     # the menu option: move up or down(1, -1), click(2), none(0)

        self.track_click = 0
        self.track_x = 0
        self.track_y = 0

    def refresh(self):
        self.joystick.refresh()
        self.throttle.refresh()

        self.yaw = self.joystick.axis[0]
        self.pitch = self.joystick.axis[1]
        self.stretch = self.joystick.button[0] - self.joystick.button[3]

        self.track_click = self.joystick.button_click[6]
        self.track_x = self.joystick.button[11] - self.joystick.button[13]
        self.track_y = self.joystick.button[10] - self.joystick.button[12]

        if self.joystick.hat_click[0][1]:
            self.menu_option = self.joystick.hat[0][1]
        elif self.joystick.button_click[1]:
            self.menu_option = 2
        else:
            self.menu_option = 0

        print(self.track_click)


class TestJoystick_Boom(object):
    def __init__(self):
        if JoyStick.count() < 1:
            raise Exception("Joystick is not connected!")
        self.joystick = JoyStick(0)

        self.tracking = False
        self.__track_btn_status = 0

        self.yaw = 0
        self.pitch = 0
        self.stretch = 0
        self.menu_option = 0

        self.track_click = 0
        self.track_x = 0
        self.track_y = 0

    def refresh(self):
        self.joystick.refresh()

        self.yaw = self.joystick.axis[0]
        self.pitch = self.joystick.axis[1]
        self.stretch = self.joystick.button[0] - self.joystick.button[2]

        if self.joystick.hat_click[0][1]:
            self.menu_option = self.joystick.hat[0][1]
        elif self.joystick.button_click[1]:
            self.menu_option = 2
        else:
            self.menu_option = 0

        self.track_click = self.joystick.button_click[6]
        self.track_x = self.joystick.button[4] - self.joystick.button[5]
        self.track_y = self.joystick.button[8] - self.joystick.button[9]


class PS4Robot(object):
    def __init__(self, id=0):
        self.ps4 = JoyStick(id)
        if self.ps4.name().find("PS4") < 0:
            raise Exception("No a PS4 handle")

        self.dx = 0.0
        self.dy = 0.0
        self.step = 0.0
        self.yaw = 0.0
        self.speed = 0.0

    def refresh(self):
        self.ps4.refresh()

        x = self.ps4.axis[2]
        y = -self.ps4.axis[3]
        xy = math.sqrt(x * x + y * y)
        if xy < 0.1:
            self.dx = 0.0
            self.dy = 0.0
            self.step = 0.0
        else:
            self.dx = x / xy
            self.dy = y / xy
            self.step = max(math.fabs(x), math.fabs(y))
        self.yaw = self.ps4.axis[0]

        if self.ps4.button_click[3]:
            self.speed += 0.1
        elif self.ps4.button_click[0]:
            self.speed -= 0.1
        self.speed = min(1.0, max(0.0, self.speed))
        print("Dir = %3.1f, %3.1f" % (self.dx, self.dy))
        print("Step = %3.1f" % self.step)
        print("Yaw = %3.1f" % self.yaw)
        print("Speed = %3.1f" % self.speed)


if __name__ == '__main__':
    #plane = BoomJoystick()
    plane = PS4Robot()
    print(plane.ps4.name())
    while True:
        plane.refresh()
        time.sleep(0.1)