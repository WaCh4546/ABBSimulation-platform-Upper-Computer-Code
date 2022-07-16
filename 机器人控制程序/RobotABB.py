from socket import *
import math
import time

"""
The communicating protocol between PC and ABB robot:
1. Query orders
a. TES:
  Used to check the connection between PC and robot. The robot returns
  RES:ok
b. RCP:
  Used to query the current position and gesture of the robot. The robot returns
  RES: x_mm, y_mm, z_mm, yaw_deg, pitch_deg, roll_deg \r\n
  
2. Move orders
a. MEC:
   Move the tool center according to the ground coordinate system. 
   MEC: cnt\r\n 
        x1_mm, y1_mm, z1_mm, yaw1_deg, pitch1_deg, roll1_deg, speed1_mms, time1_s\r\n 
        ...,
        xn_mm, yn_mm, zn_mm, yawn_deg, pitchn_deg, rolln_deg, speedn_mms, timen_s \r\n
    Note: There are speed and time simultaneously in each trace point, which obeys following rule:
          1) When time_s is zero, speed_mms will be used. Otherwise speed_mms is masked.
          2) If time_s cannot be satisfied, the robot moves in its maximum speed. 
       
b. MIC:
   Move the tool center according to the TCP coordinate system. A TCP coordinate has the tool center as its origin, 
   therefore the specified movement is the relative offset of the current robot position and gesture. 
   MIC: cnt\r\n 
        x1_mm, y1_mm, z1_mm, yaw1_deg, pitch1_deg, roll1_deg, speed1_mms, time1_s\r\n 
        ...\r\n
        xn_mm, yn_mm, zn_mm, yawn_deg, pitchn_deg, rolln_deg, speedn_mms, timen_s \r\n
    Note: As a TCP coordinate has the tool center as its origin, it is carried out incrementally.
c. MJO:   
   Move robot joints incrementally. The order format:
   MJO: cnt\r\n 
        J1, J2, J3, J4, J5, J6, speed_degs, time_s\r\n
        ...
        J1, J2, J3, J4, J5, J6, speed_degs, time_s\r\n
   
d. RES:
   If the order is successfully carried out, the robot returns
   RES:Ready
   otherwise it returns
   RES:NotReady
   
3. Other response
   When robot receives orders that cannot be carried out, or with wrong position or gesture, etc., it returns
   MES:message
   The detailed message content is decided by the robot.
"""


class RobotABB(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.connected = self.connect(host, port)
        self.status = 0
        self.pos_euler = None
        self.error = ""
        self.EGM = False
        self.idle = False
        self.started = False

    def connect(self, host, port):
        try:
            self.socket.connect((host, port))
            return True
        except:
            print("Failed to connect the robot!")
            return False

    def __send_order(self, order):
        # in case the connection is closed for some reason, try to connect the robot once.
        if not self.connected:
            print("Reconnect robot")
            self.connected = self.connect(self.host, self.port)
            if not self.connected:
                print("Reconnection failed")
                return None

        # send order
        try:
            self.socket.send(order.encode('utf-8'))
            rcv = self.socket.recv(128).decode()

            if rcv.find(order[0:3]) >= 0:
                return rcv.split()
            else:
                return None
        except:
            self.connected = False
            return None

    def __extract_pos_euler(self, line):
        params = line.split(',')
        return float(params[0]), float(params[1]), float(params[2]), \
               float(params[3]), float(params[4]), float(params[5])

    def refresh_status(self):
        rcv = self.__send_order("RCP")
        if rcv is None:
            return False

        # line 1 returns working mode of the robot
        self.status = int(rcv[1])
        self.EGM = int(rcv[2]) == 1
        self.started = int(rcv[3]) == 1
        self.pos_euler = self.__extract_pos_euler(rcv[4])

        if self.status == -1:
            self.error = "The target(%d) unaccessable" % int(rcv[5])
        elif self.status == -2:
            self.error = "Targets number unmatching"
        elif self.status == -3:
            self.error = "(%s, %s) data parsing error" % (rcv[5].split(','))
        elif self.status == -4:
            self.error = "EGM mode"
        elif self.status == -5:
            self.error = "Not started"
        return True

    def __pack_move_order(self, trace):
        order = "%d\r\n" % len(trace)
        for _, point in enumerate(trace):
            if len(point) != 8:
                break
            for _, v in enumerate(point):
                order += "%1.2f," % v
            order = order[0:-2] + "\r\n"
        return order

    def move_by_ground_coordinate(self, trace):
        """
        The order follows the MEC format.
        :param trace:
        :return:
        """
        order = "MEC:" + self.__pack_move_order(trace)
        rcv = self.__send_order(order)
        return rcv is not None

    def move_by_TCP_coordinate(self, trace):
        """
        The order follows the MEC format.
        :param trace:
        :return:
        """
        order = "MIC:" + self.__pack_move_order(trace)
        rcv = self.__send_order(order)
        return rcv is not None

    def move_by_joint(self, trace):
        order = "MJO:" + self.__pack_move_order(trace)
        rcv = self.__send_order(order)
        return rcv is not None

    def stop_moving(self):
        order = "PAU\r\n"
        rcv = self.__send_order(order)
        return rcv is not None

    def start(self):
        order = "STA\r\n"
        rcv = self.__send_order(order)
        return rcv is not None

    def stop(self):
        order = "STO\r\n"
        rcv = self.__send_order(order)
        return rcv is not None

    def start_EGM(self):
        order = "EMO\r\n"
        if self.__send_order(order) is not None:
            self.EGM = True
            return True
        else:
            return False

    def stop_EGM(self):
        order = "EMC\r\n"
        if self.__send_order(order) is not None:
            self.EGM = False
            return True
        else:
            return False

    def __norm(self, v1, v2):
        x = 0.0
        for i in range(len(v1)):
            x += (v1[i] - v2[i]) * (v1[i] - v2[i])
        return math.sqrt(x)

    def start_EGM_at(self, pos_euler):
        # have robot move to the EGM start position
        # wait until it arrives
        self.move_by_ground_coordinate(pos_euler)

        run_time = 0
        while run_time < 30.0:
            if self.refresh_status() and self.__norm(self.pos_euler, pos_euler[-1][0:6]) < 1:
                return self.start_EGM()
            time.sleep(0.1)
            run_time += 0.1
        return False

    def stop_EGM_at(self, pos_euler):
        if not self.stop_EGM():
            return False
        self.move_by_ground_coordinate(pos_euler)

        run_time = 0
        while run_time < 30.0:
            if self.refresh_status() and self.__norm(self.pos_euler, pos_euler[-1][0:6]) < 1:
                return True
            time.sleep(0.1)
            run_time += 0.1
        return False

    def clear(self):
        order = "ROK\r\n"
        rcv = self.__send_order(order)
        return rcv is not None


if __name__ == '__main__':
    robot = RobotABB(host="192.168.1.103", port=8002)

    robot.clear()

    key = 0
    while key != ord('e') and key != ord('E'):
        time.sleep(0.5)
        if robot.refresh_status():
            print(robot.status)
            if robot.status < 0:
                print(robot.error)
            else:
                print("Pos and euler is")
                print(robot.pos_euler)
        else:
            print("Failed to communicate with robot")
