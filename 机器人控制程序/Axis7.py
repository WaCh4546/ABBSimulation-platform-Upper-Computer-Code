import socket
import numpy as np
import time
from enum import IntEnum
import cv2

"""
Order sent by PC:
--------------------------------------------
 WORD 0  |  WORD 1  | WORD 2 -- WORD 3
--------------------------------------------
  CMD    |  SPEED   |    RESERVED
--------------------------------------------
There are 6 commands:
0: Stop.
1: Normal control. The axis must have been zeroed.
2: Go zero.
3: Go to middle position.
4: Move forward.
5: Move backward.

Fdk send by PLC:
--------------------------------------------
 WORD 0  |  WORD 1  |  WORD 2  |  WORD 3
--------------------------------------------
  CMD    | POSITION |  ZEROED  | LIMITED
--------------------------------------------
CMD returns the command that PC has just sent.
POSITION is the position of the robot. If the 
"""


class AXIS7_CMD(IntEnum):
    STOP = 0,
    NORMAL_CTRL = 1,
    GO_ZERO = 2,
    GO_MIDDLE = 3,
    MOVE_BKWD = 4,
    MOVE_FWD = 5


class Axis7(object):
    def __init__(self, local_port, remote_ip, remote_port, timeout=0.1):
        self._local_port = local_port
        self._remote_ip = remote_ip
        self._remote_port = remote_port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("192.168.1.101", local_port))
        self.socket.settimeout(timeout)

        self.speed = 0
        self.cmd = AXIS7_CMD.STOP

        self.position = -1
        self.zeroed = False
        self.limited = False
        self.timeout = False

    def refresh(self):
        # send order to PLC
        order = np.array((self.cmd, 0x0, self.speed >> 8, self.speed, 0, 0, 0, 0)).astype(np.uint8)
        self.socket.sendto(order, (self._remote_ip, self._remote_port))

        try:
            self.timeout = False
            rcv, addr = self.socket.recvfrom(32)
            if rcv[0] != self.cmd:
                return False

            position = int((rcv[2] << 8) | rcv[3])
            self.position = position if not position or position < 0x8000 else position - 0x10000
            self.zeroed = rcv[5]
            self.limited = rcv[7]
            return True
        except socket.timeout:
            self.timeout = True
            print("timeout")
        return False


if __name__ == '__main__':
    axis = Axis7(local_port=2002, remote_ip="192.168.1.105", remote_port=2001)
    axis.cmd = AXIS7_CMD.STOP
    axis.speed = 0
    while True:
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(img, "Robot zeroed = %d, limited = %d, position = %d" % (axis.zeroed, axis.limited, axis.position),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "0: Stop robot", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "1: Normal control", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "2: Go zero", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "3: Go middle", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "4: Go backward", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "5: Go forward", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "A: Accelerate", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "D: Decelerate", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.putText(img, "e: Exit", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)

        cv2.imshow("", img)

        key = cv2.waitKey(100)
        if key == ord('0'):
            axis.cmd = AXIS7_CMD.STOP
        elif key == ord('1'):
            axis.cmd = AXIS7_CMD.NORMAL_CTRL
        elif key == ord('2'):
            axis.cmd = AXIS7_CMD.GO_ZERO
        elif key == ord('3'):
            axis.cmd = AXIS7_CMD.GO_MIDDLE
        elif key == ord('4'):
            axis.cmd = AXIS7_CMD.MOVE_BKWD
        elif key == ord('5'):
            axis.cmd = AXIS7_CMD.MOVE_FWD
        elif key == ord('a') or key == ord('A'):
            axis.speed += 1
        elif key == ord('d') or key == ord('D'):
            axis.speed -= 1
        elif key == ord('e') or key == ord('E'):
            break

        axis.refresh()

