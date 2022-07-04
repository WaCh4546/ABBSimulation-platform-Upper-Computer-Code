import socket
import numpy as np
import time
from enum import IntEnum

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
    MOVE_FWD = 4,
    MOVE_BKWD = 5


class Axis7(object):
    def __init__(self, local_port, remote_ip, remote_port, timeout=0.1):
        self._local_port = local_port
        self._remote_ip = remote_ip
        self._remote_port = remote_port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", local_port))
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
            rcv, addr = self.socket.recvfrom(128)
            if rcv[0] != self.cmd:
                return False

            self.position = (rcv[2] << 8) | rcv[3]
            self.zeroed = rcv[5]
            self.limited = rcv[7]
            return True
        except socket.timeout:
            self.timeout = True
        return False


if __name__ == '__main__':
    axis = Axis7("127.0.0.1", 20001)
    axis.cmd = 1
    axis.speed = 10
    while True:
        time.sleep(1)
        axis.refresh()

