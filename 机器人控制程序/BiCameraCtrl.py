import socket
import numpy as np


class BiCameraCtrl(object):
    """
    BiCameraCtrl is used to communicate with the PLC of the bino-camera to adjust the two cameras.
    The order format:
    Byte 0: Transportation of the right camera. 0: still, 1: move left, 2: move right
    Byte 1: Rotation of the right camera. 0: still, 1: clockwise, 2: anticlockwise
    The PLC send back what it receives.
    """
    def __init__(self, local_port, remote_ip, remote_port, timeout=0.1):
        self._local_port = local_port
        self._remote_ip = remote_ip
        self._remote_port = remote_port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("192.168.1.100", local_port))
        self.socket.settimeout(timeout)

    def send_order(self, trans, rotate):
        order = np.array((trans, rotate)).astype(np.uint8)
        self.socket.sendto(order, (self._remote_ip, self._remote_port))

        try:
           rcv, addr = self.socket.recvfrom(32)
           for i, byte in enumerate(rcv):
               if byte != order[i]:
                   return False
        except socket.timeout:
            return False
        return True


if __name__ == '__main__':
    udp = BiCameraCtrl(local_port=20005, remote_ip="192.168.1.101", remote_port=20001, timeout=1)
    try:
        ret = udp.send_order(trans=2, rotate=1)
        print(ret)
    except socket.timeout:
        print("timeout!")