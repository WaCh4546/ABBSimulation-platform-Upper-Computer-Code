import socket
import numpy as np


class UDPMaster(object):
    def __init__(self, local_port, remote_ip, remote_port, timeout):
        self._local_port = local_port
        self._remote_ip = remote_ip
        self._remote_port = remote_port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", local_port))
        self.socket.settimeout(timeout)

    def send_for_response(self, order):
        #pkg = np.array((1, 2)).astype(np.uint8)
        #self.socket.sendto(order.encode('gbk'), (self._remote_ip, self._remote_port))
        self.socket.sendto(order, (self._remote_ip, self._remote_port))
        try:
            rcv, addr = self.socket.recvfrom(128)
        except socket.timeout:
            pass

        for i, b in enumerate(rcv):
            if pkg[i] != b:
                return False
        return True


if __name__ == '__main__':
    a = int(-300.0)
    b = int(-1000.0)
    bytes = np.array(((a >> 8), a, (b >> 8), b), dtype=np.uint8)

    udp = UDPMaster(local_port=1234, remote_ip="127.0.0.1", remote_port=4196, timeout=20)
    ret = udp.send_for_response(bytes)
    print(ret)