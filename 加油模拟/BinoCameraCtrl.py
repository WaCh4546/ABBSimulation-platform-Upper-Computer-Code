import socket
import time


class BitoCameraCtrl(object):
    def __init__(self, IP, Port):
        self.__IP = IP
        self.__Port = Port
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #self.__socket.bind(("127.0.0.1", 20005))
        self.__socket.settimeout(5)

    def send(self, move, rotate):
        # send order
        str = "%1d%1d" % (move, rotate)
        self.__socket.sendto(str.encode('utf-8'), (self.__IP, self.__Port))

        # wait for response
        #data, addr = self.__socket.recvfrom(32)
        #print(addr)
        #print(data)


if __name__ == '__main__':
    PLC = BitoCameraCtrl(IP="192.168.1.101", Port=20001)
    while True:
        time.sleep(0.5)
        PLC.send(move=0, rotate=0)