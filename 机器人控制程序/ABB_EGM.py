import clr
clr.FindAssembly("ABB_EGM.dll")
clr.AddReference("ABB_EGM")

import pygame
from abb.egm import *


class EGM(object):
    def __init__(self, IP, PortNum):
        '''
        初始化时输入端口号即可，小飞机6510；加油杆6511
        启动后等待机器人连接
        ReceiveMessage获得机器人连接状态及当前位置
        SendMessage发送计划位置给机器人
        '''
        #实例化ABB机器人动态链接库里的类
        self.ABB_EGM = ABBEGM(IP, PortNum)
        self.robot=None
    def __del__(self):
        self.ABB_EGM._udpServer.Close()
    def get_robot_pos_euler(self):
        self.robot = self.ABB_EGM.ReceiveMessage()#接受ABB机器人发送的信息，并反序列化。
        x = self.robot.FeedBack.Cartesian.Pos.X
        y = self.robot.FeedBack.Cartesian.Pos.Y
        z = self.robot.FeedBack.Cartesian.Pos.Z
        roll = self.robot.FeedBack.Cartesian.Euler.X
        pitch = self.robot.FeedBack.Cartesian.Euler.Y
        yaw = self.robot.FeedBack.Cartesian.Euler.Z
        pos = (x, y, z, yaw, pitch, roll)
        return pos

    def set_robot_pos_euler(self, pos1):
        sensor = self.ABB_EGM.CreateNewMessage()   #创建新的消息句柄
        planned = EgmPlanned.Builder()
        pos = EgmPose.Builder()
        pc = EgmCartesian.Builder()
        po = EgmEuler.Builder()
        pc.SetX(pos1[0])
        pc.SetY(pos1[1])
        pc.SetZ(pos1[2])
        po.SetX(pos1[5])
        po.SetY(pos1[4])
        po.SetZ(pos1[3])
        pos.SetPos(pc)
        pos.SetEuler(po)
        planned.SetCartesian(pos)
        sensor.SetPlanned(planned)#写入计划位置
        return self.ABB_EGM.SendMessage(sensor)


if __name__=='__main__':
    myabb=EGM(6511)
    while True:
       pos=myabb.get_robot_pos_euler()
       print(pos)
       if pos is not None:
           pos[1]+=1
           num=myabb.set_robot_pos_euler(pos)
           for i in range (6):
            print(round(pos[i],2),end="  ")
