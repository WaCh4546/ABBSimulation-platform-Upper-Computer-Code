import clr
clr.FindAssembly("ABB_EGM.dll")
clr.AddReference("ABB_EGM")
from abb.egm import *
import time
class EGM():
    def __init__(self,IP,PortNum):
        '''
        初始化时输入端口号即可，小飞机6510；加油杆6511
        启动后等待机器人连接
        ReceiveMessage获得机器人连接状态及当前位置
        SendMessage发送计划位置给机器人
        '''
        #实例化ABB机器人动态链接库里的类
        self.ABB_EGM=ABBEGM(IP,PortNum)
        self.robot=None
        self.sensor=None
        #获取ABB机器人连接状态
    def ReceiveMessage(self):
        self.robot=self.ABB_EGM.ReceiveMessage()#接受ABB机器人发送的信息，并反序列化。
        status=self.robot.RapidExecState.State
        if status==EgmRapidCtrlExecState.Types.RapidCtrlExecStateType.RAPID_RUNNING:
            X = self.robot.FeedBack.Cartesian.Pos.X;
            Y = self.robot.FeedBack.Cartesian.Pos.Y;
            Z = self.robot.FeedBack.Cartesian.Pos.Z;
            ROLL = self.robot.FeedBack.Cartesian.Euler.X;
            PITCH = self.robot.FeedBack.Cartesian.Euler.Y
            YAW = self.robot.FeedBack.Cartesian.Euler.Z
            pos=[X,Y,Z,ROLL,PITCH,YAW]
            return True,pos
        else:
            return False,None


        #写入待移动的位置信息，并发送消息给机器人
    def SendMessage(self,pos1):
        self.sensor=self.ABB_EGM.CreateNewMessage()#创建新的消息句柄
        planned = EgmPlanned.Builder()
        pos=EgmPose.Builder()
        pc = EgmCartesian.Builder()
        po = EgmEuler.Builder()
        pc.SetX(pos1[0])
        pc.SetY(pos1[1])
        pc.SetZ(pos1[2])
        po.SetX(pos1[3])
        po.SetY(pos1[4])
        po.SetZ(pos1[5])
        pos.SetPos(pc)
        pos.SetEuler(po)
        planned.SetCartesian(pos)
        self.sensor.SetPlanned(planned)#写入计划位置
        return self.ABB_EGM.SendMessage(self.sensor)
    def GetrobotJoint(self):
        self.robot=self.ABB_EGM.ReceiveMessage()#接受ABB机器人发送的信息，并反序列化。
        status=self.robot.RapidExecState.State
        if status==EgmRapidCtrlExecState.Types.RapidCtrlExecStateType.RAPID_RUNNING:
            X = self.robot.FeedBack.Cartesian.Pos.X;
            Y = self.robot.FeedBack.Cartesian.Pos.Y;
            Z = self.robot.FeedBack.Cartesian.Pos.Z;
            ROLL = self.robot.FeedBack.Cartesian.Euler.X;
            PITCH = self.robot.FeedBack.Cartesian.Euler.Y
            YAW = self.robot.FeedBack.Cartesian.Euler.Z
            joint=[X,Y,Z,ROLL,PITCH,YAW]
            return True,joint
        else:
            return False,None

if __name__=='__main__':
    myabb=EGM("192.168.1.103",6511)
    num1=0
    while True:
       a = time.time()
       flag,pos=myabb.ReceiveMessage()
       num1+=10
       print(num1)
       if flag:
           pos[0]+=5
           #joint=myabb.robot.FeedBack.Joints.JointsList
           #joint=list(joint)
           #print(joint)

           num=myabb.SendMessage(pos)
           b=time.time()
           print ((b-a) * 1000.0)
           for i in range (6):
            print(round(pos[i],2),end="  ")
       #time.sleep(0.4)

