

class PID(object):
    def __init__(self, kp, ki, kd, T, K, type_inc=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.T = T
        self.K = K
        self.type_inc = type_inc

        self.__e_sum = 0
        self.__lst_e = 0
        self.__lst_de = 0
        self.__lst_pos = 0

    def __out_by_pos(self, e):
        u = self.kp * e

        self.__e_sum += e
        u += self.ki * self.T * self.__e_sum

        de = (self.K * self.__lst_de + e - self.__lst_e) / (self.T + self.K)
        u += self.kd * de
        self.__lst_de = de

        return u

    def __call__(self, e):
        if self.type_inc:
            pos = self.__out_by_pos(e)
            out = pos - self.__lst_pos
            self.__lst_pos = pos
        else:
            out = self.__out_by_pos(e)
        return out

