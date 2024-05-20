from math import sin, cos, pi, exp, e
from numpy import arange, random
from tqdm import trange

"""
初始化未知参数
"""
phi_0 = 58.63 / 180 * pi
x4_0 = 1
t_max = 0.01
"""
初始参数
"""
alpha = 22 / 180 * pi
g = 9.7833
m = 0.1212
r = 0.137
"""
周期性边界条件损失函数
"""


def Loss(x3_t, x4_t):
    return (((x3_t - 131.37 / 180 * pi - phi_0) / x3_t) ** 2 + ((x4_t - x4_0) / x4_t) ** 2)/2


"""
4RK函数
"""


def RK_4(h, y, f):
    k1 = h * f(t, x1, x2, x3, x4)
    k2 = h * f(t + 0.5 * h, x1 + 0.5 * k1, x2 + 0.5 * k1, x3 + 0.5 * k1, x4 + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, x1 + 0.5 * k2, x2 + 0.5 * k2, x3 + 0.5 * k2, x4 + 0.5 * k2)
    k4 = h * f(t + h, x1 + 0.5 * k3, x2 + 0.5 * k3, x3 + 0.5 * k3, x4 + 0.5 * k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


"""
微分方程拆解
"""


def Dx1(t, x1, x2, x3, x4):
    return x2


def Dx2(t, x1, x2, x3, x4):
    return g / r *(cos(x1)*cos(x3)*cos(alpha)-sin(x1)*cos(alpha))-0.5*x4**2*sin(2*x1)


def Dx3(t, x1, x2, x3, x4):
    return x4


def Dx4(t, x1, x2, x3, x4):
    return -g*sin(x3)*sin(alpha) / r / sin(x1)


"""
主循环
"""
if __name__ == '__main__':
    low_loss = 1E3
    for i in trange(100):
        for phi_k in range(100):
            phi_1 = phi_0 + (random.random() - 0.5) * 0.01
            x4_1 = x4_0
            t_max_1 = t_max
            t = 0
            dt = 0.00001  # 步长
            # 初始化参数
            x1 = 51.88 / 180 * pi
            x2 = 0
            x3 = phi_1 - 48.63 / 180 * pi
            x4 = x4_1
            for j in arange(0, t_max_1, dt):
                x1 = RK_4(dt, x1, Dx1)
                x2 = RK_4(dt, x2, Dx2)
                x3 = RK_4(dt, x3, Dx3)
                x4 = RK_4(dt, x4, Dx4)
            loss = Loss(x3, x4)
            if loss < low_loss and 48.63 / 180 * pi < phi_1 < pi / 2:
                low_loss = loss
                phi_0 = phi_1
            elif phi_1 >= pi / 2 or phi_1 <= 48.63 / 180 * pi:
                phi_1 = ((random.random() - 0.5) * (pi / 2 - 48.63 / 180 * pi)) + 48.63 / 180 * pi
        for x4_k in range(100):
            phi_1 = phi_0
            x4_1 = x4_0 + (random.random() - 0.5) * 0.1
            t_max_1 = t_max
            dt = 0.00001  # 步长
            # 初始化参数
            x1 = 51.88 / 180 * pi
            x2 = 0
            x3 = phi_1 - 48.63 / 180 * pi
            x4 = x4_1
            for j in arange(0, t_max_1, dt):
                x1 = RK_4(dt, x1, Dx1)
                x2 = RK_4(dt, x2, Dx2)
                x3 = RK_4(dt, x3, Dx3)
                x4 = RK_4(dt, x4, Dx4)
            loss = Loss(x3, x4)
            if loss < low_loss and 10 > x4_1 > 0:
                low_loss = loss
                x4_0 = x4_1
            elif x4_1 <= 0 or x4_1 >= 100:
                x4_1 = ((random.random() - 0.5) * 1) + 4
        for t_k in range(100):
            phi_1 = phi_0
            x4_1 = x4_0
            t_max_1 = t_max + (random.random() - 0.5) * 0.005
            dt = 0.00001  # 步长
            # 初始化参数
            x1 = 51.88 / 180 * pi
            x2 = 0
            x3 = phi_1 - 48.63 / 180 * pi
            x4 = x4_1
            for j in arange(0, t_max_1, dt):
                x1 = RK_4(dt, x1, Dx1)
                x2 = RK_4(dt, x2, Dx2)
                x3 = RK_4(dt, x3, Dx3)
                x4 = RK_4(dt, x4, Dx4)
            loss = Loss(x3, x4)
            if loss < low_loss and t_max_1 > 0:
                low_loss = loss
                t_max = t_max_1
            elif t_max_1 <= 0:
                t_max_1 = random.random() * 0.005 + 0.01
        if i % 1 == 0:
            print(f"low_loss:{low_loss}")
            print(f"t_max:{t_max}")
            print(f"phi_0:{phi_0}")
            print(f"x4_0:{x4_0}")
            print(f"V_t:{x4_0 * r * cos(pi / 2 - phi_0)}")
