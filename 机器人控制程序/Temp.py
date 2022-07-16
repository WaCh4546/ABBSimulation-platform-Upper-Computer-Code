import math
import numpy as np

lx = 1040
ly = 500
d = 400
max_f = 0
max_theta = 0
for i in range(1000):
    theta = math.pi * i / (2.0 * 1000)
    f1 = (lx + d * math.cos(theta)) / (2.0 * lx)
    f2 = (ly + d * math.sin(theta)) / (2.0 * ly)
    f = math.sqrt(f1 * f1 + f2 * f2)

    if f > max_f:
        max_f = f
        max_theta = theta

print(max_f / 2)
print(max_theta * 180 / math.pi)