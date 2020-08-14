#!/usr/bin/env python3

import numpy as np

W, H = 4032, 3024
X = W // 2
Y = H // 2

fov_x = fov_y = np.radians(78)

f_x = X / np.tan(fov_x / 2)
f_Y = Y / np.tan(fov_y / 2)

C = np.matrix([[f_x, 0, X], [0, f_Y, Y], [0, 0, 1]])

print(C)