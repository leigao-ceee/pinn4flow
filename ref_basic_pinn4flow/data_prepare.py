#!/usr/bin/env python
# @Time    : 7/8/2022 12:07
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import numpy as np


def generate_data():
    Nx = 151
    Ny = 101  # change to 100
    node_x = np.linspace(0.0, 1.5, Nx)[:, None]
    node_y = np.linspace(0.0, 1.0, Ny)[:, None]
    node_x = np.tile(node_x, (1, node_y.shape[0]))  # Nx x Ny
    node_y = np.tile(node_y, (1, node_x.shape[0])).T  # Nx x Ny
    input = np.stack((node_x, node_y), axis=-1)
    field = np.zeros((Nx, Ny, 3), dtype=np.float32)
    # field[0, :, 1] = 4 * input[0, :, -1] * (1 - input[0, :, -1])  # 左侧进口边界条件
    field[-1, :, 1] = 4 * input[-1, :, -1] * (1 - input[-1, :, -1])  # 控制时为出口边界
    # print(field[0, :, 1])
    field[50+1:100, 0, 2] = 0.3;  field[50+1:100, -1, 2] = 0.3  # 底/顶部进口边界条件
    field[0:51, 0, :] = 0;  field[0:51, -1, :] = 0; field[100:, 0, :] = 0 ; field[100:, -1, :] = 0  # 底/顶部壁面边界条件

    normv = np.zeros_like(input)  # 法向量
    normv[0, :, 0] = -1; normv[-1, :, 0] = 1
    normv[:, 0, 1] = -1; normv[:, -1, 1] = 1
    return input, field, normv