#!/usr/bin/env python
# @Time    : 7/8/2022 13:08
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import matplotlib.pyplot as plt
import numpy as np
from visual_data import matplotlib_vision
from NS_control_model import inference
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    name = 'NS-2d-control'
    work_path = os.path.join('work', name)
    result = torch.load(os.path.join(work_path, 'latest_model.pth'))
    log_loss = result['log_loss']
    # 可视化
    Visual = matplotlib_vision('/', input_name=('x', 'y'), field_name=('p','u','v',))
    plt.figure(2, figsize=(10, 5))
    plt.clf()
    Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
    Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'bcs_loss_u')
    Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'bcs_loss_p')
    Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'obj_loss')
    Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
    plt.title('training loss')
    plt.show()