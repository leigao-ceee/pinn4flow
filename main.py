#!/usr/bin/env python
# @Time    : 7/8/2022 11:45
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
from data_prepare import generate_data
from NS_control_model import Net, train

import time
from tqdm import trange
import torch
import torch.nn as nn
import os

name = 'NS-2d-control'
work_path = os.path.join('work', name)
isCreated = os.path.exists(work_path)
if not isCreated:
    os.makedirs(work_path)

# prepare data
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
input, field, normv = generate_data()

nodes_train = torch.tensor(input, dtype=torch.float32).to(device)
field_train = torch.tensor(field, dtype=torch.float32).to(device)
normv_train = torch.tensor(normv, dtype=torch.float32).to(device)

# 建立网络
Net_model = Net(planes=[2] + [64] * 3 + [3],).to(device)
# 损失函数
L2loss = nn.MSELoss()
# 优化算法
Optimizer = torch.optim.Adam(Net_model.parameters(), lr=0.001, betas=(0.7, 0.9))
# 下降策略
Scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=[150000, 250000], gamma=0.1)


if __name__ == '__main__':
    star_time = time.time()
    log_loss = []
    pbar = trange(30000)

    inn_var = nodes_train
    inn_var.requires_grad_(True)

    # Training
    for iter in pbar:
        learning_rate = Optimizer.state_dict()['param_groups'][0]['lr']
        train(inn_var, normv_train, field_train, Net_model, L2loss, Optimizer, Scheduler, log_loss)
        # if iter > 0 and iter % 200 == 0:
        # print('iter: {:6d}, lr: {:.3e}, eqs_loss: {:.3e}, dat_loss: {:.3e}, bon_loss1: {:.3e}, cost: {:.2f}'.
        #       format(iter, learning_rate, log_loss[-1][0], log_loss[-1][-1], log_loss[-1][1], time.time()-star_time))
        pbar.set_postfix({'lr': learning_rate, 'dat_loss': log_loss[-1][-1], 'cost:':  time.time()-star_time,
                          'eqs_loss': log_loss[-1][0], 'bcs_loss_u': log_loss[-1][1],
                          'bcs_loss_p': log_loss[-1][2], 'obj_loss': log_loss[-1][3]})
    torch.save({'log_loss': log_loss, 'model': Net_model.state_dict(), }, os.path.join(work_path, 'latest_model.pth'))