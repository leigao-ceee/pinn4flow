import matplotlib.tri
import numpy as np
import torch
import torch.nn as nn
from basic_model import DeepModel_single, DeepModel_multi, gradients
from visual_data import matplotlib_vision
import random
import time
from tqdm import trange
import matplotlib.pyplot as plt
import os


#################################### 定义网络框架 ###################################################################
# 输入 inn_var : x, y, t
# 输出 out_var : p, u, v, s11, s12, s22
class Net(DeepModel_single):
    def __init__(self, planes, rho, miu):
        super(Net, self).__init__(planes, active=nn.Tanh())
        self.rho = rho
        self.miu = miu

    # 将网络的直接输出 psi ,p, s11, s12, s22 转化为 p, u, v, s11, s12, s22
    def output_transform(self, inn_var, out_var):
        psi, p, s11, s22, s12 = \
            out_var[..., 0:1], out_var[..., 1:2], out_var[..., 2:3], out_var[..., 3:4], out_var[..., 4:5]

        w = gradients(psi, inn_var)
        u, v = w[..., 1:2], -w[..., 0:1]
        return torch.cat((p, u, v, s11, s12, s22), dim=-1)

    # 计算残差
    def equation(self, inn_var, out_var):
        p, u, v, s11, s12, s22 = out_var[..., (0,)], out_var[..., (1,)], out_var[..., (2,)], \
                                 out_var[..., (3,)], out_var[..., (4,)], out_var[..., (5,)]
        dpda, duda, dvda = gradients(p, inn_var), gradients(u, inn_var), gradients(v, inn_var)
        dpdy = dpda[..., 1:2]
        dudx, dudy, dudt, dvdx, dvdy, dvdt = duda[..., 0:1], duda[..., 1:2], duda[..., 2:3], \
                                             dvda[..., 0:1], dvda[..., 1:2], dvda[..., 2:3]

        s11_1 = gradients(s11, inn_var)[..., 0:1]
        s12_2 = gradients(s12, inn_var)[..., 1:2]
        s22_2 = gradients(s22, inn_var)[..., 1:2]
        s12_1 = gradients(s12, inn_var)[..., 0:1]

        eq_p = p + (s11 + s22) / 2
        eq_u = self.rho * dudt + self.rho * (u*dudx + v*dudy) - s11_1 - s12_2
        eq_v = self.rho * dvdt + self.rho * (u*dvdx + v*dvdy) - s12_1 - s22_2
        eq_s11 = -p + 2*self.miu*dudx - s11
        eq_s22 = -p + 2*self.miu*dvdy - s22
        eq_s12 = self.miu*(dudy+dvdx) - s12
        eqs = torch.cat((eq_p, eq_u, eq_v, eq_s11, eq_s12, eq_s22), dim=-1)

        return eqs, torch.cat((dpdy, dudy, dvdy), dim=-1)


######################## 获取 nodes 在 box 流域内的边界节点  ########################
def BCS_ICS(nodes, box):
    BCS = []
    Num_Nodes = nodes.shape[0]
    Index = np.arange(Num_Nodes)

    BCS.append(Index[nodes[:, 0] == box[0]])  # inlet
    BCS.append(Index[nodes[:, 0] == box[2]])  # outlet
    BCS.append(Index[nodes[:, 1] == box[1]])  # top
    BCS.append(Index[nodes[:, 1] == box[3]])  # bottom
    BCS.append(Index[np.abs((nodes[:, 0]-0.2)**2 + (nodes[:, 1]-0.2)**2 - (D/2)**2) < 1e-7])  # cylinder wall

    if nodes.shape[-1] == 3:
        BCS.append(Index[nodes[:, 2] == 0])  # initial

    return BCS


######################## 读取数据  ########################
def read_data(**kwargs):
    import scipy.io as sio
    if kwargs['steady']:
        data = sio.loadmat('..\\data\\2D_cylinder\\mixed\\steady_data.mat')
        INLET, OUTLET, WALL= data['INLET'][..., :2], data['OUTLET'], data['WALL']
        num = INLET.shape[0] + OUTLET.shape[0] + WALL.shape[0]
        XY_c  = data['XY_c'][:-num]
        data = sio.loadmat('..\\data\\2D_cylinder\\mixed\\steady_Fluent.mat')
        fields_fluent= np.squeeze(data['field']).T[..., (0, 1, 4, 2, 3)]
        return XY_c, INLET, OUTLET, WALL, fields_fluent
    else:
        data = sio.loadmat('..\\data\\2D_cylinder\\mixed\\unsteady_data.mat')
        INLET, OUTLET, WALL, INITIAL= data['INB'][..., :3], data['OUTB'], data['WALL'], data['IC']
        num = INLET.shape[0] + OUTLET.shape[0] + WALL.shape[0]
        XY_c  = data['XY_c'][:-num]
    return XY_c, INLET, OUTLET, WALL, INITIAL


################################## 单次训练步骤  ##################################
def train(inn_var, BCs, out_true, model, Loss, optimizer, scheduler, log_loss):

    BC_in = BCs[0]
    BC_out = BCs[1]
    BC_bot = BCs[2]
    BC_top = BCs[3]
    BC_wall = BCs[4]
    BC_inital = BCs[-1]

    def closure():
        inn_var.requires_grad_(True)
        optimizer.zero_grad()
        out_var = model(inn_var)
        out_var = model.output_transform(inn_var, out_var)
        res_i, _ = model.equation(inn_var, out_var)
        out_var = out_var[..., 0:3]
        
        y_inb = inn_var.detach()[BC_in, 1:2]
        t_inb = inn_var.detach()[BC_in, 2:3]
        # u_inb = 4*U_max*y_inb*(0.41-y_inb)/(0.41**2)*(np.sin(2*3.1416*t_inb/T+3*3.1416/2)+1.0)
        # v_inb = np.zeros_like(x_inb)
        

        bcs_loss_in = Loss(out_var[BC_in, 1:], 
                           torch.cat((4*U_max*y_inb*((Box[-1] - Box[1])-y_inb)/((Box[-1] - Box[1])**2)*
                                      (torch.sin(torch.pi*t_inb/tmax+3*torch.pi/2)+1.0), 0 * y_inb), dim=-1))
        # bcs_loss_in = Loss(out_var[BC_in, 1:], torch.cat((torch.ones_like(y_inb), 0*y_inb), dim=-1)) # u = 1
        bcs_loss_out = (out_var[BC_out, 0] ** 2).mean()
        bcs_loss_wall = (out_var[BC_wall, 1:] ** 2).mean()
        bcs_loss_top = (out_var[BC_top, 1:]**2).mean()
        bcs_loss_bot = (out_var[BC_bot, 1:]**2).mean()
        bcs_loss_ini = (out_var[BC_inital, :] ** 2).mean()
        bcs_loss = bcs_loss_in * 5 + bcs_loss_out + bcs_loss_wall * 5 + bcs_loss_top + bcs_loss_bot + bcs_loss_ini

        eqs_loss = (res_i ** 2).mean()

        loss_batch = bcs_loss * 1. + eqs_loss
        loss_batch.backward()

        # data_loss = Loss(out_var, out_true)
        log_loss.append([eqs_loss.item(), bcs_loss.item(),
                         bcs_loss_wall.item(), bcs_loss_top.item(), bcs_loss_bot.item(), bcs_loss_in.item(),
                         bcs_loss_out.item(), bcs_loss_ini.item(),
                         0])

        return loss_batch

    optimizer.step(closure)
    scheduler.step()


################################## 预测  ##################################
def inference(inn_var, model):
    inn_var = inn_var.cuda()
    inn_var.requires_grad_(True)
    out_var = model(inn_var)
    out_var = model.output_transform(inn_var, out_var)
    equation, _ = model.equation(inn_var, out_var)
    return out_var.detach().cpu(), equation.detach().cpu()


if __name__ == '__main__':

    name = 'trans-cylinder-2d-mixed-'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    #################### 定义问题相关参数 ####################
    Rho, Miu, D = 1.0, 0.005,  0.1
    U_max, tmax = 0.5, 0.5  # 入口流速的最大值以及非定常周期 tmax = T/2
    Box = [0, 0, 1.1, 0.41]  # 矩形流域
    
    #################### 读入数据 ####################
    data = read_data(steady=False)
    data = list(map(np.random.permutation, data)) # np.random.shuffle & random.shuffle 返回None,此外， python 3 中map返回的是迭代器
    XY_c = np.concatenate(data, 0)
    BCs = BCS_ICS(XY_c, Box)  # 获得不同边界节点的index
    input = torch.tensor(XY_c[:, :3], dtype=torch.float32).to(device)
    # 采用三角形 对非结构化网格建立节点连接关系
    triang = matplotlib.tri.Triangulation(data[-1][:, 0], data[-1][:, 1])
    triang.set_mask(np.hypot(data[-1][triang.triangles, 0].mean(axis=1) - 0.2,
                             data[-1][triang.triangles, 1].mean(axis=1) - 0.2) < D/2)
    # plt.figure(1, figsize=(20, 5))
    # t = plt.tricontourf(triang, fields_fluent[:, 2])
    # plt.axis('equal')
    # plt.show()

    #################### 定义损失函数、优化器以及网络结构 ####################
    L2Loss = nn.MSELoss().cuda()
    Net_model = Net(planes=[3] + 3 * [50] + [5], rho=Rho, miu=Miu).to(device)
    Optimizer1 = torch.optim.Adam(Net_model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    Boundary_epoch1 = [200]
    Scheduler1 = torch.optim.lr_scheduler.MultiStepLR(Optimizer1, milestones=Boundary_epoch1, gamma=0.1)
    Optimizer2 = torch.optim.LBFGS(Net_model.parameters(), lr=1.)
    Boundary_epoch2 = [500]
    Scheduler2 = torch.optim.lr_scheduler.MultiStepLR(Optimizer2, milestones=Boundary_epoch2)
    Visual = matplotlib_vision('/', field_name=('p', 'u', 'v'), input_name=('x', 'y'))
    # self.optimizer = optim.LBFGS(self.model.parameters(), lr=lr)
    ################################### 训练 #####################################
    star_time = time.time()
    log_loss = []
    """load a pre-trained model"""
    # Net_model.loadmodel(res_path + 'latest_model.pth')
    for epoch in range(Boundary_epoch2[-1]):

        #如果GPU内存不充足，可以分批次进行训练
        #iter = 10
        # for i in range(iter):
        #     XY_c = np.concatenate(list(map(lambda x: x[i*int(x.shape[0]/iter):(i+1)*int(x.shape[0]/iter)], data)), 0)
        #     BCs = BCS_ICS(XY_c, Box)
        #     input = torch.tensor(XY_c[:, :3], dtype=torch.float32).to(device)
        #     train(input, BCs, 0, Net_model, L2Loss, Optimizer, Scheduler, log_loss)
        if epoch < 200:
            learning_rate = Optimizer1.state_dict()['param_groups'][0]['lr']
            train(input, BCs, 0, Net_model, L2Loss, Optimizer1, Scheduler1, log_loss)
        else:
            learning_rate = Optimizer2.state_dict()['param_groups'][0]['lr']
            train(input, BCs, 0, Net_model, L2Loss, Optimizer2, Scheduler2, log_loss)
        if epoch > 0 and epoch % 50 == 0:
            print('epoch: {:6d}, lr: {:.1e}, cost: {:.2e}, dat_loss: {:.2e}, eqs_loss: {:.2e}, bcs_loss: {:.2e}'.
                  format(epoch, learning_rate, time.time() - star_time,
                         log_loss[-1][-1], log_loss[-1][0], log_loss[-1][1],))
            star_time = time.time()

            # 损失曲线
            plt.figure(1, figsize=(15, 5))
            plt.clf()
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'bcs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, -1], 'dat_loss')
            plt.savefig(os.path.join(work_path, 'log_loss.svg'))

            # 详细的损失曲线
            plt.figure(2, figsize=(15, 10))
            plt.clf()
            plt.subplot(211)
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'eqs_loss')
            plt.subplot(212)
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'bcs_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'wall_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'top_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'bot_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 5], 'in_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 6], 'out_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 7], 'ini_loss')
            plt.savefig(os.path.join(work_path, 'detail_loss.svg'))


            # 根据模型预测流场， 若有真实场，则与真实场对比
            input_visual_p = torch.tensor(data[-1][..., :3], dtype=torch.float32)  # 取初场的空间坐标
            input_visual_p[:, -1] = input_visual_p[:, -1] + tmax   # 时间取最大
            field_visual_p, _ = inference(input_visual_p, Net_model)
            # field_visual_t = data[-1][..., 3:].cpu().numpy()
            field_visual_p = field_visual_p.cpu().numpy()[..., 0:3]
            field_visual_t = field_visual_p

            plt.figure(3, figsize=(30, 8))
            plt.clf()
            Visual.plot_fields_tr(field_visual_t, field_visual_p, input_visual_p.detach().cpu().numpy(), triang)
            # plt.savefig(res_path + 'field_' + str(t) + '-' + str(epoch) + '.jpg')
            plt.savefig(os.path.join(work_path, 'global_' + str(epoch) + '.jpg'), dpi=200)
            plt.savefig(os.path.join(work_path, 'global_now.jpg'))

            torch.save({'epoch': epoch, 'model': Net_model.state_dict(), }, os.path.join(work_path, 'latest_model.pth'))
