import numpy as np
import paddle
import paddle.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import time
import os
import scipy.io as sio
# from basic_model import numpy_32, tensor_32
from basic_model_paddle import PaddleModel_single, gradients
from visual_data import matplotlib_vision


# 输入 inn_var : x, y, t
# 输出 out_var : p, u, v, s11, s12, s22
class Net(PaddleModel_single):
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
        return paddle.concat((p, u, v, s11, s12, s22), axis=-1)

    # 计算残差
    def equation(self, inn_var, out_var):
        p, u, v, s11, s12, s22 = out_var[..., 0:1], out_var[..., 1:2], out_var[..., 2:3], \
                                 out_var[..., 3:4], out_var[..., 4:5], out_var[..., 5:6]
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
        eqs = paddle.concat((eq_p, eq_u, eq_v, eq_s11, eq_s12, eq_s22), axis=-1)

        return eqs, paddle.concat((dpdy, dudy, dvdy), axis=-1)

######################## 获取 nodes 在 box 流域内的边界节点  ########################
def BCS_ICS(nodes, box):
    BCS = []
    Num_Nodes = nodes.shape[0]
    Index = np.arange(Num_Nodes)

    BCS.append(Index[nodes[:, 0] == box[0]])  # inlet
    BCS.append(Index[nodes[:, 0] == box[2]])  # outlet
    BCS.append(Index[nodes[:, 1] == box[1]])  # top
    BCS.append(Index[nodes[:, 1] == box[3]])  # bottom
    BCS.append(Index[np.abs((nodes[:, 0]-0)**2 + (nodes[:, 1]-0)**2 - (D/2)**2) < 1e-7])  # cylinder wall

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

def read_paddle_data(num_time):
    file_path = '..\\data\\2D_cylinder\\paddle_openfoam\\'
    cyl = np.loadtxt(file_path + 'domain_cylinder.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)] #xypuv
    inlet = np.loadtxt(file_path + 'domain_inlet.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)]
    outlet = np.loadtxt(file_path + 'domain_outlet.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)]
    train = np.loadtxt(file_path + 'domain_train.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)]
    initial = np.loadtxt(file_path + 'initial/ic0.1.csv', skiprows=1, delimiter=',')[..., (4, 5, 0, 1, 2,)]
    ind_bot = initial[:, 1] == -20.
    bot = initial[ind_bot]
    ind_top = initial[:, 1] == 20.
    top = initial[ind_top]

    # plt.figure(1)
    # plt.plot(cyl[:, 0], cyl[:, 1], 'r.')
    # plt.plot(inlet[:, 0], inlet[:, 1], 'b.')
    # plt.plot(outlet[:, 0], outlet[:, 1], 'b.')
    # plt.plot(train[:, 0], train[:, 1], 'g.')
    # plt.show()
    # plt.figure(2)
    # plt.plot(train[:, 0], train[:, 1], 'k.')
    # plt.show()
    # plt.figure(2)
    # plt.subplot(221)
    # plt.scatter(initial[:, 0], initial[:, 1], s=0.1, c=initial[:, 2])
    # plt.subplot(222)
    # plt.scatter(initial[:, 0], initial[:, 1], s=0.1, c=initial[:, 3])
    # plt.subplot(223)
    # plt.scatter(initial[:, 0], initial[:, 1], s=0.1, c=initial[:, 4])
    # plt.show()
    # plt.figure(3)
    # plt.plot(train[:, 0], train[:, 1], 'g.')
    probe = []
    times_list_all = []
    dirs = os.listdir(file_path + 'probe/')
    #####获取时间
    # for file in dirs:
    #     time = float(file[5:-4])
    #     times_list_all.append(time)
    # times_list_all = np.array(times_list_all)


    times_list = np.arange(1, 51)  # np.random.choice(times_list_all, num_time)


    # times_list = np.random.choice(times_list_all, num_time)


    for time in times_list:
        data = np.loadtxt(file_path + '/probe/probe0.' + str(time) + '.csv', skiprows=1, delimiter=',')[..., (5, 6, 0, 1, 2,)]
        t_len = data.shape[0]
        supervised_t = np.array([time] * t_len).reshape((-1, 1))
        data = np.concatenate((data[..., (0, 1)], supervised_t, data[..., (2, 3, 4)]), axis=1)
        probe.append(data)

    full_supervised_data = np.concatenate(probe)

    inlet = replicate_time_list(times_list, inlet.shape[0],  inlet)
    outlet = replicate_time_list(times_list, outlet.shape[0],  outlet)
    bot = replicate_time_list(times_list, bot.shape[0],  bot)
    top = replicate_time_list(times_list, top.shape[0],  top)
    initial = replicate_time_list([1], initial.shape[0],  initial)
    cyl = replicate_time_list(times_list, cyl.shape[0],  cyl)
    train = replicate_time_list(times_list, train.shape[0],  train)

    return train,  inlet, outlet, top, bot, cyl, full_supervised_data, initial


def replicate_time_list(time_list, domain_shape, spatial_data):
    all_t = []
    count = 0
    all_data = []
    for t in time_list:
        tmp_t = [t] * domain_shape
        all_t.append(tmp_t)
        tmp = spatial_data
        all_data.append(tmp)
    replicated_t = np.array(all_t).reshape(-1, 1)
    spatial_data = np.concatenate(all_data)
    spatial_data = np.concatenate((spatial_data[..., (0, 1)], replicated_t, spatial_data[..., (2, 3, 4)]), axis=1)
    return spatial_data


################################## 单次训练步骤  ##################################
def train_adam(inn_var, BCs, out_true, model, Loss, optimizer, scheduler, log_loss):
    BC_in = paddle.to_tensor(BCs[0][..., 0:3], dtype='float32', place='gpu:0')
    BC_out = paddle.to_tensor(BCs[1][..., 0:3], dtype='float32', place='gpu:0')
    BC_top = paddle.to_tensor(BCs[2][..., 0:3], dtype='float32', place='gpu:0')
    BC_bot = paddle.to_tensor(BCs[3][..., 0:3], dtype='float32', place='gpu:0')
    BC_wall = paddle.to_tensor(BCs[4][..., 0:3], dtype='float32', place='gpu:0')
    BC_initial = paddle.to_tensor(BCs[-1][..., 0:3], dtype='float32', place='gpu:0')
    field_supervised = paddle.to_tensor(out_true[..., 0:3], dtype='float32', place='gpu:0')

    BC_in_m = paddle.to_tensor(BCs[0][..., 3:], dtype='float32', place='gpu:0')
    BC_out_m = paddle.to_tensor(BCs[1][..., 3:], dtype='float32', place='gpu:0')
    BC_top_m = paddle.to_tensor(BCs[2][..., 3:], dtype='float32', place='gpu:0')
    BC_bot_m = paddle.to_tensor(BCs[3][..., 3:], dtype='float32', place='gpu:0')
    BC_wall_m = paddle.to_tensor(BCs[4][..., 3:], dtype='float32', place='gpu:0')
    BC_initial_m = paddle.to_tensor(BCs[-1][..., 3:], dtype='float32', place='gpu:0')
    field_supervised_m = paddle.to_tensor(out_true[..., 3:], dtype='float32', place='gpu:0')


    inn_var.stop_gradient = False
    BC_in.stop_gradient = False
    BC_out.stop_gradient = False
    BC_top.stop_gradient = False
    BC_bot.stop_gradient = False
    BC_wall.stop_gradient = False
    BC_initial.stop_gradient = False
    field_supervised.stop_gradient = False
    optimizer.clear_grad()

    out_var = model(inn_var)
    out_var = model.output_transform(inn_var, out_var)
    res_i, _ = model.equation(inn_var, out_var)
    out_var = out_var[..., 0:3]

    ##inlet loss  u,v

    pred_in = model(BC_in)
    pred_in = model.output_transform(BC_in, pred_in)
    bcs_loss_in = Loss(pred_in[..., 1:3], BC_in_m[...,  1:3])

    ##outlet loss p
    pred_out = model(BC_out)
    pred_out = model.output_transform(BC_out, pred_out)
    bcs_loss_out = Loss(pred_out[..., 0], BC_out_m[..., 0])

    ##wall loss u,v
    pred_top = model(BC_top)
    pred_top = model.output_transform(BC_top, pred_top)
    bcs_loss_top = Loss(pred_top[..., 1:3], BC_top_m[..., 1:3])

    pred_bot = model(BC_bot)
    pred_bot = model.output_transform(BC_bot, pred_bot)
    bcs_loss_bot = Loss(pred_bot[..., 1:3], BC_bot_m[..., 1:3])

    pred_wall = model(BC_wall)
    pred_wall = model.output_transform(BC_wall, pred_wall)
    bcs_loss_wall = Loss(pred_wall[...,  1:3], BC_wall_m[...,  1:3])
    ##initial loss u v p
    pred_initial = model(BC_initial)
    pred_initial = model.output_transform(BC_initial, pred_initial)
    bcs_loss_initial = Loss(pred_initial[..., :3], BC_initial_m[..., :3])

    bcs_loss = bcs_loss_in * 10 + bcs_loss_out + bcs_loss_top * 10 + bcs_loss_bot * 10 \
               + bcs_loss_wall * 10 + bcs_loss_initial * 10

    ## supervised loss
    pred_field = model(field_supervised)
    pred_field = model.output_transform(field_supervised, pred_field)
    supervised_loss = Loss(pred_field[..., :3], field_supervised_m[..., :3])

    eqs_loss = (res_i ** 2).mean()

    loss_batch = bcs_loss * 1. + eqs_loss + supervised_loss *10
    loss_batch.backward()

    # data_loss = Loss(out_var, out_true)
    log_loss.append([eqs_loss.item(), bcs_loss.item(), bcs_loss_top.item(), bcs_loss_bot.item(),
                     bcs_loss_wall.item(), bcs_loss_in.item(),
                     bcs_loss_out.item(), bcs_loss_initial.item(), supervised_loss.item()])
    optimizer.step()
    scheduler.step()

################################## 预测  ##################################
def inference(inn_var, model):
    inn_var = inn_var.cuda()
    inn_var.stop_gradient = False
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

    if paddle.fluid.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device('cpu')

    #################### 定义问题相关参数 ####################
    Rho, Miu, D = 1.0, 0.02, 2
    num_time = 50  #####随机抽取时间步个数
    # U_max, tmax = 0.5, 0.5  # 入口流速的最大值以及非定常周期 tmax = T/2
    # Box = [0, 0, 1.1, 0.41]  # 矩形流域

    #################### 读入数据 ####################
    data_ori = read_paddle_data(num_time)
    data = list(map(np.random.permutation, data_ori))  # np.random.shuffle & random.shuffle 返回None,此外， python 3 中map返回的是迭代器
    input = data[0]
    input = paddle.to_tensor(input[:, :3], dtype='float32', place='gpu:0')
    BCs = (data[1], data[2], data[3], data[4], data[5], data[-1])  ## 边界数据
    field = data[-2]  ##检测的流场点


    # 采用三角形 对非结构化网格建立节点连接关系
    triang = matplotlib.tri.Triangulation(data[-1][:, 0], data[-1][:, 1])
    triang.set_mask(np.hypot(data[-1][triang.triangles, 0].mean(axis=1),
                             data[-1][triang.triangles, 1].mean(axis=1)) < D / 2)
    # plt.figure(1, figsize=(20, 5))
    # t = plt.tricontourf(triang, data[-1][:, 3])
    # plt.axis('equal')
    # plt.show()

    #################### 定义损失函数、优化器以及网络结构 ####################
    L2Loss = nn.MSELoss()
    Net_model = Net(planes=[3] + 7 * [50] + [5], rho=Rho, miu=Miu)
    Boundary_epoch_1 = [100000, 200000]
    Scheduler_1 = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.0005, milestones=Boundary_epoch_1, gamma=0.1)
    Optimizer_1 = paddle.optimizer.Adam(parameters=Net_model.parameters(), learning_rate=Scheduler_1, beta1=0.9,
                                        beta2=0.999)
    Boundary_epoch_2 = [200000, 250000, 300000]
    Scheduler_2 = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.0005, milestones=Boundary_epoch_2, gamma=0.1)
    Optimizer_2 = paddle.optimizer.Adam(parameters=Net_model.parameters(), learning_rate=Scheduler_2, beta1=0.9,
                                        beta2=0.999)
    # Optimizer_2 = paddle.incubate.optimizer.functional.minimize_lbfgs(parameters=Net_model.parameters(), learning_rate=.1, max_iter=100)
    Visual = matplotlib_vision('/', field_name=('p', 'u', 'v'), input_name=('x', 'y'))

    ################################### 训练 #####################################
    star_time = time.time()
    log_loss = []

    """load a pre-trained model"""
    Net_model.loadmodel(work_path + '\\latest_model.pth')
    for epoch in range(Boundary_epoch_2[-1]):
        # 如果GPU内存不充足，可以分批次进行训练
        if epoch < 200000:
            iter = 3
            for i in range(iter):
                data_itr = list(map(lambda x: x[i * int(x.shape[0] / iter):(i + 1) * int(x.shape[0] / iter)], data))
                input = data_itr[0]
                input = paddle.to_tensor(input[:, :3], dtype='float32', place='gpu:0')
                BCs = (data_itr[1], data_itr[2], data_itr[3], data_itr[4], data_itr[5], data_itr[-1])  ## 边界数据
                field = data_itr[-2]  ##检测的流场点
                train_adam(input, BCs, field, Net_model, L2Loss, Optimizer_1, Scheduler_1, log_loss)
            learning_rate = Scheduler_1.get_lr()
        if epoch >= 200000:
            iter = 3
            for i in range(iter):
                data_itr = list(map(lambda x: x[i * int(x.shape[0] / iter):(i + 1) * int(x.shape[0] / iter)], data))
                input = data_itr[0]
                input = paddle.to_tensor(input[:, :3], dtype='float32', place='gpu:0')
                BCs = (data_itr[1], data_itr[2], data_itr[3], data_itr[4], data_itr[5], data_itr[-1])  ## 边界数据
                field = data_itr[4]  ##检测的流场点
                train_adam(input, BCs, field, Net_model, L2Loss, Optimizer_2, Scheduler_2, log_loss)
                # train_LBFGS(input, BCs, field, Net_model, L2Loss, Optimizer_2, Scheduler_2, log_loss)
            learning_rate = Optimizer_2.state_dict()['param_groups'][0]['lr']

        if epoch > 0 and epoch % 200 == 0:
            print('epoch: {:6d}, lr: {:.1e}, cost: {:.2e}, dat_loss: {:.2e}, eqs_loss: {:.2e}, bcs_loss: {:.2e}'.
                  format(epoch, learning_rate, time.time() - star_time,
                         log_loss[-1][-1], log_loss[-1][0], log_loss[-1][1], ))
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
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 2], 'top_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 3], 'bot_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 4], 'wall_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 5], 'in_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 6], 'out_loss')
            Visual.plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 7], 'ini_loss')
            plt.savefig(os.path.join(work_path, 'detail_loss.svg'))

            # 根据模型预测流场， 若有真实场，则与真实场对比
            input_visual_p = paddle.to_tensor(data[-1][..., :3], dtype='float32', place='gpu:0')  # 取初场的空间坐标
            input_visual_p[:, -1] = input_visual_p[:, -1]  # 时间取最大
            field_visual_p, _ = inference(input_visual_p, Net_model)
            field_visual_t = data[-1][..., 3:]
            field_visual_p = field_visual_p.cpu().numpy()[..., 0:3]
            # field_visual_t = field_visual_p

            plt.figure(3, figsize=(30, 8))
            plt.clf()
            Visual.plot_fields_tr(field_visual_t, field_visual_p, input_visual_p.detach().cpu().numpy(), triang)

            # plt.savefig(res_path + 'field_' + str(t) + '-' + str(epoch) + '.jpg')
            plt.savefig(os.path.join(work_path, 'global_' + str(epoch) + '.jpg'), dpi=200)
            plt.savefig(os.path.join(work_path, 'global_now.jpg'))

            for i in range(5):
                input_visual_p = paddle.to_tensor(data[-1][..., :3], dtype='float32', place='gpu:0')  # 取初场的空间坐标
                tim = (i+1)*10
                input_visual_p[:, -1] = input_visual_p[:, -1] -1 + tim# 时间
                field_visual_p, _ = inference(input_visual_p, Net_model)
                field_visual_p = field_visual_p.cpu().numpy()[..., 0:3]
                # field_visual_t = field_visual_p

                plt.figure(3, figsize=(30, 8))
                plt.clf()
                Visual.plot_fields_tr(field_visual_p, field_visual_p, input_visual_p.detach().cpu().numpy(), triang)

                plt.savefig(os.path.join(work_path, 'global_' + str(epoch) +'time'+str(tim)  + '.jpg'), dpi=200)

            paddle.save({'epoch': epoch, 'model': Net_model.state_dict(), },
                        os.path.join(work_path, 'latest_model.pth'))