import numpy as np
import paddle
import paddle.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def numpy_32(x):
    if isinstance(x, (list, tuple)):
        y = []
        for xx in x:
            y.append(xx.detach().cpu().numpy())
    else:
        y = x.detach().cpu().numpy()
    return y

def tensor_32(x):
    if isinstance(x, (list, tuple)):
        y = []
        for xx in x:
            y.append(paddle.to_tensor(xx, dtype='float32', place='gpu:0'))
    else:
        y = paddle.to_tensor(x, dtype='float32', place='gpu:0')
    return y

###################自动微分求梯度以及Jacobian矩阵######################
def gradients(y, x, order=1, create=True):
    if order == 1:
        return paddle.grad(y, x, create_graph=create, retain_graph=True)[0]
        # return paddle.stack([paddle.grad([y[..., i].sum()], [x], retain_graph=True, create_graph=True)[0]
        #                     for i in range(y.size(-1))], axis=-1).squeeze(-1)
    else:
        return paddle.stack([paddle.grad([y[:, i].sum()], [x], create_graph=True, retain_graph=True)[0]
                            for i in range(y.shape[1])], axis=-1)

###################多个单一输出的神经网络########################
class PaddleModel_multi(nn.Layer):
    def __init__(self, planes,  active=nn.Tanh()):
        super(PaddleModel_multi, self).__init__()
        self.planes = planes
        self.active = active

        self.layers = nn.LayerList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1],
                                       weight_attr=nn.initializer.XavierNormal()))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1))
            self.layers.append(nn.Sequential(*layer))
            # self.layers[-1].apply(initialize_weights)

    def forward(self,in_var):
        # in_var = self.x_norm.norm(in_var)     正则化
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        return paddle.concat(y, axis=-1)

    def loadmodel(self, File):
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取  ! 可能有问题
            start_epoch = checkpoint['epoch']
            print("load start epoch at epoch " + str(start_epoch))
            Log_loss = checkpoint['log_loss'].tolist()
            return Log_loss
        except:
            print("load model failed！ start a new model.")
            return []

#############################################################################
class PaddleModel_single(nn.Layer):
    def __init__(self, planes, active=nn.GELU()):
        """
        :param planes: list，[M,...,N],全连接神经网络的输入维度，每个隐含层维度，输出维度
        :param active: 激活函数
                       与multi，single采用1个全连接层,该全连接层输出维度为N
        """
        super(PaddleModel_single, self).__init__()
        self.planes = planes
        self.active = active

        self.layers = nn.LayerList()
        for i in range(len(self.planes)-2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))

        self.layers = nn.Sequential(*self.layers)
        # self.apply(initialize_weights)

    def forward(self, in_var):
        out_var = self.layers(in_var)
        return out_var

    def loadmodel(self, File):

        try:
            checkpoint = paddle.load(File)
            self.load_state_dict(checkpoint['model'])        # 从字典中依次读取
            start_epoch = len(checkpoint['log_loss'])
            print("load start epoch at epoch " + str(start_epoch))
        except:
            print("load model failed！ start a new model.")


    def equation(self, **kwargs):
        return 0


# def adaptive_weights(loss_list, model):
#     max_grad_list = []
#     avg_grad_list = []
#     for i in range(len(loss_list)-1):
#         avg_grad_list.append([])

#     for name, param in model.named_parameters():
#         if 'bias' not in name:
#             max_grad_list.append(gradients(loss_list[0], param).abs().max().detach())
#             for k, loss in enumerate(loss_list[1:]):
#                 avg_grad_list[k].append(gradients(loss, param).abs().mean().detach())

#     avg_grad = torch.tensor(avg_grad_list).mean()
#     max_grad = torch.tensor(max_grad_list).max()

#     return max_grad / avg_grad


# def causal_weights_loss(res):
#     tol = 100.
#     Nt = res.shape[0]
#     M_t = torch.triu(torch.ones((Nt, Nt), device=res.device), diagonal=1).T
#     L_t = torch.mean(res**2, dim=1)
#     W_t = torch.exp(-tol*(M_t @ L_t.detach()))
#     loss = torch.mean(W_t * L_t)
#     return loss, W_t

def initialize_weights(net):
    for m in net.layers():
        if isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.xavier_uniform_(m.weight, gain=1)
            m.bias.data.zero_()

if __name__ == '__main__':
    Net_model = PaddleModel_multi(planes=[2] + [64] * 4 + [4])
    x = paddle.ones([1000, 2])
    y = Net_model(x)
    print(y)
