#!/usr/bin/env python
# @Time    : 7/8/2022 11:49
# @Author  : Lei Gao
# @Email    : leigao@umd.edu
import torch
import torch.nn as nn
from basic_model import DeepModel_single, gradients


class Net(DeepModel_single):
    def __init__(self, planes):
        super(Net, self).__init__(planes, active=nn.Tanh())
        self.Re = 100.

    def equation(self, inn_var, out_var):
        p = out_var[..., (0,)]
        u = out_var[..., (1,)]
        v = out_var[..., (2,)]
        dpda = gradients(p, inn_var)
        duda = gradients(u, inn_var)
        dvda = gradients(v, inn_var)

        dpdx, dpdy = dpda[..., (0,)], dpda[..., (1,)]
        dudx, dudy = duda[..., (0,)], duda[..., (1,)]
        dvdx, dvdy = dvda[..., (0,)], dvda[..., (1,)]

        d2udx2 = gradients(dudx, inn_var)[..., (0,)]
        d2udy2 = gradients(dudy, inn_var)[..., (1,)]
        d2vdx2 = gradients(dvdx, inn_var)[..., (0,)]
        d2vdy2 = gradients(dvdy, inn_var)[..., (1,)]

        res_x = u * dudx + v * dudy + dpdx - (d2udx2+d2udy2) / self.Re
        res_y = u * dvdx + v * dvdy + dpdy - (d2vdx2+d2vdy2) / self.Re
        res_c = dudx + dvdy

        return torch.cat((res_x, res_y, res_c), dim=-1), torch.stack((dpda, duda, dvda), dim=-1)


def train(inn_var, bounds, out_true, model, Loss, optimizer, scheduler, log_loss):

    def closure():

        optimizer.zero_grad()
        out_var = model(inn_var)
        res_i, Dout_var = model.equation(inn_var, out_var)
        bcs_loss_inn_u = Loss(out_var[0, :, 2:], out_true[0, :, 2:])
        bcs_loss_bot_u = Loss(out_var[:, 0, 1:], out_true[:, 0, 1:])
        bcs_loss_top_u = Loss(out_var[:, -1, 1:], out_true[:, -1, 1:])
        bound_out_u = (bounds[-1, :, :].unsqueeze(-1) * Dout_var[-1, :, :, 1:]).sum(dim=-2)
        bcs_loss_out_u = Loss(bound_out_u, torch.zeros_like(bound_out_u))
        bcs_loss_u = bcs_loss_bot_u + bcs_loss_top_u + bcs_loss_out_u + bcs_loss_inn_u

        bcs_loss_out_p = Loss(out_var[-1, :, 0], out_true[-1, :, 0])
        bound_inn_p = (bounds[0, :, :] * Dout_var[0, :, :, 0]).sum(dim=-1)
        bound_top_p = (bounds[:, -1, :] * Dout_var[:, -1, :, 0]).sum(dim=-1)
        bound_bot_p = (bounds[:, 0, :] * Dout_var[:, 0, :, 0]).sum(dim=-1)
        bcs_loss_inn_p = Loss(bound_inn_p, torch.zeros_like(bound_inn_p))
        bcs_loss_top_p = Loss(bound_top_p, torch.zeros_like(bound_top_p))
        bcs_loss_bot_p = Loss(bound_bot_p, torch.zeros_like(bound_bot_p))
        bcs_loss_p = bcs_loss_out_p + bcs_loss_inn_p + bcs_loss_top_p + bcs_loss_bot_p

        obj_loss = Loss(out_var[-1, :, 1:], out_true[-1, :, 1:])

        eqs_loss = Loss(res_i[1:-1, 1:-1], torch.zeros_like(res_i[1:-1, 1:-1], dtype=torch.float32))
        loss_batch = bcs_loss_u * 10 + bcs_loss_p * 10 + eqs_loss + obj_loss * 5
        loss_batch.backward()

        data_loss = Loss(out_var, out_true)
        log_loss.append([eqs_loss.item(), bcs_loss_u.item(), bcs_loss_p.item(), obj_loss.item(), data_loss.item()])

        return loss_batch

    optimizer.step(closure)
    scheduler.step()


def inference(inn_var, model):

    out_pred = model(inn_var)
    equation, _ = model.equation(inn_var, out_pred)
    return out_pred, equation