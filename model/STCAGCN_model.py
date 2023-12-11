# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:15:35 2023

@author: TongNie
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """
    def __init__(self, in_features, out_features, orders, activation='relu', adpt_type="pam"):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.num_matrices = 2 * self.orders + 1
        self.adpt_type = adpt_type
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_features * self.num_matrices,
                                                     out_features))
        self.bias1 = nn.Parameter(torch.FloatTensor(out_features))
        self.Theta2 = nn.Parameter(torch.FloatTensor(in_features * self.num_matrices,
                                                     out_features))
        self.bias2 = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        self.fusion = nn.Linear(3 * out_features, out_features)

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.bias1.shape[0])
        self.bias1.data.uniform_(-stdv2, stdv2)

        stdv3 = 1. / math.sqrt(self.Theta2.shape[1])
        self.Theta2.data.uniform_(-stdv3, stdv3)
        stdv4 = 1. / math.sqrt(self.bias2.shape[0])
        self.bias2.data.uniform_(-stdv4, stdv4)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h, adpt_adj=None):
        """
        :param X: Input data of shape (batch_size, in_features, num_timesteps, in_nodes)
        :return: Output data of shape (batch_size, out_features, num_timesteps, in_nodes)
        :A_q: The forward random walk matrix (batch_size, num_nodes, num_nodes)
        :A_h: The backward random walk matrix (batch_size, num_nodes, num_nodes)
        adpt_adj: (batch_size, num_nodes, num_nodes)
        """
        batch_size = X.shape[0]  # batch_size
        num_features = X.shape[1]  # in_features
        num_times = X.shape[2]  # time_length
        num_nodes = X.shape[3]  # in_nodes

        supports = []
        supports_add = []
        supports.append(A_q)
        supports.append(A_h)
        supports_add.append(adpt_adj)
        supports_add.append(adpt_adj)

        if adpt_adj is not None:
            x0 = X.permute(0, 3, 1, 2)  # (batch_size, in_nodes, in_features, num_timesteps)
            x0 = torch.reshape(x0, shape=[batch_size, num_nodes, num_features * num_times])
            x_add = torch.unsqueeze(x0, 0)
            for support in supports_add:
                x1 = torch.bmm(support, x0)  # (batch_size, num_nodes, num_features * num_times)
                x_add = self._concat(x_add, x1)  # (num_matrices, batch_size, num_nodes, num_features * num_times)
                for k in range(2, self.orders + 1):
                    x2 = torch.bmm(support, x1)
                    x_add = self._concat(x_add, x2)
                    x1 = x2  # (num_matrices, batch_size, num_nodes, num_features * num_times)
            x_add = torch.reshape(x_add, shape=[self.num_matrices, batch_size, num_nodes, num_features, num_times])
            x_add = x_add.permute(1, 4, 2, 3, 0)  # (batch_size, num_times, num_nodess, num_features,order)
            x_add = torch.reshape(x_add, shape=[batch_size, num_times, num_nodes, num_features * self.num_matrices])
            x_add1 = torch.einsum("btji,io->btjo", [x_add, self.Theta1])
            x_add1 += self.bias1  # (batch_size, num_timesteps, num_nodes, out_features)
            x_add2 = torch.einsum("btji,io->btjo", [x_add, self.Theta2])
            x_add2 += self.bias2  # (batch_size, num_timesteps, num_nodes, out_features)

        x0 = X.permute(0, 3, 1, 2)  # (batch_size, in_nodes, in_features, num_timesteps)
        x0 = torch.reshape(x0, shape=[batch_size, num_nodes, num_features * num_times])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.bmm(support, x0)  # (batch_size, num_nodes, num_features * num_times)
            x = self._concat(x, x1)  # (num_matrices, batch_size, num_nodes, num_features * num_times)
            for k in range(2, self.orders + 1):
                x2 = torch.bmm(support, x1)
                x = self._concat(x, x2)
                x1 = x2  # (num_matrices, batch_size, num_nodes, num_features * num_times)
        x = torch.reshape(x, shape=[self.num_matrices, batch_size, num_nodes, num_features, num_times])
        x = x.permute(1, 4, 2, 3, 0)  # (batch_size, num_times, num_nodes, num_features,order)
        x = torch.reshape(x, shape=[batch_size, num_times, num_nodes, num_features * self.num_matrices])
        x = torch.einsum("btji,io->btjo", [x, self.Theta1])
        x += self.bias1  # (batch_size, num_timesteps, num_nodes, out_features)

        if adpt_adj is not None:
            x_res = torch.concat([x, x_add1, x_add2], -1)  # (batch_size, num_timesteps, num_nodess, out_features*3)
            x_res = self.fusion(x_res)
        else:
            x_res = x

        if self.activation == 'relu':
            x_res = F.relu(x_res)
        elif self.activation == 'selu':
            x_res = F.selu(x_res)

        return x_res.permute(0, 3, 1, 2)  # (batch_size, out_features, num_timesteps, in_nodes)


class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class tcn_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="linear", dropout=0.1):
        super(tcn_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1, padding=(1, 0))
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1, padding=(1, 0))
        self.dropout = dropout

    def forward(self, x):
        """
        :param x: Input data of shape (batch_size, num_variables, num_timesteps, num_nodes)
        :return: Output data of shape (batch_size, num_features, num_timesteps - kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, :, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            h = (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            return F.dropout(h, self.dropout, training=self.training)
        if self.act == "sigmoid":
            h = torch.sigmoid(self.conv(x) + x_in)
            return F.dropout(h, self.dropout, training=self.training)
        h = self.conv(x) + x_in
        return F.dropout(h, self.dropout, training=self.training)


class PAM(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda'):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)
        self.linear2 = nn.Linear(in_features=out_channels, out_features=out_channels, bias=False)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Mult = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        nn.init.xavier_uniform_(self.Mult)

    def forward(self, speed_data, zeros=False, epsilon=0.1):
        """
        :param speed_data: tensor, [batch_size, time_step, num_nodes]. the speed series of each nodes in each batch
        NOTE that during training stage, the speed data in each batch is used to calculate PAM
        :return: learned pattern aware adjacency matrices.
        """
        speed_data = torch.transpose(speed_data, 1, 2)  # to (batch_size, num_nodes, time_step)
        speed_data = self.LeakyReLU(self.linear2(self.LeakyReLU(self.linear1(speed_data))))
        pams = torch.bmm(speed_data, torch.transpose(speed_data, 1, 2))
        denom = torch.bmm(torch.linalg.norm(speed_data, axis=2, keepdim=True),
                          torch.linalg.norm(speed_data, axis=2, keepdim=True).transpose(1, 2))
        pams = pams / denom

        if zeros == True:
            pams -= torch.diag_embed(torch.diagonal(pams, dim1=-2, dim2=-1))
        pams[pams <= epsilon] = 0

        return pams  # pams: (batch_size, num_nodes, num_nodes)


class STCAGCN(nn.Module):
    def __init__(self, time_len, order=1, in_variables=1, channels=32, t_kernel=2, adpt_type="pam"):
        super(STCAGCN, self).__init__()
        self.time_dimension = time_len
        self.hidden_dimnesion = channels
        self.order = order
        self.in_variables = in_variables


        self.GNN1 = D_GCN(self.in_variables, self.hidden_dimnesion, self.order)
        self.TCN = tcn_layer(t_kernel, self.hidden_dimnesion, self.hidden_dimnesion, dropout=0)
        self.GNN2 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)
        self.GNN3 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)
        self.GNN4 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)
        self.GNN5 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)

        self.num_units = 5
        self.linear = nn.Linear(self.hidden_dimnesion * self.num_units, self.in_variables)

        self.adpt_type = adpt_type
        if adpt_type == "pam":
            self.getpam = PAM(self.time_dimension, self.hidden_dimnesion)

    def forward(self, X, A_q, A_h, A_q0, A_h0, speed_data):
        """
        :param X: Input data of shape (batch_size, in_features, num_timesteps, in_nodes)
         speed_data: (batch_size, num_timesteps, num_nodes)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, in_features, num_timesteps, in_nodes)
        """
        batch_size = X.shape[0]
        num_nodes = X.shape[-1]
        A_q_0 = torch.stack((A_q0,) * batch_size, dim=0)
        A_h_0 = torch.stack((A_h0,) * batch_size, dim=0)
        A_q_af = torch.stack((A_q,) * batch_size, dim=0)
        A_h_af = torch.stack((A_h,) * batch_size, dim=0)
        if self.adpt_type == "pam":
            adpt_adj = self.getpam(speed_data, zeros=True)  # (batch_size, num_nodes, num_nodes)
            adpt_adj_af = self.getpam(speed_data, zeros=False)  # (batch_size, num_nodes, num_nodes)
            adpt_adj_nm = F.softmax(adpt_adj, dim=-1)
            adpt_adj_af_nm = F.softmax(adpt_adj_af, dim=-1)


        X_S = self.GNN1(X, A_q_0, A_h_0, adpt_adj_nm)  # (batch_size, num_features, num_timesteps, num_nodes)
        X_s1 = self.TCN(X_S)  # (batch_size, num_features, num_timesteps - kt+1, num_nodes)
        X_s2 = self.GNN2(X_s1, A_q_af, A_h_af, adpt_adj_af_nm) + X_s1
        X_s3 = self.GNN3(X_s2, A_q_af, A_h_af, adpt_adj_af_nm) + X_s2
        X_s4 = self.GNN3(X_s2, A_q_af, A_h_af, adpt_adj_af_nm) + X_s3
        X_s5 = self.GNN3(X_s2, A_q_af, A_h_af, adpt_adj_af_nm) + X_s4
        X_res = torch.stack([X_S, X_s2, X_s3, X_s4, X_s5],
                            dim=-1)  # (batch_size, out_features, num_timesteps - kt, in_nodes, num_units)

        X_res = X_res.permute(0, 2, 3, 1, 4)  # (batch_size,num_timesteps - kt+1, in_nodes, out_features，num_units)
        X_res = torch.reshape(X_res,
                              [batch_size, self.time_dimension, num_nodes, self.hidden_dimnesion * self.num_units])
        X_res = self.linear(X_res)  # (batch_size,num_timesteps - kt+1, in_nodes, in_features)
        X_res = X_res.permute(0, 3, 1, 2)  # (batch_size, in_features, num_timesteps - kt+1, num_nodes)


        return X_res, adpt_adj_af, adpt_adj_nm  # (batch_size, in_features, num_timesteps, num_nodes)