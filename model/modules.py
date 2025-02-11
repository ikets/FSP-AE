import math
import numpy as np
import torch as th
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class Net(th.nn.Module):
    def __init__(self, model_name="network", use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.model_name = model_name

    def save(self, model_dir, suffix=""):
        '''
        save the network to model_dir/model_name.suffix.net
        :param model_dir: directory to save the model to
        :param suffix: suffix to append after model name
        '''
        if self.use_cuda:
            self.cpu()

        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"

        th.save(self.state_dict(), fname)
        if self.use_cuda:
            self.cuda()

    def load_from_file(self, model_file):
        '''
        load network parameters from model_file
        :param model_file: file containing the model parameters
        '''
        if self.use_cuda:
            self.cpu()

        states = th.load(model_file)
        self.load_state_dict(states)

        if self.use_cuda:
            self.cuda()
        print(f"Loaded: {model_file}")

    def load(self, model_dir, suffix=""):
        '''
        load network parameters from model_dir/model_name.suffix.net
        :param model_dir: directory to load the model from
        :param suffix: suffix to append after model name
        '''
        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"
        self.load_from_file(fname)

    def num_trainable_parameters(self):
        '''
        :return: the number of trainable parameters in the model
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResLinearBlock(nn.Module):
    def __init__(self, channel=64, droprate=0.0, norm=None, bn_c=None):
        super().__init__()
        if norm == 'ln':
            self.layers1 = nn.Sequential(
                nn.Linear(channel, channel),
                nn.LayerNorm(channel),
                nn.Mish(),
                nn.Dropout(droprate),
                nn.Linear(channel, channel)
            )
            self.layers2 = nn.Sequential(
                nn.LayerNorm(channel),
                nn.Mish(),
                nn.Dropout(droprate)
            )
        elif norm == 'bn':
            self.layers1 = nn.Sequential(
                nn.Linear(channel, channel),
                nn.BatchNorm1d(bn_c),
                nn.Mish(),
                nn.Dropout(droprate),
                nn.Linear(channel, channel)
            )
            self.layers2 = nn.Sequential(
                nn.BatchNorm1d(bn_c),
                nn.Mish(),
                nn.Dropout(droprate)
            )
        elif norm is None:
            self.layers1 = nn.Sequential(
                nn.Linear(channel, channel),
                nn.Mish(),
                nn.Dropout(droprate),
                nn.Linear(channel, channel)
            )
            self.layers2 = nn.Sequential(
                nn.Mish(),
                nn.Dropout(droprate)
            )

    def forward(self, input):
        out_mid = self.layers1(input)
        in_mid = out_mid + input  # skip connection
        out = self.layers2(in_mid)
        return out


class HyperLinear(nn.Module):
    def __init__(self, ch_in, ch_out, input_size=3, ch_hidden=32, num_hidden=1, use_res=True, droprate=0.0):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out

        # ==== weight ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.Mish(),
            nn.Dropout(droprate)]
        if not use_res:
            for _ in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.Mish(),
                    nn.Dropout(droprate)
                ])
        else:
            for _ in range(round(num_hidden / 2)):
                modules.extend([ResLinearBlock(ch_hidden, droprate=0.0)])
        modules.extend([
            nn.Linear(ch_hidden, ch_out * ch_in)
        ])
        self.weight_layers = nn.Sequential(*modules)

        # ==== bias ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.Mish(),
            nn.Dropout(droprate)]
        if not use_res:
            for _ in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.Mish(),
                    nn.Dropout(droprate)
                ])
        else:
            for _ in range(round(num_hidden / 2)):
                modules.extend([ResLinearBlock(ch_hidden, droprate=0.0)])
        modules.extend([
            nn.Linear(ch_hidden, ch_out)
        ])
        self.bias_layers = nn.Sequential(*modules)

    def forward(self, input):
        x, z = input
        batches = list(x.shape)[:-1]  # (...,)
        num_batches = math.prod(batches)

        weight = self.weight_layers(z)  # (..., ch_out * ch_in)
        weight = weight.reshape([num_batches, self.ch_out, self.ch_in])  # (num_batches, ch_out, ch_in)
        bias = self.bias_layers(z)  # (..., ch_out)

        # output = {}
        wx = th.matmul(weight, x.reshape(num_batches, -1, 1)).reshape(batches + [-1])
        # output["x"] = wx + bias  # F.linear(x, weight, bias)
        # output["z"] = z

        return (wx + bias, z)


class HyperLinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32, num_hidden=1, droprate=0.0, use_res=True, cond_dim=3, post_prcs=True):
        super().__init__()
        self.hyperlinear = HyperLinear(in_dim, out_dim, ch_hidden=hidden_dim, num_hidden=num_hidden, use_res=use_res, input_size=cond_dim, droprate=droprate)
        if post_prcs:
            self.layers_post = nn.Sequential(
                nn.LayerNorm(out_dim),
                nn.Mish(),
                nn.Dropout(droprate)
            )
        else:
            self.layers_post = nn.Sequential(
                nn.Identity()
            )

    def forward(self, input):
        # x, z = input
        # x = input["x"] # (B, ch_in)
        # z = input["z"]  # (B, input_size=3)
        y, z = self.hyperlinear(input)  # x,z]
        y = self.layers_post(y)

        return (y, z)


class FourierFeatureMapping(nn.Module):
    def __init__(self, num_features, input_dim, trainable=True):
        super(FourierFeatureMapping, self).__init__()
        self.num_features = num_features
        self.input_dim = input_dim
        multinormdist = MultivariateNormal(th.zeros(self.input_dim), th.eye(self.input_dim))
        self.v = multinormdist.sample(sample_shape=th.Size([self.num_features]))  # self.num_features, dim_data
        if trainable:
            self.v = nn.Parameter(self.v)
        else:
            self.v = self.v.cuda()

    def forward(self, x):
        # x: (S,B,D)
        x_shape = list(x.shape)
        x_shape[-1] = self.num_features
        x = x.reshape(-1, self.input_dim).permute(1, 0)  # D, SB
        vx = th.matmul(self.v.to(x.device), x)  # J, SB
        vx = vx.permute(1, 0).reshape(x_shape)  # J, SB -> SB, J -> S, B, J
        fourierfeatures = th.cat((th.sin(2 * np.pi * vx), th.cos(2 * np.pi * vx)), dim=-1)  # S, B, 2J

        return fourierfeatures
