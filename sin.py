import torch.nn as nn
import torch.nn.functional as F
import torch


class Condition(nn.Module):
    ''' Compute the region style of non-shadow regions'''

    def __init__(self, in_nc=3, nf=128):
        super(Condition, self).__init__()
        stride = 1
        pad = 0
        self.conv1 = nn.Conv2d(in_nc, nf // 4, 1, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf // 4, nf // 2, 1, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf // 2, nf, 1, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        cond = torch.mean(out, dim=[2, 3], keepdim=False)

        return cond




class SINLayer(nn.Module):
    def __init__(self, dims_in=256):
        super(SINLayer, self).__init__()

        self.reduction = nn.Conv2d(128,16,1,bias=False)
        self.gamma_conv0 = nn.Conv2d(dims_in + 16, dims_in // 2, 1)
        self.gamma_conv1 = nn.Conv2d(dims_in // 2, dims_in, 1)
        self.gamma_conv2 = nn.Conv2d(dims_in, dims_in, 1)
        self.bate_conv0 = nn.Conv2d(dims_in + 16, dims_in // 2, 1)
        self.bate_conv1 = nn.Conv2d(dims_in // 2, dims_in, 1)
        self.bate_conv2 = nn.Conv2d(dims_in, dims_in, 1)

    def forward(self, x, sdf):
        sdf = self.reduction(sdf)
        cond_f = torch.cat([x,sdf], dim=1)
        gamma = self.gamma_conv2(self.gamma_conv1(F.leaky_relu(self.gamma_conv0(cond_f), 0.2, inplace=True)))
        beta = self.bate_conv2(self.bate_conv1(F.leaky_relu(self.bate_conv0(cond_f), 0.2, inplace=True)))

        return x * gamma + beta


class ResidualBlock_SIN(nn.Module):

    def __init__(self, in_features=256, cond_dim=128):
        super(ResidualBlock_SIN, self).__init__()
        self.conv0 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_features, in_features, 3))
        self.local_scale0 = nn.Sequential(
            nn.Linear(cond_dim, in_features // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
        )
        self.local_shift0 = nn.Sequential(
            nn.Linear(cond_dim, in_features // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
        )

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_features, in_features, 3))
        self.local_scale1 = nn.Sequential(
            nn.Linear(cond_dim, in_features // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
        )
        self.local_shift1 = nn.Sequential(
            nn.Linear(cond_dim, in_features // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
        )

        self.in_features = in_features
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.InstanceNorm2d(in_features)
        self.norm = nn.InstanceNorm2d(in_features)
        self.SIN0 = SINLayer(in_features)
        self.SIN1 = SINLayer(in_features)

    def forward(self, x, cond, sdf):

        local_scale_0 = self.local_scale0(cond)
        local_scale_1 = self.local_scale1(cond)
        local_shift_0 = self.local_shift0(cond)
        local_shift_1 = self.local_shift1(cond)

        out_1 = self.conv0(x)
        out_2 = self.norm(out_1)

        out_3 = out_2 * (local_scale_0.view(-1, self.in_features, 1, 1)) + local_shift_0.view(-1, self.in_features, 1, 1)
        out_4 = self.SIN0(out_3,sdf)

        out_5 = self.act(out_4)

        out_6 = self.conv1(out_5)
        out_7 = self.norm(out_6)
        out_8 = out_7 * (local_scale_1.view(-1, self.in_features, 1, 1)) + local_shift_1.view(-1, self.in_features, 1,1)
        out_9 = self.SIN1(out_8)
        out_9 += identity

        return out_9
