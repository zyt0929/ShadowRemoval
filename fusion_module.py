import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return x * out



class ResBlock_fft_bench(nn.Module):
    def __init__(self, in_channel, out_channel, norm='backward'): # 'ortho'
        super(ResBlock_fft_bench, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channel*2, out_channel*2, kernel_size=1, stride=1, bias=False)
        # m['relu1'] = nn.ReLU(True)
        # m['conv2'] = nn.Conv2d(in_channel*2, out_channel*2, kernel_size=1, stride=1, bias=False)
        self.main_fft = nn.Sequential(m)

        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return  y


class DoubleAtten(nn.Module):

    def __init__(self,in_c,reduction=4):
        super().__init__()
        self.reduction = reduction
        self.in_c = in_c

        self.convA = nn.Conv2d(in_c,in_c//reduction,kernel_size=1)
        self.convB = nn.Conv2d(in_c,in_c//reduction,kernel_size=1)
        self.convV = nn.Conv2d(in_c,in_c//reduction,kernel_size=1)
        self.conv3 = nn.Conv2d(in_c//reduction, in_c, kernel_size=1)


        self.conv2 = nn.Conv2d(in_c*2, in_c,1)
        self.conv1 = nn.Conv2d(in_c*2, in_c,1)

    def forward(self,input,x2):

        feature_maps = self.convA(input)
        atten_map = self.convB(self.conv1(torch.cat([input,x2],dim=1)))
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c//self.reduction, h*w) # 对 A 进行reshape
        atten_map = atten_map.view(b, self.in_c//self.reduction, 1, h*w)       # 对 B 进行reshape 生成 attention_maps
        # print(feature_maps.shape)    # (b,1,c,hw)
        # print(atten_map.shape)       # (b,c,1,hw)
        # print((feature_maps * F.softmax(atten_map, dim=-1)).shape) # (b,c,c,hw)
        global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)),dim=-1) # 特征图与attention_maps 相乘生成全局特征描述子

        v = self.convV(self.conv2(torch.cat([input,x2],dim=1)))
        atten_vectors = F.softmax(v.view(b, self.in_c//self.reduction, h*w), dim=-1) # 生成 attention_vectors
        # print((atten_vectors.permute(0,2,1)).shape) # (b,hw,c)
        # print(global_descriptors.shape) # (b,c,c)
        # print(torch.bmm(atten_vectors.permute(0, 2, 1), global_descriptors).shape) # (b,hw,c)
        out = torch.bmm(atten_vectors.permute(0,2,1), global_descriptors).permute(0,2,1).view(b, _, h, w)# 注意力向量左乘全局特征描述子
        out = self.conv3(out)
        return out



class Fusion1(nn.Module):
    def __init__(self, in_c,reduction=4):
        super().__init__()
        self.ffc = ResBlock_fft_bench(in_c, in_c)
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
        )
        self.sa = SpatialAttention()
        self.fusion = DoubleAtten(in_c, reduction=reduction)
        self.conv1x1 = nn.Conv2d(in_c * 2 , in_c, 1)

    def forward(self, x1, a):

        x_g = self.sa(x1[1])
        x_l = self.sa(x1[0])
        x_l_fusion = self.fusion(x_l, a)
        x_g_fusion = self.fusion(x_g, a)
        x = self.conv1x1(torch.cat([x_g_fusion, x_l_fusion], dim=1))

        return x

if __name__=="__main__":
    a = torch.randn(size=(4,32,16,16)).to("cuda")
    b = torch.randn(size=(4,32,16,16)).to("cuda")
    model = DoubleAtten(32).to("cuda")
    a = model(a,b).to("cuda")
    # print(a.shape)








