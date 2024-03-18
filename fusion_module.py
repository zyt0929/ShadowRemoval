import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.utils import save_image
from einops import rearrange

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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res

class DoubleAtten(nn.Module):
    """
    A2-Nets: Double Attention Networks. NIPS 2018
    """
    def __init__(self,in_c,reduction=4):
        super().__init__()
        self.reduction = reduction
        self.in_c = in_c

        self.convA = nn.Conv2d(in_c,in_c//reduction,kernel_size=1)
        self.convB = nn.Conv2d(in_c,in_c//reduction,kernel_size=1)
        self.convV = nn.Conv2d(in_c,in_c//reduction,kernel_size=1)
        self.conv3 = nn.Conv2d(in_c//reduction, in_c, kernel_size=1)

        self.conv1 = nn.Conv2d(in_c*2, in_c, 1)
        self.conv2 = nn.Conv2d(in_c*2, in_c, 1)

    def forward(self,input,sdf):

        feature_maps = self.convA(input)
        atten_map = self.convB(self.conv1(torch.cat([input, sdf], dim=1)))
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c//self.reduction, h*w) # 对 A 进行reshape
        atten_map = atten_map.view(b, self.in_c//self.reduction, 1, h*w)       # 对 B 进行reshape 生成 attention_maps
        global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)),dim=-1) # 特征图与attention_maps 相乘生成全局特征描述子

        v = self.convV(sdf)
        atten_vectors = F.softmax(v.view(b, self.in_c//self.reduction, h*w), dim=-1) # 生成 attention_vectors
        out = torch.bmm(atten_vectors.permute(0,2,1), global_descriptors).permute(0,2,1).view(b, _, h, w) # 注意力向量左乘全局特征描述子
        out = self.conv3(out)
        return out

class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y)) # feature map
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))    # sdf

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Fusion1(nn.Module):
    def __init__(self, in_c,reduction=4):
        super().__init__()
        self.sa = SpatialAttention()

        self.fusion = Attention(in_c)
        self.conv = nn.Sequential(DWConv(2*in_c, in_c, 3), DWConv(in_c, in_c, 3), nn.Sigmoid())

    def forward(self, x, a):
        x1 = self.fusion(a,x)
        attention_map = self.conv(torch.cat([x1,a],dim=1))

        out = x1 * attention_map

        return out







