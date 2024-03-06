import torch.nn.functional as F
import torch.nn as nn
import torch
from training.modules.fusion_module import Fusion1, DoubleAtten
from collections import OrderedDict
from training.modules.sin import *
from einops import rearrange


class ResBlock_G(nn.Module):
    def __init__(self, in_channel, out_channel, norm='backward'):  # 'ortho'
        super().__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channel * 2, out_channel * 2, kernel_size=1, stride=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(in_channel*2, out_channel*2, kernel_size=1, stride=1, bias=False)
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
        return y

class ResBlock_L(nn.Module):
    def __init__(self, in_channel, out_channel):  # 'ortho'
        super().__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        m['relu2'] = nn.ReLU(True)
        self.conv = nn.Sequential(m)

    def forward(self, x):
        out = self.conv(x) + x

        return out

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

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=True, use_se=True, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            self.se = SELayer(self.conv_layer.in_channels)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()  # 没有对特征图进行改变

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)

        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)

        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        output = self.conv2(x + output + xs)
        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4,
                 stride=2, padding=1,before=None, after=None):
        super(FFC, self).__init__()

        self.w = nn.Parameter(torch.tensor([0.0], requires_grad=True)) # 设置一个可以学习的参数
        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid


        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.conv_l2l = nn.Conv2d(in_channels, out_channels, kernel_size,stride, padding, bias=False, padding_mode="reflect")
        # self.convg2l = nn.Conv2d(in_channels, out_channels, kernel_size,stride, padding, bias=False, padding_mode="reflect")
        self.convl2g = nn.Conv2d(in_channels, out_channels, kernel_size,stride, padding, bias=False, padding_mode="reflect")
        self.conv_g2g = SpectralTransform(in_channels, out_channels, stride, groups=1, enable_lfu=True)
    def forward(self, x):
        if type(x) is tuple:
            x_l, x_g = x
        else:
            x_l = x
            x_g = x

        if hasattr(self, 'before'):
            x_l = self.before(x_l)
            x_g = self.before(x_g)

        x_l = self.conv_l2l(x_l)
        x_g = self.conv_g2g(x_g) + self.convl2g(x_g)

        if hasattr(self, 'after'):
            x_l = self.after(x_l)

        out = (x_l, x_g)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_l = ResBlock_L(channels, channels)
        self.conv_g = ResBlock_G(channels, channels)

    def forward(self, x):
        x_l, x_g = x
        outx_l = self.conv_l(x_l)
        outx_g = self.conv_g(x_g)
        out = x_l + outx_l + outx_g

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

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

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

class FuseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv_l = ResBlock_L(channels, channels)
        self.conv_g = ResBlock_G(channels, channels)
        self.fre_att = Attention(dim=channels)
        self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(DWConv(2*channels, channels, 3), DWConv(channels, 2*channels, 3), nn.Sigmoid())

    def forward(self, x):

        fre = self.conv_l(x[0])
        spa = self.conv_g(x[1])
        fre = self.fre_att(fre, spa)+fre
        spa = self.fre_att(spa, fre)+spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa
        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('gaussian'))

        if after == 'BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x

class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.conv.apply(weights_init('gaussian'))

        if after == 'BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif before == 'Tanh':
            self.before = nn.Tanh()

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x





class Generator(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super(Generator, self).__init__()

        self.Cv0 = FFC(input_channels, 64)
        self.Cv1 = FFC(64, 128, before='ReLU')
        self.Cv2 = FFC(128, 256, before='ReLU')
        self.Cv3 = FFC(256, 256, before='ReLU')

        self.res1 = FuseBlock(256)
        self.res2 = FuseBlock(128)
        self.res3 = FuseBlock(64)

        # self.block_sin = ResidualBlock_SIN(256)

        self.fusion_f1 = Fusion1(256)
        self.fusion_f2 = Fusion1(128)
        self.fusion_f3 = Fusion1(64)


        self.CvT4 = CvTi(256, 256, before='ReLU',after='BN')
        self.CvT5 = CvTi(512, 128, before='ReLU')
        self.CvT6 = CvTi(256, 64, before='ReLU')
        self.CvT7 = CvTi(128, output_channels, before='ReLU', after='Tanh')

        self.Av0 = Cvi(3, 64)
        self.Av1 = Cvi(64, 128)
        self.Av2 = Cvi(128, 256)
        self.Av3 = Cvi(256, 128)


        self.condition = Condition()
        self.modulation = SINLayer()
    def forward(self, shadow, sdf, inv_shadow):
        a0 = self.Av0(torch.cat([sdf, sdf, sdf], dim=1))
        a1 = self.Av1(a0)
        a2 = self.Av2(a1)
        a3 = self.Av3(a2)

        input = torch.cat([shadow, sdf], dim=1)
        x0 = self.Cv0(input)  # (64,128,128)
        x1 = self.Cv1(x0)     # (128,64,64)
        x2 = self.Cv2(x1)     # (256,32,32)
        x3 = self.Cv3(x2)     # (256,16,16)
        x4 = self.res1(x3)    #


        x0_f = self.fusion_f3(x0, a0)
        x1_f = self.fusion_f2(x1, a1)
        x2_f = self.fusion_f1(x2, a2)


        cond = self.condition(inv_shadow)
        x4 = self.modulation(x4,a3)
        x4 = self.modulation(x4,a3)
        x4 = self.modulation(x4,a3)


        x4 = self.CvT4(x4)               # (256,32,32)
        x4 = x4 + x2_f
        x5 = self.CvT5(torch.cat([x4, self.res1(x2)], dim=1))  # (128,64,64)
        x5 = x5 + x1_f
        x6 = self.CvT6(torch.cat([x5, self.res2(x1)], dim=1))  # (64,128,128)
        x6 = x6 + x0_f
        x7 = self.CvT7(torch.cat([x6, self.res3(x0)], dim=1))  #

        return x7


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        self.Cv0 = Cvi(input_channels, 64)

        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')

        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')

        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')

        self.Cv4 = Cvi(512, 1, before='LReLU', after='sigmoid')


    def forward(self, input):
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        out = self.Cv4(x3)

        return out



if __name__ == '__main__':
    from torchsummary import summary
    G = Generator().to('cuda')
    input = torch.randn(1, 3, 256, 128).to('cuda')
    sdf = torch.randn(1, 1, 128, 128).to('cuda')
    m = torch.randn(1, 1, 128, 128).to('cuda')
    summary(G, [(3, 256, 128), (1, 128, 128), (1, 128, 128)], device='cuda')

