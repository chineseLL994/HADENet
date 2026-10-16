import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.NEW.swin_transformer import SwinTransformer
from model.NEW.smt import smt_t,smt_b

from model.aspp import ASPP
from model.NEW.MobileNetV2 import mobilenet_v2
from timm.models.layers import trunc_normal_
import math
from model.NEW.SBA import SBA
from model.NEW.MFM import MFM
from torch.nn import init

# from model.NEW.DMF import DynamicModalityFusion

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class out_block(nn.Module):
    def __init__(self, infilter):
        super(out_block, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(infilter, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)])
        self.conv2 = nn.Conv2d(64, 1, 1)

    def forward(self, x, H, W):
        x = F.interpolate(self.conv1(x), (H, W), mode='bilinear', align_corners=True)
        return self.conv2(x)

# U-Net
class decoder_stage(nn.Module):
    def __init__(self, infilter, midfilter, outfilter):
        super(decoder_stage, self).__init__()
        self.layer = nn.Sequential(
            *[nn.Conv2d(infilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              nn.Conv2d(midfilter, midfilter, 3, padding=1, bias=False), nn.BatchNorm2d(midfilter),
              nn.ReLU(inplace=True),
              nn.Conv2d(midfilter, outfilter, 3, padding=1, bias=False), nn.BatchNorm2d(outfilter),
              nn.ReLU(inplace=True)])

    def forward(self, x):
        return self.layer(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class UIM(nn.Module):
    def __init__(self, in_planes):
        super(UIM, self).__init__()
        self.catconv1 = BasicConv2d(in_planes=in_planes * 2, out_planes=in_planes, kernel_size=3,
                                    padding=1, stride=1)
        self.catconv2 = BasicConv2d(in_planes=in_planes * 4, out_planes=in_planes, kernel_size=3,
                                    padding=1, stride=1)

    def forward(self, input1, input2):
        cat_out = self.catconv1(torch.cat([input1, input2], dim=1))
        mul_out = input1 * input2
        sub_out = input1 - input2
        max_put = torch.maximum(input1, input2)
        interactive_out = self.catconv2(torch.cat([cat_out, mul_out, sub_out, max_put], dim=1))

        return interactive_out

def convblock(in_, out_, ks, st=1, pad=0,dila=1):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad, dila, bias = False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class CASA(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(CASA, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        x = x.mul(self.sigmoid(out))

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



class SFSSM(nn.Module):
    def __init__(self, in_dim, heads=4):
        super(SFSSM, self).__init__()
        self.in_dim = in_dim
        self.heads = heads
        self.scale = (in_dim // heads) ** -0.5

        # 线性投影
        self.q_proj = nn.Linear(in_dim, in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)

        # LN
        self.ln_rgb = nn.LayerNorm(in_dim)
        self.ln_depth = nn.LayerNorm(in_dim)

        # 使用MAM模块代替融合
        self.mam = MFM(in_dim, in_dim)   # 输入两个特征，输出保持 in_dim

    def forward(self, f_r, f_d):
        B, C, H, W = f_r.shape

        # ---- 1) 共享注意力提取 ----
        fr = f_r.permute(0, 2, 3, 1).reshape(B, H*W, C)
        fd = f_d.permute(0, 2, 3, 1).reshape(B, H*W, C)

        Q = self.q_proj(self.ln_rgb(fr))
        K = self.k_proj(self.ln_depth(fd))
        V_r = self.v_proj(fr)
        V_d = self.v_proj(fd)

        Ms = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)

        # 共享特征加权
        fr_s = torch.softmax(Ms, dim=-1) @ V_r + fr
        fd_s = torch.softmax(Ms, dim=-1) @ V_d + fd

        # reshape回原始形状
        fr_s = fr_s.reshape(B, H, W, C).permute(0, 3, 1, 2)
        fd_s = fd_s.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # ---- 2) 用MAM融合共享特征 ----
        out = self.mam(fr_s, fd_s)  # 输出通道数 = in_dim
        # out = fr_s+fd_s
        return out

class sSE(nn.Module):  # 空间(Space)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)  # 定义一个卷积层，用于将输入通道转换为单通道
        self.norm = nn.Sigmoid()  # 应用Sigmoid激活函数进行归一化

    def forward(self, x):
        q = self.conv(x)  # 使用卷积层减少通道数至1：b c h w -> b 1 h w
        q = self.norm(q)  # 对卷积后的结果应用Sigmoid激活函数：b 1 h w
        return x * q  # 通过广播机制将注意力权重应用到每个通道上

class cSE(nn.Module):  # 通道(channel)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 使用自适应平均池化，输出大小为1x1
        self.relu = nn.ReLU()  # ReLU激活函数
        self.Conv_Squeeze = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)  # 通道压缩卷积层
        self.norm = nn.Sigmoid()  # Sigmoid激活函数进行归一化
        self.Conv_Excitation = nn.Conv2d(in_ch // 2, in_ch, kernel_size=1, bias=False)  # 通道激励卷积层

    def forward(self, x):
        z = self.avgpool(x)  # 对输入特征进行全局平均池化：b c 1 1
        z = self.Conv_Squeeze(z)  # 通过通道压缩卷积减少通道数：b c//2 1 1
        z = self.relu(z)  # 应用ReLU激活函数
        z = self.Conv_Excitation(z)  # 通过通道激励卷积恢复通道数：b c 1 1
        z = self.norm(z)  # 对激励结果应用Sigmoid激活函数进行归一化
        return x * z.expand_as(x)  # 将归一化权重乘以原始特征，使用expand_as扩展维度与原始特征相匹配

class CIM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CIM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class Trans(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.apply(self._init_weights)

    def forward(self, d):
        return self.trans(d)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



# 特征融合模块
class LowFeatureFusion(nn.Module):
    def __init__(self, in_channels_low4=64, in_channels_low3=128, out_channels=64):
        super(LowFeatureFusion, self).__init__()

        # 调整 low_3 通道数，使其与 low_4 对齐
        self.reduce_low3_channels = nn.Sequential(
            nn.Conv2d(in_channels_low3, in_channels_low4, kernel_size=1),
            nn.BatchNorm2d(in_channels_low4),
            nn.ReLU(inplace=True)
        )

        # CBAM 注意力模块
        self.cbam_low4 = sSE(in_channels_low4)
        self.cbam_low3 = cSE(in_channels_low4)

        # 融合后的降维
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels_low4 * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_4, low_3):
        # 上采样 low_3 到与 low_4 相同的尺寸
        low_3 = F.interpolate(low_3, size=low_4.shape[2:], mode='bilinear', align_corners=False)

        # 通道对齐
        low_3 = self.reduce_low3_channels(low_3)

        # 注意力增强
        low_4 = self.cbam_low4(low_4)
        low_3 = self.cbam_low3(low_3)

        # 融合
        fused = torch.cat([low_4, low_3], dim=1)
        fused = self.fusion_conv(fused)

        return fused

class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm = nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)
class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class BIEF(nn.Module):
    def __init__(self, nf=64):
        super(BIEF, self).__init__()
        # self-process
        self.conv1 = ResidualBlock_noBN(nf)
        self.conv2 = self.conv1
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1)
        self.convf2 = self.convf1
        self.convf3 = nn.Conv2d(nf * 3, nf, 1, 1)
        self.scale = nf ** -0.5
        self.norm_s = LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)

        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)

        # self.uim = UIM(nf)
        self.catconv2 = BasicConv2d(in_planes=nf * 2, out_planes=nf, kernel_size=3,
                                            padding=1, stride=1)

        # initialization
        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_1, x_2):
        # x_1, x_2 =x
        b, c, h, w = x_1.shape
        x_s = x_1+x_2

        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]

        v_1 = self.v1(x_1).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
        v_2 = self.v2(x_2).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]

        att1 = torch.bmm(shared_class_center1, v_1) * self.scale # [b, c, hw] x [b, hw, c] -> [b, c, c]#K
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale  # [b, c, hw] x [b, hw, c] -> [b, c, c]

        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]

        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w), shared_class_center2.view(b, c, h, w)], dim=1))

        out = torch.cat([out_1+x_1,out_2+x_2,x_s_+x_s],dim=1)
        out = self.convf3(out)
        # out = self.attention(out)

        # out = out_1*x_1_ + out_2*x_2_ + x_s_*x_s
        return out
        # return saout

class Model(nn.Module):
    def __init__(self, spatial_ckpt=None, flow_ckpt=None, depth_ckpt=None):
        # def __init__(self, inchannels, mode, model_path):
        super(Model, self).__init__()
        # self.spatial_net = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        # self.flow_net = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        # self.depth_net = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        self.spatial_net = smt_b(True)
        self.flow_net = smt_b(True)
        self.depth_net = smt_b(True)

        self.low_feature = mobilenet_v2(pretrained=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        if spatial_ckpt is not None:
            # self.spatial_net.load_state_dict(torch.load(spatial_ckpt, map_location='cpu'))
            # print("Successfully load spatial:{}".format(spatial_ckpt))
            self.spatial_net.load_state_dict(torch.load('model/NEW/smt_tiny.pth', weights_only=False)['model'], strict=False)
            print('loading pretrained model success!')
        if flow_ckpt is not None:
            # self.flow_net.load_state_dict(torch.load(flow_ckpt, map_location='cpu'))
            # print("Successfully load flow:{}".format(flow_ckpt))
            self.flow_net.load_state_dict(torch.load('model/NEW/smt_tiny.pth', weights_only=False)['model'], strict=False)
            print('loading pretrained model success!')
        if depth_ckpt is not None:
            # self.depth_net.load_state_dict(torch.load(depth_ckpt, map_location='cpu'))
            # print("Successfully load depth:{}".format(depth_ckpt))
            self.depth_net.load_state_dict(torch.load('model/NEW/smt_tiny.pth', weights_only=False)['model'], strict=False)
            print('loading pretrained model success!')

        self.aspp4 = ASPP(512, 512, [1, 6, 12, 18])
        self.aspp3 = ASPP(256, 256, [1, 6, 12, 18])
        self.aspp2 = ASPP(128, 128, [1, 6, 12, 18])
        self.aspp1 = ASPP(64, 64, [1, 6, 12, 18])

        self.mfm1 = SFSSM(64, 4)
        self.mfm2 = SFSSM(128, 4)
        self.mfm3 = SFSSM(256, 4)
        self.mfm4 = SFSSM(512, 4)

        self.bief1 = BIEF(64)
        self.bief2 = BIEF(128)
        self.bief3 = BIEF(256)
        self.bief4 = BIEF(512)

        self.catconv4 = BasicConv2d(2 * 512, 512, kernel_size=3, stride=1, padding=1)
        self.catconv3 = BasicConv2d(2 * 256, 256, kernel_size=3, stride=1, padding=1)
        self.catconv2 = BasicConv2d(2 * 128, 128, kernel_size=3, stride=1, padding=1)
        self.catconv1 = BasicConv2d(2 * 64, 64, kernel_size=3, stride=1, padding=1)


        self.meem1 = MEEM(64, 32)
        self.meem2 = MEEM(128, 32)

        self.edge1 = EdgeEnhancer(64)
        self.edge2 = EdgeEnhancer(128)

        self.cim4 = CIM(64, 64)
        self.cim3 = CIM(64, 64)
        self.cim2 = CIM(64, 64)
        self.cim1 = CIM(64, 64)

        self.low_trans_4 = Trans(320, 512)
        self.low_trans_3 = Trans(96, 256)
        self.low_trans_2 = Trans(32, 128)
        self.low_trans_1 = Trans(24, 64)

        self.low_feature_fusion2 = LowFeatureFusion(in_channels_low4=64, in_channels_low3=128)
        self.low_feature_fusion1 = LowFeatureFusion(in_channels_low4=64, in_channels_low3=64)
        self.low_feature_fusion = LowFeatureFusion(in_channels_low4=64, in_channels_low3=64)


        # self.mccm = MCCM(64)
        #
        # Components of PTM module
        self.inplanes = 32
        self.deconv1 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.inplanes = 16
        self.deconv2 = self._make_transpose(TransBasicBlock, 16, 3, stride=2)
        self.agant1 = self._make_agant_layer(64, 32)
        self.agant2 = self._make_agant_layer(32, 16)
        self.outconv2 = nn.Conv2d(16 * 1, 1, kernel_size=1, stride=1, bias=True)


        # ------------decoder----------------#
        self.decoder4 = decoder_stage(64, 128, 64)  #
        self.decoder3 = decoder_stage(128, 128, 64)  #
        self.decoder2 = decoder_stage(128, 128, 64)  #
        self.decoder1 = decoder_stage(128, 128, 64)  #

        self.out4 = out_block(64)
        self.out3 = out_block(64)
        self.out2 = out_block(64)
        self.out1 = out_block(64)

        self.channel64to1 = nn.Sequential(nn.Conv2d(64, 1, 1), nn.BatchNorm2d(1), nn.ReLU(inplace=True))
        self.channel128to64 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.channel256to64 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.channel512to64 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.channel1920to1 = nn.Sequential(nn.Conv2d(1920, 1, 1), nn.BatchNorm2d(1), nn.ReLU(inplace=True))


    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, image, flow, depth):
        # score_list_flow, score_PE = self.flow_net(flow)
        # score_list_rgb, score_PE = self.spatial_net(image)
        # score_list_depth, score_PE = self.spatial_net(depth)

        _, (x1_rgb, x2_rgb, x3_rgb, x4_rgb) = self.spatial_net(image)
        _, (x1_depth, x2_depth, x3_depth, x4_depth) = self.depth_net(depth)
        _, (x1_flow, x2_flow, x3_flow, x4_flow) = self.flow_net(flow)

        x1_rgb_aspp = self.aspp1(x1_rgb)
        x2_rgb = self.aspp2(x2_rgb)
        x3_rgb = self.aspp3(x3_rgb)
        x4_rgb = self.aspp4(x4_rgb)

        rd1 = self.mfm1(x1_rgb_aspp, x1_depth)
        rd2 = self.mfm2(x2_rgb, x2_depth)
        rd3 = self.mfm3(x3_rgb, x3_depth)
        rd4 = self.mfm4(x4_rgb, x4_depth)

        rf1 = self.mfm1(x1_rgb_aspp, x1_flow)
        rf2 = self.mfm2(x2_rgb, x2_flow)
        rf3 = self.mfm3(x3_rgb, x3_flow)
        rf4 = self.mfm4(x4_rgb, x4_flow)

        # fusion1 = self.mfm1(x1_rgb_aspp, x1_depth)
        # fusion2 = self.mfm2(x2_rgb, x2_depth)
        # fusion3 = self.mfm3(x3_rgb, x3_depth)
        # fusion4 = self.mfm4(x4_rgb, x4_depth)

        # fusion2 = self.channel128to64(fusion2)
        # fusion3 = self.channel256to64(fusion3)
        # fusion4 = self.channel512to64(fusion4)

        fusion1 = self.bief1(rd1, rf1)
        fusion2 = self.bief2(rd2, rf2)
        fusion3 = self.bief3(rd3, rf3)
        fusion4 = self.bief4(rd4, rf4)

        # fusion1 = self.scsa1(rd1, rf1)
        # fusion2 = self.scsa2(rd2, rf2)
        # fusion3 = self.scsa3(rd3, rf3)
        # fusion4 = self.scsa4(rd4, rf4)

        fusion2 = self.channel128to64(fusion2)
        fusion3 = self.channel256to64(fusion3)
        fusion4 = self.channel512to64(fusion4)

        # fusion1 = self.rfm1(x1_rgb_aspp, rd1, rf1)
        # fusion2 = self.rfm2(x2_rgb, rd2, rf2)
        # fusion3 = self.rfm3(x3_rgb, rd3, rf3)
        # fusion4 = self.rfm4(x4_rgb, rd4, rf4)

        feature4 = self.decoder4(fusion4)
        feature4 = self.cim4(feature4)
        B, C, H, W = fusion3.size()

        feature3 = self.decoder3(
            torch.cat((F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True), fusion3), 1))
        feature3 = self.cim3(feature3)
        B, C, H, W = fusion2.size()

        feature2 = self.decoder2(
            torch.cat((F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True), fusion2), 1))
        feature2 = self.cim2(feature2)
        B, C, H, W = fusion1.size()

        feature1 = self.decoder1(
            torch.cat((F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True), fusion1), 1))
        feature1 = self.cim1(feature1)

        decoder_out4 = self.out4(feature4, H * 4, W * 4)
        decoder_out3 = self.out3(feature3, H * 4, W * 4)
        decoder_out2 = self.out2(feature2, H * 4, W * 4)

        # PTM module
        y = self.agant1(feature1)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        decoder_out1 = self.outconv2(y)

        _, low_1, low_2, low_3, low_4 = self.low_feature(image)

        low_1 = self.low_trans_1(low_1)
        low_2 = self.low_trans_2(low_2)
        low_1 = self.meem1(low_1)
        low_2 = self.meem2(low_2)
        # low_1 = self.edge1(low_1)
        # low_2 = self.edge2(low_2)

        low_fusion = self.low_feature_fusion(x1_rgb_aspp, feature1)
        low_fusion128 = self.low_feature_fusion2(low_fusion, low_2)
        low_fusion64 = self.low_feature_fusion1(low_1, low_fusion128)


        # _, low_d1, low_d2, low_d3, low_d4 = self.low_feature(depth)
        #
        # low_d1 = self.low_trans_1(low_d1)
        # low_d2 = self.low_trans_2(low_d2)
        # low_d1 = self.meem1(low_d1)
        # low_d2 = self.meem2(low_d2)
        #
        # low_fusion_d = self.low_feature_fusion(x1_depth, feature1)
        # low_fusion128_d = self.low_feature_fusion2(low_fusion_d, low_d2)
        # low_fusion64_d = self.low_feature_fusion1(low_d1, low_fusion128_d)
        #
        # low = self.agant1(low_fusion64 + low_fusion64_d)
        # low = self.deconv1(low)
        # low = self.agant2(low)
        # low = self.deconv2(low)
        # low_out = self.outconv2(low)

        low = self.agant1(low_fusion64)
        low = self.deconv1(low)
        low = self.agant2(low)
        low = self.deconv2(low)
        low_out = self.outconv2(low)

        return decoder_out1, decoder_out2, decoder_out3, decoder_out4, low_out
    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
