import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
import math
########################################################################################################################



class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=80.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = True
        self.max_depth =max_depth
        self.min_depth = min_depth

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])


        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        # self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=crf_dims[0])

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))
        self.pool1 = nn.AvgPool2d(kernel_size=8, stride=8)  # 从 h/4 -> h/32
        self.pool2 = nn.AvgPool2d(kernel_size=4, stride=4)  # 从 h/8 -> h/32
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)  # 从 h/16 -> h/32
        self.conv1x1 = nn.Conv2d(2880, 512, kernel_size=1, stride=1, padding=0)
        self.fusion3 = nn.Conv2d(2048, 1536, kernel_size=1, stride=1, padding=0)
        self.fusion2 = nn.Conv2d(1280, 768, kernel_size=1, stride=1, padding=0)
        self.fusion1 = nn.Conv2d(896, 384, kernel_size=1, stride=1, padding=0)
        self.fusion0 = nn.Conv2d(704, 192, kernel_size=1, stride=1, padding=0)
        self.hypercompute = HyperComputeModule(512, 512, 10) # c1 c2 and threshold
        features_channels = [128, 128, 256]
        self.bts = bts(self.max_depth, features_channels, 512)

        # self.max_depth = max_depth

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        # self.decoder.init_weights()
        # if self.with_auxiliary_head:
        #     if isinstance(self.auxiliary_head, nn.ModuleList):
        #         for aux_head in self.auxiliary_head:
        #             aux_head.init_weights()
        #     else:
        #         self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)



    def forward(self, imgs, focal):

        feats = self.backbone(imgs)    # feats3 = [b, 1536, h/32, w/32]
        if self.with_neck:
            # feats = self.neck(feats)


            # 下采样第 1、2、3 个张量到 h/32, w/32
            x1_down = self.pool1(feats[0])  # [b, c, h/32, w/32]
            x2_down = self.pool2(feats[1])  # [b, 2c, h/32, w/32]
            x3_down = self.pool3(feats[2])  # [b, 4c, h/32, w/32]

            # x4 不需要下采样，直接使用
            x4_down = feats[3]  # [b, 8c, h/32, w/32]

            # 在通道维度拼接 (dim=1 是通道维度)
            x_concat = torch.cat([x1_down, x2_down, x3_down, x4_down], dim=1)  # [b, 15c, h/32, w/32]

            # 定义 1x1 卷积，将通道数从 15c 降到 512

            # print(x_concat.shape)
            # 通过 1x1 卷积
            output = self.conv1x1(x_concat)  # [b, 512, h/32, w/32]

            fusion_feats = self.hypercompute(output)    #  [b, 512, h/32, w/32]

            fusion = interpolate_fusion(fusion_feats)

            feats = feats + (fusion_feats,)

        # ppm_out = self.decoder(feats)      # [b, 512, h/32, w/32]

        ff3 = torch.cat((feats[3], fusion[3]), dim = 1)  # feats3= 1536  fusion[3] = 512
        ff3 = self.fusion3(ff3)

        ff2 = torch.cat((feats[2], fusion[2]), dim=1) # feats3= 768  fusion[2] = 512
        ff2 = self.fusion2(ff2)

        ff1 = torch.cat((feats[1], fusion[1]), dim=1)  # feats3= 384  fusion[1] = 512
        ff1 = self.fusion1(ff1)

        ff0 = torch.cat((feats[0], fusion[0],), dim=1)  # feats3= 192  fusion[0] = 512
        ff0 = self.fusion0(ff0)

        # e3 = self.crf3(ff3, ppm_out)  # [b, 1024, h/32, w/32] feats3= 1536  ppm = 512
        e3 = self.crf3(ff3, fusion_feats)
        f3 = nn.PixelShuffle(2)(e3)         # [b, 256, h/16, w/16]
        e2 = self.crf2(ff2, f3)        # [b, 512, h/16, w/16] feats2= 768
        f2 = nn.PixelShuffle(2)(e2)          # [b, 128, h/8, w/8]
        e1 = self.crf1(ff1, f2)         # [b, 256, h/8, w/8]  feats1= 384
        f1 = nn.PixelShuffle(2)(e1)           # [b, 64, h/4, w/4]
        e0 = self.crf0(ff0, f1)          # [b, 128, h/4, w/4]  feats0= 196

        # LPC
        # e00 = upsample(e0, 2)
        # # e000 = upsample(e00, 2)
        # features = [e1, e0, e00] # H/8 128 H/4 128  H/2 256
        #
        # # print(features.shape)
        # final_depth = self.bts(features, focal)



        if self.up_mode == 'mask':# 现有方法走的线性插值 而不是这个mask

            mask = self.mask_head(e0)
            d1 = self.disp_head1(e0, 1)
            d1 = self.upsample_mask(d1, mask)
        else:
            d1 = self.disp_head1(e0, 4)

        depth = d1 * self.max_depth

        return depth

def interpolate_fusion(fusion_feats: torch.Tensor) -> tuple:
    """
    将输入的 fusion_feats 上采样 3 次，生成一个包含 4 个张量的 tuple fusion。
    fusion[3] = 原始输入大小 [b, 512, h/32, w/32]
    fusion[2] = 上采样到 [b, 512, h/16, w/16]
    fusion[1] = 上采样到 [b, 512, h/8, w/8]
    fusion[0] = 上采样到 [b, 512, h/4, w/4]

    参数：
    - fusion_feats: 输入张量，形状为 [b, 512, h/32, w/32]

    返回：
    - fusion: 一个包含 4 个张量的 tuple
    """
    # 原始张量 (h/32, w/32)
    fusion_3 = fusion_feats  # [b, 512, h/32, w/32]

    # 上采样到 (h/16, w/16)
    fusion_2 = F.interpolate(fusion_feats, scale_factor=2, mode='bilinear', align_corners=False)  # [b, 512, h/16, w/16]

    # 上采样到 (h/8, w/8)
    fusion_1 = F.interpolate(fusion_2, scale_factor=2, mode='bilinear', align_corners=False)  # [b, 512, h/8, w/8]

    # 上采样到 (h/4, w/4)
    fusion_0 = F.interpolate(fusion_1, scale_factor=2, mode='bilinear', align_corners=False)  # [b, 512, h/4, w/4]

    # 将结果存储到 tuple 中
    fusion = (fusion_0, fusion_1, fusion_2, fusion_3)

    return fusion


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


class DispUnpack(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128):
        super(DispUnpack, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, output_size):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
        # x = torch.reshape(x, [x.shape[0], 1, x.shape[2]*4, x.shape[3]*4])
        x = self.pixel_shuffle(x)

        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


# HyperCompute learning module to fuse the feature from different scales
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse

class MANet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")


    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()


    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))

        return x



class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2, bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                              padding=1)
        self.ratio = ratio

    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.ratio, mode='bilinear')
        out = self.conv(up_x)
        out = self.elu(out)
        return out

class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final',
                                          torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                        kernel_size=1, stride=1, padding=0),
                                                              nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(
                                          nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)

        return net


class local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)

class bts(nn.Module):
    def __init__(self, max_depth, feat_out_channels, num_features=512):
        super(bts, self).__init__()
        self.max_depth = max_depth

        # self.upconv5 = upconv(feat_out_channels[4], num_features)
        # self.bn5 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        #
        # self.conv5 = torch.nn.Sequential(
        #     nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False),
        #     nn.ELU())
        # self.upconv4 = upconv(num_features, num_features // 2)
        self.bn4 = nn.BatchNorm2d(num_features//2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
            nn.ELU())
        self.bn4_2 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(num_features // 2 + num_features // 4 , num_features // 4, 6)
        self.daspp_12 = atrous_conv(num_features , num_features // 4, 12)
        self.daspp_18 = atrous_conv(num_features + num_features // 4 , num_features // 4, 18)
        self.daspp_24 = atrous_conv(num_features + num_features // 2 , num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc8x8 = reduction_1x1(num_features // 4, num_features // 4, self.max_depth)
        self.lpg8x8 = local_planar_guidance(8)

        self.upconv3 = upconv(num_features // 4, num_features // 4)
        self.bn3 = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc4x4 = reduction_1x1(num_features // 4, num_features // 8, self.max_depth)
        self.lpg4x4 = local_planar_guidance(4)

        self.upconv2 = upconv(num_features // 4, num_features // 8)
        self.bn2 = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias=False),
            nn.ELU())

        self.reduc2x2 = reduction_1x1(num_features // 8, num_features // 16, self.max_depth)
        self.lpg2x2 = local_planar_guidance(2)

        self.upconv1 = upconv(num_features // 8, num_features // 16)
        self.reduc1x1 = reduction_1x1(num_features // 16, num_features // 32, self.max_depth, is_final=True)
        self.conv1 = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),
                                         nn.ELU())
        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())

    def forward(self, features, focal):
        # skip0, skip1, skip2, skip3 = features[0], features[1], features[2], features[3]
        # dense_features = torch.nn.ReLU()(features[4])

        # upconv5 = self.upconv5(dense_features)  # H/16
        # upconv5 = self.bn5(upconv5)
        # concat5 = torch.cat([upconv5, skip3], dim=1)
        # iconv5 = self.conv5(concat5)
        skip0, skip1, skip2 = features[2], features[1], features[0] #  H/2, H/4 , H/8 256

        # upconv4 = self.upconv4(torch.nn.ReLU()(skip3))  # H/8
        # upconv4 = self.bn4(skip2)    # H/8
        # concat4 = torch.cat([skip2], dim=1)
        # iconv4 = self.conv4(concat4)
        # iconv4 = self.bn4_2(iconv4)
        iconv4 = self.bn4(skip2) # 256
        # iconv4 = skip2

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([skip2, daspp_3], dim=1)  # 256 + 128
        daspp_6 = self.daspp_6(concat4_2)               # 128
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)  # 384 + 128
        daspp_12 = self.daspp_12(concat4_3)                 # 128
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)  # 512 + 128
        daspp_18 = self.daspp_18(concat4_4)                  # 128
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)  # 640 + 128
        daspp_24 = self.daspp_24(concat4_5)                  # 128
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1) # 256 + 128 + 128 + 128 + 128 + 128
        daspp_feat = self.daspp_conv(concat4_daspp) # 128

        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = F.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8 = self.lpg8x8(plane_eq_8x8, focal)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.max_depth
        depth_8x8_scaled_ds = F.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='bilinear')

        upconv3 = self.upconv3(daspp_feat)  # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)

        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = F.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.max_depth
        depth_4x4_scaled_ds = F.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='bilinear')

        upconv2 = self.upconv2(iconv3)  # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)

        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = F.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.max_depth

        upconv1 = self.upconv1(iconv2)
        reduc1x1 = self.reduc1x1(upconv1)
        concat1 = torch.cat([upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
        iconv1 = self.conv1(concat1)
        final_depth = self.max_depth * self.get_depth(iconv1)
        # if self.params.dataset == 'kitti':
        # final_depth = final_depth * focal.view(-1, 1, 1, 1).float() / 715.0873

        return final_depth

