import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        if activation is None:
            super().__init__(conv2d, upsampling)
        else:
            super().__init__(conv2d, upsampling, activation)


class ConvBnRelu(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            add_relu: bool = True,
            use_batchnorm: bool = True,
            interpolate: bool = False
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups
        )
        self.add_relu = add_relu
        self.use_batchnorm = use_batchnorm
        self.interpolate = interpolate
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EASPPBranch(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(EASPPBranch, self).__init__()
        neck = inplanes // 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, neck, 1),
            nn.BatchNorm2d(neck)
        )

        self.atrous_conv = nn.Sequential(
            nn.Conv2d(neck, neck, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(neck),
            nn.Conv2d(neck, neck, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(neck)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(neck, planes, 1),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.atrous_conv(x)

        x = self.conv2(x)

        return x


class EASPP(nn.Module):
    def __init__(self, inplanes, planes):
        super(EASPP, self).__init__()
        neck = inplanes // 8
        self.conv0 = nn.Sequential(
            nn.Conv2d(inplanes, neck, 1),
            nn.BatchNorm2d(neck)
        )
        self.easpp_branch1 = EASPPBranch(inplanes, neck, dilation=3)
        self.easpp_branch2 = EASPPBranch(inplanes, neck, dilation=6)
        self.easpp_branch3 = EASPPBranch(inplanes, neck, dilation=12)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inplanes, neck, 1, bias=False),
            nn.BatchNorm2d(neck),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(neck * 5, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv0(x)

        x2 = self.easpp_branch1(x)

        x3 = self.easpp_branch2(x)

        x4 = self.easpp_branch3(x)

        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.final_conv(x)

        return x


class LightResidualBlock(nn.Module):
    def __init__(self, in_channels, is_depthwise=True, **kwargs):
        super(LightResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        if is_depthwise:
            self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        else:
            self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, x):
        residual = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        return out


class BnReluConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, use_bn=True):
        super(BnReluConv, self).__init__()
        padding = get_padding(kernel_size, stride, dilation)

        block = [nn.BatchNorm2d(in_channels)] if use_bn else []
        block += [nn.ReLU(inplace=True),
                  nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class SqueezeResidualBlock(nn.Module):
    def __init__(self, in_channels, squeeze_factor=4, use_bn=True):
        super(SqueezeResidualBlock, self).__init__()
        squeeze_channels = in_channels // squeeze_factor

        self.block1 = BnReluConv(in_channels, squeeze_channels, kernel_size=1, use_bn=use_bn)
        self.block2 = BnReluConv(squeeze_channels, squeeze_channels, kernel_size=3, use_bn=use_bn)
        self.block3 = BnReluConv(squeeze_channels, in_channels, kernel_size=1, use_bn=use_bn)

    def forward(self, x):
        residual = x

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out += residual
        return out


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = self.bn_function_factory()
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

    def bn_function_factory(self):
        norm, relu, conv = self.norm1, self.relu1, self.conv1

        def bn_function(*inputs):
            concated_features = torch.cat(inputs, 1)
            bottleneck_output = conv(relu(norm(concated_features)))
            return bottleneck_output

        return bn_function


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_interp_size(input_tensor, s_factor=1, z_factor=1):  # for caffe
    ori_h, ori_w = input_tensor.shape[2:]

    # shrink (s_factor >= 1)
    ori_h = (ori_h - 1) // s_factor + 1
    ori_w = (ori_w - 1) // s_factor + 1

    # zoom (z_factor >= 1)
    ori_h = ori_h + (ori_h - 1) * (z_factor - 1)
    ori_w = ori_w + (ori_w - 1) * (z_factor - 1)

    resize_shape = ori_h, ori_w
    return resize_shape


def interp(input_tensor, output_size, mode="bilinear"):
    n, c, ih, iw = input_tensor.shape
    oh, ow = output_size

    # normalize to [-1, 1]
    h = torch.arange(0, oh, dtype=torch.float, device=input_tensor.device) / (oh - 1) * 2 - 1
    w = torch.arange(0, ow, dtype=torch.float, device=input_tensor.device) / (ow - 1) * 2 - 1

    grid = torch.zeros(oh, ow, 2, dtype=torch.float, device=input_tensor.device)
    grid[:, :, 0] = w.unsqueeze(0).repeat(oh, 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(ow, 1).transpose(0, 1)
    grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)  # grid.shape: [n, oh, ow, 2]
    if input_tensor.is_cuda:
        grid = grid.cuda()

    return F.grid_sample(input_tensor, grid, mode=mode)


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


