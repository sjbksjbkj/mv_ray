import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange,repeat,reduce


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class LearnedGroupConv(nn.Module):
    # progress代表全局的epoch进度，=cur_epoch/num_epoch
    global_progress = 0.0

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 condense_factor=None, dropout_rate=0.):
        super(LearnedGroupConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        # 在这使用nn.Conv2d来定义一个卷积权重和超参数，
        # 卷积权重可以进行梯度更新，但实质上并没有用到里面的卷积函数
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride,
                              padding, dilation, groups=1, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.condense_factor = condense_factor
        if self.condense_factor is None:
            self.condense_factor = self.groups
        ### Parameters that should be carefully used
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))
        ### Check if arguments are valid
        assert self.in_channels % self.groups == 0, \
            "group number can not be divided by input channels"
        assert self.in_channels % self.condense_factor == 0, \
            "condensation factor can not be divided by input channels"
        assert self.out_channels % self.groups == 0, \
            "group number can not be divided by output channels"

    def forward(self, x):
        self._check_drop()
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output
        weight = self.conv.weight * self.mask
        return F.conv2d(x, weight, None, self.conv.stride,
                        self.conv.padding, self.conv.dilation, 1)
    # 检查是否要进行进行新一轮stage的剪枝
    def _check_drop(self):
        progress = LearnedGroupConv.global_progress
        delta = 0
        ### Get current stage
        # 前半部分总epoch数的1/2用来C-1个condensing stage
        for i in range(self.condense_factor - 1):
            if progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        # stage的状态从0开始计数，所以condense_factor-1就是optim stage
        else:
            stage = self.condense_factor - 1
        ### Check for dropping
        if not self._reach_stage(stage):
            self.stage = stage  # 复值给self.stage当前的stage，
            delta = self.in_channels // self.condense_factor
        # 之后，如果没有发生self.stage的变化，delta就是0，不会发生剪枝
        if delta > 0:
            self._dropping(delta)
        return

    # delta=R/C
    # 生成mask向量
    def _dropping(self, delta):
        weight = self.conv.weight * self.mask
        ### Sum up all kernels
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        # OxRx1x1→OxR
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        # 交换0和1的维度，
        weight = weight.transpose(0, 1).contiguous()
        # 变为OxR
        weight = weight.view(self.out_channels, self.in_channels)
        ### Sort and drop
        for i in range(self.groups):
            # 一组这一段的filter weights
            wi = weight[i * d_out:(i + 1) * d_out, :]
            ### Take corresponding delta index
            # 通过L1_norm来选择重要的特征
            # self.count之前那是被mask掉的，所以最小的从self.count开始
            # [1]是获取sort()函数返回的下标
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                # 以i为起点,self.groups为步长，mask掉shuffle之前的卷积weights
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)

    def _reach_stage(self, stage):
        # 返回1或0，表示是否全部>=
        return (self._stage >= stage).all()

    @property
    def lasso_loss(self):
        if self._reach_stage(self.groups - 1):
            return 0
        weight = self.conv.weight * self.mask
        ### Assume only apply to 1x1 conv to speed up
        assert weight.size()[-1] == 1
        weight = weight.squeeze().pow(2)
        d_out = self.out_channels // self.groups
        ### Shuffle weight
        weight = weight.view(d_out, self.groups, self.in_channels)
        # 对应论文里的每组内，每列的和组成的新的weight
        weight = weight.sum(0).clamp(min=1e-6).sqrt()
        return weight.sum()


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = LearnedGroupConv(in_channels=channel, out_channels=dim,kernel_size=1)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = LearnedGroupConv(in_channels=dim, out_channels=channel,kernel_size=1)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph = self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h = h // self.ph, w=w // self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph   , pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        layers = [conv_nxn_bn(3, channels[0], stride=2)] #320 -- 160  3-16
        block1 = MV2Block
        block2 = MobileViTBlock
        conv = LearnedGroupConv

        # channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        layers.append(block1(channels[0], channels[1], 1, expansion))             #16-32
        layers.append(block1(channels[1], channels[2], 2, expansion)) #160 -- 80  #32-64
        layers.append(block1(channels[2], channels[3], 1, expansion))             #64-64

        layers.append(block1(channels[2], channels[3], 1, expansion))             #64-64
        layers.append(block1(channels[3], channels[4], 2, expansion)) #80 -- 40   #64-96
        layers.append(block2(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))  #96-96
        layers.append(block1(channels[5], channels[6], 2, expansion)) #40 -- 20   #96-128

        layers.append(block2(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))  #128-128
        layers.append(block1(channels[7], channels[8], 2, expansion)) #20 -- 10                       #128-160
        layers.append(block2(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))  #160-160
        layers.append(conv(kernel_size=1,in_channels=channels[-2], out_channels=channels[-1]))        #160-640

        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn
        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)


    def forward(self, x):

        y1 = self.features[0](x)
        # x = F.interpolate(x, size=(160, 160))
        x = self.pool2(x)
        x = self.conv(inp=3,oup=16)(x)
        x1 = x + y1

        y2 = self.features[1](x1)
        x1 = self.conv(inp=16,oup=32)(x1)
        x2 = x1 + y2

        y3 = self.features[2](x2)
        x2 = self.pool2(x2)
        x2 = self.conv(inp=32,oup=64)(x2)
        #x2 = F.interpolate(x+x1+x2,size=(80,80))
        x3 = x2 + y3

        y4 = self.features[3](x3)
        x4 = x3 + y4

        y5 = self.features[4](x4)
        x5 = x4 + y5

        y6 = self.features[5](x5)
        x5 = self.pool2(x5)
        x5 = self.conv(inp=64,oup=96)(x5)
        #x5 = F.interpolate(x+x1+x2+x3+x4+x5,size=(40,40))
        x6 = x5 + y6

        y7 = self.features[6](x6)
        x7 = x6 + y7

        y8 = self.features[7](x7)
        x7 = self.pool2(x7)
        x7 = self.conv(inp=96,oup=128)(x7)
        #x7 = F.interpolate(x + x1 + x2 + x3 + x4 + x5 + x6 + x7, size=(20, 20))
        x8 = x7 + y8

        y9 = self.features[8](x8)
        x9 = x8 + y9

        y10 = self.features[9](x9)
        x9 = self.pool2(x9)
        x9 = self.conv(inp=128,oup=160)(x9)
        #x9  = F.interpolate(x+x1+x2+x3+x4+x5+x6+x7+x8+x9,size=(10,10))
        x10 = x9 + y10

        y11 = self.features[10](x10)
        x11 = x10 + y11

        y12 = self.features[11](x11)
        x11 = self.conv(inp=160,oup=640)(x11)
        x = x11 + y12

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s(pretrained=False):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT((256, 256), dims, channels, num_classes=1000)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

