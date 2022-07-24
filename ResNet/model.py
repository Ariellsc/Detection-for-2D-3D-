import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    定义resnet18 & resnet34的残差结构。
    """
    expansion = 1  # r18 & r34 每一个残差块儿中的两个卷积层中卷积核的个数是一模一样的（1倍关系），所以这里设置expansion=1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):  # 下采样downsample对应残差块的虚线shortcut部分
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 使用BN的时候是不需要加入偏置的
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)  # 使用BN的时候是不需要加入偏置的
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 如果下采样为None，也就是对应实线的残差连接。
        # 如果下采样不等于None，那么将input x输入下采样函数，得到捷径分支的输出。
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # identity mapping
        out = self.relu(out)

        return out


class BootleNeck(nn.Module):
    """
    定义resnet50 & resnet101 & resnet152层的残差结构。
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    expansion = 4  # 根据论文，r50 & r101 & r152，每个残差块中，第三个卷积层深度是前两个的4倍（256=64*4）

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(BootleNeck, self).__init__()
        width = int(out_channel * (width_per_group/64.)) / groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        #
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=groups,
                               kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        #
        self.conv3





