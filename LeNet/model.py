import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    一般是定义一个类，比如此处定义的LeNet类，继承自父类nn.module，包含两个方法，init方法中定义需要使用到的网络结构，forward方法中定义反向传播过程。
    当我们实例化这个类之后，将参数传入这个实例化之后的类，就会按照这个forward方法进行正向传播过程。
    1998年：LeCun use BP train LeNet network，设备受限，CPU计算。
    """

    def __init__(self):
        """
        卷积后矩阵大小：N = (W - F + 2P)/S + 1
        """
        super(LeNet, self).__init__()  # 解决多重继承，继承父类的构造函数
        self.conv1 = nn.Conv2d(3, 16, 5)  # in_channels, out_channels, kernel_size; 使用16个卷积核
        self.pool1 = nn.MaxPool2d(2, 2)  # kernel_size, stride; 池化不改变深度，只改变宽度和高度
        self.conv2 = nn.Conv2d(16, 32, 5)  # 使用32个卷积核，对应out_channel的32, 卷积核尺寸依然为5
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # in节点个数, out节点个数
        self.fc2 = nn.Linear(120, 84)  # in_features, out_features/in节点个数, out节点个数
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(c=3, h=32, w=32) output(16, 28, 28)==> (32 - 5 + 2*0)/1 + 1 = 28
        x = self.pool1(x)  # output(16, 14, 14) ==> (28 - 2 + 2*0)/2 + 1 = 14
        x = F.relu(self.conv2(x))  # output(32, 10, 10) ==> (14 - 5 + 2*0)/1 + 1 = 10
        x = self.pool2(x)  # output(32, 5, 5) ==> (10 - 2 + 2*0)/2 + 1 = 5
        x = x.view(-1, 32 * 5 * 5)  # 全连接层需要一维向量作为输入，第一个维度设置为-1（自动推理），第二个维度为节点个数;[32, 800]=[Batch, 32*5*5]
        x = F.relu(self.fc1(x))  # x.shape: (32, 120) output: 120
        x = F.relu(self.fc2(x))  # x.shape: (32, 84) output: 84
        x = self.fc3(x)  # x.shape: (32, 10), output: 10; 这里省略了softmax操作，因为使用到到的交叉熵损失函数中有对应的更加高效的softmax方法操作。

        return x

## test
# import torch
# input1 = torch.randn(32, 3, 32, 32) # [B, C, H, W]
# model = LeNet()
# output = model(input1)