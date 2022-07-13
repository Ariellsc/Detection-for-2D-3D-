import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet(from Hinton and his student Alex Krizhevsky)是2012年ILSVRC 2012竞赛的冠军网络，自此深度学习开始迅速发展。
    分类准确率由70%+提升到80%+。
    """
    def __init__(self, num_classes=5, init_weights=False):
        super(AlexNet, self).__init__()
        # define self.features==>特征提取模块
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2),  # input[3,32,32], output[48,55,55]
            nn.ReLU(inplace=True),  # inplace选项: 是否将得到的值计算得到的值覆盖之前的值,对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, padding=2),  # output[128, 14, 14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 7, 7]
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1),  # outtput[192, ]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # define classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*6*6, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

