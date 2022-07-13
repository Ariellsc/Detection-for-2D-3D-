import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from model import LeNet


def main():
    # 1.define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert PIL format (H, W, C) to (C, H ,W)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # 2.define train dataset
    train_set = torchvision.datasets.CIFAR10(root='/data/dataPart2/liushichao/raw_data/data_mine/cifar10', train=True,
                                             transform=transform, download=False)  # set download true when first using

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32,
                                               shuffle=True, num_workers=8)

    # 3.define val dataset
    val_set = torchvision.datasets.CIFAR10(root='/data/dataPart2/liushichao/raw_data/data_mine/cifar10', train=False,
                                           transform=transforms, download=False)  # set download true when first using
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=3000,
                                             shuffle=False, num_workers=8)

    val_data_iter = iter(val_loader)  # define a iterator
    val_image, val_label = val_data_iter.next()  # get validation images and labels iterately

    # class labels
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 4.实例化network
    net = LeNet()
    # 5.define loss_function; note that cross entropy loss function including logsoftmax loss and null loss, that's why we don't use softmax in our forward operation(see model.py).
    loss_function = nn.CrossEntropyLoss()
    # 6.define optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)  # 所有网络参数都参与优化

    # 7.start train
    for epoch in range(10):
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # 1、get input data, which is a list of [inputs, labels]
            inputs, labels = data

            # 2、zero the parameter gradients
            # 不手动清零，torch会进行历史梯度累加，实现了batchsize的变相扩大（一定条件下，batchsize越大效果越好）。
            optimizer.zero_grad()  # 反向传播前手动清零梯度的原因，https://www.zhihu.com/question/303070254

            # 3、loss
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            # 4、backward
            loss.backward()
            # 5、optimization
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 steps
                with torch.no_grad():  # don't update gradient during val stage
                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]  # dim=1:第0维是batch，我们要第1维的结果数据；[1]:只取index，不要具体数值。
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    print('[%d, %3d] train_loss: %3f test_acc: %3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))

                    running_loss = 0.0
        print('Training Done!')

        save_path = './LeNet.pth'
        torch.save(net.state_dict, save_path)


if __name__ == '__main__':
    main()