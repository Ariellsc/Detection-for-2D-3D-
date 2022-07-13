import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    # specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "raw_data", "data_mine", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform['train'])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx  # dict format
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict to json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min(os.cpu_count(), batch_size if batch_size > 1 else 0, 8)  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val"]
    )
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=nw
    )
    print("Using {} images for training, {} images for validation.".format(train_num, val_num))


    ### test image show (shutdown)
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img/2 + 0.5  # un-normalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #     plt.savefig('./saved.jpg')
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))
    ### test image show (shutdown)

    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()

    # para = list(net.parameters())  # see net parameters
    optimizer = optim.Adam(net.parameters(), lr=2e-4)  # 0.0002

    epochs = 20
    train_steps = len(train_loader)
    # start train
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = 'train epoch[{}/{}] loss:{:.3f}'.format(epoch+1, epochs, loss)

    # validate
    net.eval()  # eval模式时，pytorch会自动把BN和Dropout固定住，不会取平均，而是使用训练好的值。
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # dim=1:第0维是batch，我们要第1维的结果数据；[1]:只取index，不要具体数值。
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。

    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f val_accuracy: %.3f'%
          (epoch+1, running_loss/train_steps, val_accurate))

    save_path = './AlexNet.pth'
    best_acc = 0.0
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)

    print("Training Done!")


if __name__ == "__main__":
    main()