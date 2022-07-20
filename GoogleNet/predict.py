import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load image data
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "image file: '{}' does not exists".format(img_path)

    img = Image.open(img_path)
    # plt.imshow(img)
    # [C,H,W]
    img = data_transforms(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: {} does not exists".format(json_path)
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    # create model
    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)

    # load model weights
    weight_path = "./googleNet.pth"
    assert os.path.exists(weight_path), "file: {} does not exists".format(weight_path)
    # strict=False: 测试的时候不使用两个辅助分类层，训练的时候这里是包含辅助分类层的，所以不是模型权重不是严格对齐的，设置为False。
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.eval()
    with torch.no_grad():
        # predict
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_cla)],
                                               predict[predict_cla].numpy())
    print("results:\n{}\n".format(print_res))

    #     plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}, prob: {:.3}".format(class_indict[str(i)],
                                                 predict[i].numpy()))
    # plt.show()


if __name__ == "__main__":
    main()