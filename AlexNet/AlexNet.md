ILSVRC 2012（ImageNet的子集，1000类）：
训练集：1,281,167张已标注图片
测试集：50,000张已标注图片
验证集：100,000张未标注图片

该网络亮点：
（1）**首次使用GPU**进行网络加速训练；
（2）使用了ReLU激活函数，而不是传统的Sigmoid以及Tanh激活函数（缓解以往损失函数带来的求导过程麻烦和网络层数过深后的梯度消失两大痛点问题）；
（3）使用了LRN局部相应归一化；
（4）在全连接层的前两层中使用了Dropout随机失活神经元操作，以减少过拟合。

过拟合：根本原因是特征维度过多，模型假设过于复杂，参数过多，训练数据过少，噪声过多，导致拟合的函数完美预测训练集而对新数据的
测试集预测结果差。过度拟合训练数据，没有考虑到泛化能力。

《ImageNet Classification with Deep Convolutional Neural Networks》 2012年
http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

Tips：Local Response Normalization、Dropout、ReLU可以参考该论文。