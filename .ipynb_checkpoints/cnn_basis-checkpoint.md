LeNet doc url: https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf 《Gradient-Based Learning Applied
to Document Recognition》

Prior Historical Knowledge：
CNN's Development:  
1986年：Rumelhart and Hinton provided “Back Propagation” algorithm；  
1998年：LeCun use BP train LeNet network, which imply that the CNN is coming out;  
2006年：Hinton proposed the concept of "deep learning" in his science paper for the first time;  
2012年：Hinton‘s student Alex Krizhevsky create a deep learning model,and got the first place of ILSVRC2012.  

CNN Basis：  
Fully Connected Layer：  
![Fully Connected Layer Image](https://github.com/Ariellsc/Vision-Classification/tree/main/material_img/fc.jpg)   
Back Propagation algorithm including 信号的前向传播和误差的反向传播两个过程:  
![network](https://github.com/Ariellsc/Vision-Classification/tree/main/material_img/network.jpg)  


CNN中独特的网络结构---卷积层：  
卷积特性：拥有局部感知机制（因为滑动窗口依局部滑动）；权值共享（因为滑动计算过程中，卷积核的矩阵值是不会发生变化的）。  
对于权值共享，举例理解：
![network](https://github.com/Ariellsc/Vision-Classification/tree/main/material_img/weight_shared.jpg)   
由于一张图片一般是3通道，那么每一个卷积核通道数量也要对应通道数量，如下图：  
![three_channel_conv](https://github.com/Ariellsc/Vision-Classification/tree/main/material_img/three_channel_conv.jpg)  
假设共有两层卷积操作，对于第一层卷积操作：每个通道在图片中的对应位置，卷积之后，相加，得到这一个卷积核在这一个位置上卷积后的值，如图中黄色矩阵左上角的“1”，为三个通道卷积后的值相加，具体的：1+0+0  
之后第二层卷积操作：第二个卷积核卷积结束，计算过程一样，将其与上一个卷积结果进行拼接。就得到了整个输出的特征矩阵。
总结一下就是：卷积核channel与输入特征层的channel相同；输出的特征矩阵channel与卷积核个数相同。  


问题：  
加上偏移量bias该如何计算呢？很简单，直接具体通道矩阵中对应位置按值加入该激活函数即可。  
加上激活函数该如何计算呢？review一下简单的activation functions:  
![activation_func](https://github.com/Ariellsc/Vision-Classification/tree/main/material_img/activation_func.jpg)  
卷积过程中出现越界怎么办呢？  
卷积操作过程中，矩阵经卷积操作后的尺寸由以下几个因数决定：输入图片的大小W x W；Filter大小F x F；步长S；padding的像素数P。  
卷积后矩阵尺寸大小计算公式： 
N = （W - F + 2P）/ S + 1  


池化层：
目的：对特征图进行稀疏处理，减少数据运算量。  
总结一下：池化层没有训练参数；只改变特征矩阵的w和h，不改变channel(深度)，一般pool size和stride相同（经验之谈）。