---
layout:     post
title:      "Softmax层原理详解"
subtitle:   " \"公式+代码\""
date:       2017-12-05 12:00:00+0800
author:     "Sundrops"
header-img: "img/home-bg-faye.png"
catalog: true
tags:
    - deep learning基础学习
---


> 最近hinton很早就提出了一个结构名为capsule，旨在解决cnn的固有缺点，本文是第一篇实现hinton capsule结构的论文[Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)，本文很大程度上翻译自[“Understanding Dynamic Routing between Capsules (Capsule Networks)”
](https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/)代码来自: [XifengGuo CapsuleNet pytorch实现版本](https://github.com/XifengGuo/CapsNet-Pytorch/tree/master/result)

## CNN的问题 ##

### 直观认识 ###
CNN中，每个卷积核就是一个filter，卷积的操作是对应位置相乘后求和，其实就是求相关性，filter就是需要匹配的模板，各种filter具有不同的作用，有的对边缘敏感，有的对颜色敏感。每个神经元包含很多个filter，以用来检测特定的feature。

![这里写图片描述](http://img.blog.csdn.net/20171205162218238?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

如果我们训练好了一个人的分类器，送入网络一幅艺术画会怎么样，网络有多大可能会把它认成真人呢？

![这里写图片描述](http://img.blog.csdn.net/20171205210516129?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

CNN很擅长检测特定的feature如鼻子眼睛，但是却很少能挖掘出这些特征间的关系，比如视角大小方向等。比如下面的图片很有可能就能欺骗一个简单的cnn把它错认为是一个真正的人脸。

![这里写图片描述](http://img.blog.csdn.net/20171205210442043?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

一个简单的cnn提取到鼻子眼睛嘴巴的特征，进而就认为这个图是一个真正的人脸，但是网络根本没有挖掘到这些特征之间的关系如方向或者大小不对，所以“错误地”得到很高的脸激活值，就认为是人脸了。
![这里写图片描述](http://img.blog.csdn.net/20171205163154767?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

现在我们这样设定，每个神经元的输出不在仅仅是一个标量值，而是一个向量包含特征的一些属性，比如输出3个值[相似度，方向，大小]。有了这些空间信息，我们可以检测到鼻子眼睛之间不一致的方向和大小，最终脸的激活值就会第一点。注: 这里的激活值就变成了输出向量的模长。


![这里写图片描述](http://img.blog.csdn.net/20171205163551445?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


### 举一个视角的例子　###

想一下我们现在需要训练一个人脸检测的cnn模型，我们要怎么处理各种不同视角人脸的情况？比如有的是正脸，有的往左歪，有的往右歪。

![这里写图片描述](http://img.blog.csdn.net/20171205210610851?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

我们需要准备很多不同视角的人脸数据进行训练，然后得到最终的能检测各种视角人脸的模型。然而这种方法只是让模型记住了训练数据库，相当于训练了一个正脸２个歪脸的模型。它需要很大的数据量，而且并不定会存在覆盖这么多视角的数据。但是小孩子识别是不是脸，根本不需要这么多，可能只需要几十张就ok，对比一下，现有的cnn在利用数据方面效率太低了。

![这里写图片描述](http://img.blog.csdn.net/20171205164333817?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


#### Equivariance vs invariance ####

Ｃapsule致力于检测特征和它的各种变种，而不是像cnn那样只检测特征的一种特殊变种。

像下面两幅图，capsule要检测到不同方向的同一物体(它的变种)
![这里写图片描述](http://img.blog.csdn.net/20171205165326170?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171205165716159?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


**Invariance**指不管这个feature怎么变都可以检测到，比如可以检测到各种方向的鼻子，但是并没有检测到它的具体方向信息，这种空间信息的缺失最终会影响到invariance模型的性能

**Equivariance** 指检测可以相互转换的物体。比如一个capsule net可以检测到一个脸旋转了20度，而不是检测到一个旋转了20度的脸。capsule net强制模型学习特征的变种，我们可以利用更少的训练来推断更多的变种，而且可以剔除掉更多的错误的变种(goodfellow他们提的一些样本攻击)。有了capsule提取的特征的属性这些更多的信息，我们可以更好地把模型泛化到其他地方，省去大量标注数据的成本

### max pooling ###

max pooling是cnn用来保持Invariance的，比如一个物体轻微移动或旋转后，依然不改变它的类别。max pooling这样做丢失了空间等信息，虽然对分类影响不大，但是对检测分割等高级任务影响很大。(其实对重叠的多物体分类还是有影响的，因为它max pooling只保留主体最大响应了，把其他扔了，而后面介绍的capsule则不会把其他信息丢掉，重叠的部分是可以复用的)

## Capsule向量的计算 ##

回忆一下全连接网络：![这里写图片描述](http://img.blog.csdn.net/20171205210645859?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20171205210626394?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

而capsule则是

![这里写图片描述](http://img.blog.csdn.net/20171205210753258?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

1. 计算该层的第j个胶囊，先利用转换矩阵(m\*k)把上一层的胶囊(k\*1)转换一下维度u(m\*1)，然后上一层共n个胶囊，得到n个u(uj|i i=1->n)，然后这n个u分别乘以各自的权重cij得到sj(m\*1)
2. 注意没有偏置项
3. vj=squash(sj)。在cnn中会使用ReLU激活函数，这里不是的，采用squash函数(挤压函数)把这个sj向量放缩到0到单位长度。sj模长很小时，它的平方和1比可以忽略，很大时，1和它的平方比可以忽略。
4. 注: 这个转换矩阵编码low level的特征和high level的特征之间的空间或者其他联系

**总结和对比**

![这里写图片描述](http://img.blog.csdn.net/20171205161635217?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 动态路由协议 ##

![这里写图片描述](http://img.blog.csdn.net/20171205212237658?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

算法解释: 需要计算l+1层的第j个胶囊(论文中迭代次数r有1和3两个配置，次数不宜过多容易过拟合)，第l层的胶囊已经转换为u
1. 设置临时变量，bij=0，i为第l层的所有胶囊。这意味着起始时第l层的每个胶囊对第l+1层的第j个胶囊贡献一样，不确定性达到最大值，低层胶囊不知道它们的输出最适合哪个高层胶囊。
2. 利用b得到第l层各个胶囊的权重c，然后得到sj，最后经过squash函数得到vj
3. 更新bij，bij=bij+u和v的点积(点积检测胶囊的输入和输出之间的相似性)
4. 继续迭代直到达到迭代次数

随着迭代的持续，输入的某个胶囊如果和输出胶囊vj不相似，它的权重cij会越来越小，而相似的话，权重越来越大。对应论文的:

> Using an iterative routing process, each active capsule will **choose a capsule in the layer above to be its parent** in the tree. For the higher levels of a visual system, this iterative process will be solving the problem of assigning parts to wholes.

## CapsuleNet结构 ##

![这里写图片描述](http://img.blog.csdn.net/20171205213919763?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```python
# 第一层还是conv，(28-9/1)+1=20
self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)
'''
第二层是PrimaryCaps,使用conv实现的，(20-9/2)+1=6 6*6*(32*8)=6*6*256 8是胶囊向量的维度
只是把32个channel的feature map变成了32channel的8-d胶囊，相当于做了8次conv然后在channel这个维度concat起来了。论文所说的each capsule in the [6 × 6] grid is sharing their weights with each other所说的是算卷积时共享权重，都是同样的卷积核。
self.conv2d = nn.Conv2d(256, 256, kernel_size=9, stride=2, padding=0)
outputs = self.conv2d(x)
而后面digitCaps时他们是不共享的，而且相当于32*6*6个胶囊，对应这行代码
outputs = outputs.view(x.size(0), -1, self.8)
'''
self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=9, stride=2, padding=0)
# 实现capsule的动态路由部分
self.digitcaps = DenseCapsule(in_num_caps=32*6*6, in_dim_caps=8, out_num_caps=classes, out_dim_caps=16, routings=routings)
# 重构部分
self.decoder = nn.Sequential(
    nn.Linear(16*classes, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
    nn.Sigmoid()
)

```
## Loss function (Margin loss) ##


![这里写图片描述](http://img.blog.csdn.net/20171205215119168?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

Tc为gt，one-hot比如是第2类[0,1,0]，vc的模长为预测值[0.3, 0.6, 0,1]。m+ 为上margin，惩罚假阴性（没有预测到存在的分类的情况,  m- 为下margin，惩罚假阳性（预测到不存在的分类的情况）**注意Tc是一个one-hot的向量，第一个式子Tc意味着gt中有的类，你要预测出来，没有预测出来要惩罚，第一个式子1-Tc意味着gt中没有的类，你不能预测出来，预测出来了要惩罚**

## 实验结果 ##

论文中发现如果把重构误差计入，可以显著地提高准确率(貌似比动态路由提升得还多)
![这里写图片描述](http://img.blog.csdn.net/20171205222840794?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


DigitCaps输出10\*16，我们把把非gt对应的行都mask掉(\*0)，然后送入重构网络，得到下面。比如下图左侧，都是分类正确的重构，可以看到重构除了还原本身外，还起到了去噪的效果。右侧模型误把”5“识别成了”3“，通过重构，模型”告诉“我们，这是因为它认为正常的”5“的头是往右边伸出的，而给它的”5“是一个下面有缺口的”3“。
![这里写图片描述](http://img.blog.csdn.net/20171205223314629?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


DigitCaps输出10\*16，10是类别数。有规律地抖动16个维度中的某个维度，然后送入重构网络进行重构，可以发现这16个维度代表了不同属性，而且呈现有规律的变化。这些证明了后面的网络没有过拟合(随便输入都能重构)，因为结果是有规律的变化的。

![这里写图片描述](http://img.blog.csdn.net/20171205222949687?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

在识别重叠数字的时候，它显示了更强的重构能力，并且拒绝重构不存在的对象（右侧*号）。比如让它从非gt, 非预测类别进行重构，就会拒绝。

![这里写图片描述](http://img.blog.csdn.net/20171205223234500?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)





----------
参考了知乎上如何看待Hinton的论文《Dynamic Routing Between Capsules》
[云梦居客的回答](https://www.zhihu.com/question/67287444/answer/251460831)
[SIY.Z的回答](https://www.zhihu.com/question/67287444/answer/251241736)
引申阅读[Aurélien Géron介绍 Capsule Networks的视频教程](https://mp.weixin.qq.com/s/9BIbthQvePqeVdLWil7vgQ)
