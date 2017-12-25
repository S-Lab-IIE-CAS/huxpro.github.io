---
layout:     post
title:      "Light-Head R-CNN解读"
subtitle:   " \"Light-Head R-CNN: In Defense of Two-Stage Object Detector\""
date:       2017-11-27 12:00:00+0800
author:     "Sundrops"
header-img: "img/home-bg-faye.png"
catalog: true
tags:
    - detection
---

> 最近对检测很有兴趣哎，这些天写了好几个相关博客了，下一步准备写SSD和YOLO了，近段时间要把检测吃透

[Light-Head R-CNN: In Defense of Two-Stage Object Detector](https://arxiv.org/pdf/1711.07264.pdf)，名字很有趣，守护two stage检测器的尊严。

## Motivation ##
region-free的方法如YOLO，SSD，速度是很快，但是总体来说精度上还是不如两段的region-based系列的Faster rcnn(及加速版R-FCN)，那我们想要精度最高速度最快，就有两个做法了，提升region-free系列的精度(这个等我再二刷SSD后再想想有木有什么思路)，另一个就是提升region-based系列的速度了，本文就是后者。

首先Faster rcnn为什么还是很慢，在我上一篇博客[R-FCN解读](http://blog.csdn.net/u013010889/article/details/78630871)中已经提过，它的第二阶段每个proposal是不共享计算的，fc大量的参数和计算严重拖了速度(其实faster rcnn+res101已经做了努力，在res5c有个global pool到2014\*1\*1，不然第二阶段第一个fc参数参数更多。kaiming论文是C5作为fc6 fc7的，C5后面也依然有global pool，最后分类肯定要有一个全连接的，global pool之后参数也少很多(2048\*1\*1\*C，不然就是2048\*7\*7\*C))。而R-FCN就在着力于解决这个第二阶段的问题，通过生成一个k^2(C+1) channel的score map和PSRoIpooling可以去掉第二阶段的隐层fc，加速了很多。

但是R-FCN生成的score map是和C相关的，在MSCOCO上有81类需要生成7\*7\*81=3969个channel的score map，这个是耗时耗内存的。所以本文想生成一个thin 的feature map，这样可以显著加速还可以腾出“时间”在后面的rcnn部分做点文章提升精度。

## Approach ##
在Resnet101 res5c后利用大的可分离卷积，生成(α * p * p)channel的feature map，α本文取10，p和其他框架一致取7，这个最终只有10\*7\*7=490，而且与类别数C无关，大大减少了计算量，第二阶段的rcnn 子网络，本文为了提升精度，相比R-FCN多了一个隐层fc，这是thin feature map为它省来的计算空间，所以速度依然很快。

![这里写图片描述](http://img.blog.csdn.net/20171127182746193?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### Basic feature extractor. ###

两种设置，一种是ResNet101，设为L，一种是自己设计的简单的Xception网络，设为S

### thin feature map ###
参考论文[Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719)
本文设置k=15， Cmid = 64 for setting S,and Cmid = 256 for setting L

![这里写图片描述](http://img.blog.csdn.net/20171127201127575?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## Ablation experiment ##

### thin feature map###

首先说明直接把feature map变为thin feature map有什么影响啊？
做法就是直接把R-FCN得到的feature map用1*1的卷积降维到490，然后由于channel减少和类别树无关了，不能像R-FCN直接vote了(R-FCN vote后得到(C+1)\*1\*1，然后可以直接softmax，而本文vote的话得到10\*1\*1，不能直接softmax，所以加个10\*1\*1\*C的全连接分类，加个10\*1\*1\*4\*C的全连接回归，图中那个fc应该是cls和loc都有的)。B1是直接复现的[R-FCN](https://github.com/msracver/Deformable-ConvNets)，B2是改了点配置(1, 图片尺度和anchor scale增多 2, 回归的loss比重扩大1倍 3, 只选前256个loss大的样本进行反向传播)
从表中可以看到，channel变少了那么多后，精度并没有损失太多，把PSRoIpooing换成roipooling情况是一样的，甚至有些提升(因为PSRoIpooing后channel变成10channel了，而RoIPooling后channel还是490，参数多了些精度有些许提升)。而且这地方channel变少后集成FPN很方便，因为fpn会在很多level的feature上通过3\*3卷积生成这个(C+1)\*k\*k channel的feature map，十分耗内存和计算，详见我的另一篇博客[FPN解读](http://blog.csdn.net/u013010889/article/details/78658135)
![这里写图片描述](http://img.blog.csdn.net/20171127201415645?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](http://img.blog.csdn.net/20171127203158129?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### large separable convolution ###
把粗暴的1*1降维换成Large separable convolution，k=15， Cmid = 256，其他和R-FCN一样

![这里写图片描述](http://img.blog.csdn.net/20171127202554329?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### R-CNN subnet ###
我们在R-CNN subnet中多加了一个2048channel的隐层fc(无dropout，**注意区别于前面的实验是直接加个10\*1\*1\*C的全连接分类(对应表格第3行的数据)，这里要有个10\*1\*1\*2048的隐层fc，然后再有个2048\*C的全连接分类，loc类似**)，这个隐层fc是2048和1024 channel都差不多，参数变化很少，精度速度影响也不大。
从表上看到最终提升了2个点左右，而且注意由于用了thin feature map，速度是比它们快的。
![这里写图片描述](http://img.blog.csdn.net/20171127203615533?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## High Accuracy and High Speed##

本文把PSRoIpooling改成和RoIalign那样的插值，然后加上和其他model的同样配置和数据增强，精度是可以达到state-of-art的

![这里写图片描述](http://img.blog.csdn.net/20171127204155172?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

然后速度方面，把base model换成自己设计的"S"，速度也是可以秒掉SSD、YOLO等region-free以追求速度为主的model，同时精度和它们相当

![这里写图片描述](http://img.blog.csdn.net/20171127204354357?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171127204404354?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

下一步是不是要精度达到region-based，速度达到region-free呢，期待中(实力暂时不够，只能期待了)

## 部分代码 ##
```python
# light head
# large kernel
conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(15, 1), pad=(7, 0), num_filter=256, name="conv_new_1", lr_mult=3.0)
relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')
conv_new_2 = mx.sym.Convolution(data=relu_new_1, kernel=(1, 15), pad=(0, 7), num_filter=10*7*7, name="conv_new_2", lr_mult=3.0)
relu_new_2 = mx.sym.Activation(data=conv_new_2, act_type='relu', name='relu2')
conv_new_3 = mx.sym.Convolution(data=relu1, kernel=(1, 15), pad=(0, 7), num_filter=256, name="conv_new_3", lr_mult=3.0)
relu_new_3 = mx.sym.Activation(data=conv_new_3, act_type='relu', name='relu3')
conv_new_4 = mx.sym.Convolution(data=relu_new_3, kernel=(15, 1), pad=(7, 0), num_filter=10*7*7, name="conv_new_4", lr_mult=3.0)
relu_new_4 = mx.sym.Activation(data=conv_new_4, act_type='relu', name='relu4')
light_head = mx.symbol.broadcast_add(name='light_head', *[relu_new_2, relu_new_4])
# PSROIPooling
roi_pool = mx.contrib.sym.PSROIPooling(name='roi_pool', data=light_head, rois=rois, group_size=7, pooled_size=7, output_dim=10, spatial_scale=0.0625)
# 隐层fc
fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=2048)
fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')
# 分类和回归
cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_1_relu, num_hidden=num_classes)
bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_1_relu, num_hidden=num_reg_classes * 4)
```

----------
参考:
[terrychenism/Deformable-ConvNets](https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784)
