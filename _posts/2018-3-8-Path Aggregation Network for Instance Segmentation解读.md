---
layout:     post
title:      "Path Aggregation Network for Instance Segmentation解读"
subtitle:   " \"Path Aggregation Network for Instance Segmentation\""
date:       2018-03-08 12:00:00+0800
author:     "Sundrops"
header-img: "img/home-bg-faye.png"
catalog: true
tags:
    - detection
    - instance segmentation
---

> 本篇论文是COCO 2017 instance segmentation的冠军，读了这篇论文再加上之前读论文的体会，和朱神交流后得到一个感悟: 同样一个work的小改动，你不能挖的深或者看得很浅，那你就是trick，而别人就能给科研界带来启发，ResNet很简单，但是kaiming他们就能把解决的问题和起因经过解释的很清楚，然后实验也很solid，给你一步一步解释，但是方法是十分简洁的。[arxiv link](https://arxiv.org/abs/1803.01534)

## Framework ##


![这里写图片描述](http://img.blog.csdn.net/20180308154240160?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

主要分为3个模块也是3点贡献
1. 改进的FPN: Bottom-up Path Augmentation
2. 改进之前的pool策略: Adaptive Feature Pooling
3. 改进mask分支: Fully-connected Fusion

## Bottom-up Path Augmentation ##

### motivation ###

FPN已经证明了加入一条top-down的旁路连接，能给feature增加high-level的语义性有利于分类。
但是low-level的feature是很利于定位用的，虽然FPN中P5也间接得有了low-level的特征，但是信息流动路线太长了如红色虚线所示(其中有ResNet50/101很多卷积层)，本文在FPN的P2-P5又加了low-level的特征，最底层的特征流动到N2-N5只需要经过很少的层如绿色需要所示(仅仅有几个降维的卷积)

![这里写图片描述](http://img.blog.csdn.net/20180308155151323?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

具体操作如下图所示，P2直接copy到N2，然后N2通过步长为2的3\*3卷积后分辨率缩小2倍，和P3尺寸一致，然后element-wise 相加。注: 所有channel和FPN中一致P2-P5, N2-N5都是256。

![这里写图片描述](http://img.blog.csdn.net/20180308155602856?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**这里就很有启示，只是简单加了low-level的特征，但是他们可以看到FPN信息流动的问题，能说出他们这样做的令人信服的道理以及后续的实验，虽然如何更直接地加low-level可能还有其他做法**

##  Adaptive Feature Pooling ##

在FPN那篇论文的解读中我们也讲过FPN从P2-P6(P6仅用作生成proposal，不用作RoIPooling时提取特征)多尺度地生成proposal，然后做RoIPooling时会根据proposal的大小将它分配到不同的level去crop特征，小的proposal去low-level的层，大的proposal去high-level的层。

这样做虽然简单也蛮有效，但它不是最好的处理方式，**尽管P2-P5(N2-N5)已经融合了low-level和high-level的特征，然后它们的主要特征还是以它本有的level为主** 这时如果小的proposal能从high-level层获取到更多的上下文语义信息是有利于它分类的，而大的proposal能从low-leve层获取到更好的细节是有利于它定位的。

因此本文打算每个proposal从所有level的特征上做RoIPooling，然后在后面融合，融合的阶段和方式都可实验，比如分类时是两个fc，这个融合阶段可以是fuse, fc1, fc2或者fc1, fuse, fc2，融合策略可是sum也可以是max，最后证明fc1, fuse, fc2和max最好。当然这个改进是增加了一些运算负担。

![这里写图片描述](http://img.blog.csdn.net/2018030816101744?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后最让我启发的是本文人家又做了实验，看看他们从所有level做RoIPooling有木有道理。因为我们后续融合时不是要做max融合嘛，这样我们可以看看到底是哪个level被选中了，这样计算一些比例。从下图可以到蓝色代表小的proposal，以此类推黄色是最大的proposal，我们以小的proposal为例，发现，只有30%确实是在最低的level被选中了，但是70%的feature值选中了更高的level。而最大的proposal，有超过50%的feature值选中了低的level(这也说明了给最高层加低层特征的必要性，即Bottom-up Path Augmentation的必要性)。

这也证明了从所有level做RoIPooling的必要性。

![这里写图片描述](http://img.blog.csdn.net/20180308161047784?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## Fully-connected Fusion ##

Mask RCNN中Mask分支就是个简版的fcn，fcn是全卷积网络，它根据一个局部的视野域来预测，且参数是全图共享，而全连接fc是全图视野域对位置更敏感，看得更大，这一点sunjian老师的large kernel也间接证明了大视野域的作用。因此本文打算多加一条用全连接层预测的支路来做mask预测，然后和fcn融合，具体做法如下图所示，至于conv4_fc接在fcn支路哪一个卷积后后面消融实验有做对比，conv3后面结果更好一点。

![这里写图片描述](http://img.blog.csdn.net/20180308162222702?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## Experiment ##

具体的消融实验部分不再贴出，详细见论文。总体的结果，backbone一个ResNet-50就打败了去年的冠军。

![这里写图片描述](http://img.blog.csdn.net/20180308163433694?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
