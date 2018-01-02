---
layout:     post
title:      "CapsuleNet一个小例子"
subtitle:   " \"Dynamic Routing Between Capsules\""
date:       2018-01-01 12:00:00+0800
author:     "Sundrops"
header-img: "img/home-bg-faye.png"
catalog: true
tags:
    - deep learning基础学习
---

> 引用YouTube上一个up主的视频，讲解一个capsulenet的一个小例子


## 识别的过程 ##

假设我们要识别右面的"船"，经过卷积得到2个识别矩形和三角形的胶囊(即向量，之前一篇博客有介绍[ CapsuleNet解读](http://blog.csdn.net/u013010889/article/details/78722140))，这两个胶囊为简单起见假设只有一维，代表旋转角度。可以看到经过Conv后得到的primary capsules，每个feature map的位置有2个胶囊，每个胶囊都是一维的向量。蓝色箭头代表三角形检测胶囊，黑色箭头代表矩形检测胶囊，箭头的长度代表向量的长度，即最终的出现概率

![这里写图片描述](http://img.blog.csdn.net/20180101212006810?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


假设下一层只有两个胶囊，房子胶囊和船胶囊。由于矩形胶囊检测到一个旋转了16°的矩形，所以房子胶囊将检测到一个旋转了16°的房子，这是有道理的，船胶囊也会检测到旋转了16°的船。 这与矩形的方向是一致的。

![这里写图片描述](http://img.blog.csdn.net/20180101212708817?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**所以，为了做出这个预测，矩形胶囊所做的就是简单地计算一个变换矩阵$W_{ij}$与它自己的激活向量$u_i$的点积。在训练期间，网络将逐渐学习第一层和第二层中的每对胶囊的变换矩阵。 换句话说，它将学习所有的部分 - 整体关系，例如墙和屋顶之间的角度，等等。**

![这里写图片描述](http://img.blog.csdn.net/20180101212859892?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

三角形胶囊同理

![这里写图片描述](http://img.blog.csdn.net/20180101213056194?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

我们可以看到，矩形胶囊和三角胶囊在船胶囊的输出方面有着强烈的一致性。换言之，矩形胶囊和三角胶囊都同意船会以这样的形式输出来。然而，矩形胶囊和三角胶囊他们俩完全不同意房子胶囊的输出形式，从图中可以看出房子的输出方向是一上一下的。

![这里写图片描述](http://img.blog.csdn.net/20180101213128980?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

这个时候需要用到之前讲的动态路由协议了，首先起始权重都是一样的，三角形胶囊以0.5的概率输出房子，0.5概率输出船胶囊，矩形胶囊一样，最终房子胶囊和船胶囊如中间所示

![这里写图片描述](http://img.blog.csdn.net/20180101211142440?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

很明显，最终的船胶囊和三角形输出的船胶囊相似，而最终的房子胶囊和三角形输出的房子胶囊很不同，所以$c_{三角\_船}$增大，$c_{三角\_房子}$减小。矩形胶囊方面同理。
**注意上一篇博客说过了是$c_{三角\_船} + c_{三角\_房子} = 1$，而不是$c_{三角\_船} + c_{矩形\_船} = 1$，是在j上求和，而不是i**

经过两轮或者3轮迭代后，发现三角形胶囊越来越趋近于输出船，矩形胶囊也越来越趋近于输出船

![这里写图片描述](http://img.blog.csdn.net/20180101211153178?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


## 总结 ##

1. 以上可以看到在计算capsule向量时变换矩阵W的重要性。在训练期间，网络将逐渐学习第一层和第二层中的每对胶囊的变换矩阵。 换句话说，它将学习所有的部分 - 整体关系，例如墙和屋顶之间的角度等等
2. 动态路由部分，既屏蔽了噪声干扰识别又提高了capsulenet的可解释性，为什么会检测到船往回推就发现是检测到了倾斜了x度的三角形和矩形。


----------
参考:
[Aurélien Géron: Capsule Networks Tutorial](https://www.youtube.com/watch?v=pPN8d0E3900)
[【干货】Hinton最新 Capsule Networks 视频教程分享和PPT解读（附pdf下载）](https://mp.weixin.qq.com/s/9BIbthQvePqeVdLWil7vgQ)
