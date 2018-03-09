---
layout:     post
title:      "Relation Networks for Object Detection解读"
subtitle:   " \"Relation Networks for Object Detection\""
date:       2018-03-09 12:00:00+0800
author:     "Sundrops"
header-img: "img/home-bg-faye.png"
catalog: true
tags:
    - detection
---

> 现在做detection的竞争相当激烈，能记住的就是ross kaiming团队和sunjian老师团队，还有今天的主角daijifeng老师团队了[arxiv link](https://arxiv.org/abs/1711.11575)

## Motivation ##

众所周知，如果能model出物体之间的关系，那么对物体识别是大有裨益的。可是在深度学习领域上还没人把这个做work，当前主流的检测模型faster rcnn等都是使用RoIPooling后独立识别各个物体，并没有考虑他们之间的关系。所以本文就提出了一种轻量级的、in-place的物体关系模块来同时处理物体们的外形特征和几何特征之间的相互关系，而且它不需要额外的监督，可以很容易的嵌入到当前的网络模型中。

## Framework ##

本文提出的物体关系模块主要用于两个方面
1. 实例识别
2. 去重(之前都用NMS来做)

![这里写图片描述](http://img.blog.csdn.net/20180309125149336?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## Object Relation Module ##

关系模块的提出主要受启发于google的《Attention is all you need中Scaled Dot-Product Attention机制：
q是查询query，K是key的各个分量，V是value的各个分量。括号内代表当前查询q与key的相似度，用点乘(除以维度dk的开方)表示。softmax后得到该查询对应的各个value分量的权重，然后乘以V得到最后的输出value

![这里写图片描述](http://img.blog.csdn.net/20180309125637101?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
$f_R(n)$代表整个物体集合于第n个物体的关系，$w^{mn}$代表第m个物体对当前第n个物体的影响
![这里写图片描述](http://img.blog.csdn.net/20180309130552935?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
$w^{mn}$的计算里多加了一个几何特征，不然就和google那个Attention一模一样了，后续也会证明这点改动的必要性。
![这里写图片描述](http://img.blog.csdn.net/20180309131043345?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
${w_A}^{mn}$的计算和google的Attention计算一致,不过首先分别乘以$W_K, W_Q$进行降维到$d_k$

![这里写图片描述](http://img.blog.csdn.net/20180309131112112?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
${w_G}^{mn}$类似于ReLU操作，首先需要乘以进行$\varepsilon_G$进行升维，将4维的集合特征(关于x,y,w,h的公式)升到$d_g$
![这里写图片描述](http://img.blog.csdn.net/20180309131122253?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
这个4维几何特征相对于Ross那个有点变化，我们需要考虑远距离的物体，而Ross那个是最后的回归，很近的距离。
![这里写图片描述](http://img.blog.csdn.net/20180309132154680?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
所以比Ross的RCNN中的回归目标中的前两项多了log操作
![这里写图片描述](http://img.blog.csdn.net/20180309151124872?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
然后我们可以加入多个RM，最后cat起来，当然为了能in-place不能改变channel，所以${w_V}^{r}$要变成输入${f_A}^{m}$ channel的$\frac{1}{N_r}$
![这里写图片描述](http://img.blog.csdn.net/20180309132533835?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](http://img.blog.csdn.net/20180309125432398?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


## Relation for Instance Recognition ##

将上述的关系模块用于实例识别，可以直接用在RoIPooling后但是参数太多了，本文中在FC1和FC2后嵌入关系模块，不仅有自己的特征还会假如整个RoIPooling后出来的物体集合与当前识别proposal的关系特征。

![这里写图片描述](http://img.blog.csdn.net/20180309142843177?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## Relation for Duplicate Removal ##

之前的去重都是通过NMS，手动设置阈值，虽然简单有效但不是最优的策略。
本文通过关系模块来实现去重。首先对RoIPooling后的(一般为300)物体分类得到score然后对这些score进行排序得到rank，然后对rank进行升维到128维，和几何特征升维的做法一致，原有的特征也从1024转换到128，最后element-wise 相加后送入关系模块，然后输出值经过$W_s$变换再经过sigmoid得到另一个打分s1，最后与本来的分数s0相乘得到最终的score。其中$W_s$可对应到COCO中不同IoU的指标，得到多个分数s1。

![这里写图片描述](http://img.blog.csdn.net/20180309143139221?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## Experiments ##

具体的消融实验我就不再贴出了，这里我感兴趣的就是这个改进识别的部分，验证本文的提升不是由于参数多了，你看第b行把隐层的size改为1432，使得总参数为44.1M和第f行差不多，但是精度并没有明显提升。第c行改成3个fc精度反而下降了，第d行是加入一些残差的block也不行。**注意第e行，加入global，也没什么提升，我也做过这样的实验我是把这个global pool之后的2048-d的特征后面另加了fc6'和fc7'得到1024-d特征后再cat到原来的1024-d，是有提升的，可能我的参数更多了点。还有个疑问：不知道它的global pool是在根据roi扣出来的特征上做的，还是在整个feature上做的**

> global: 2048-d global average pooled features are concatenated with the second 1024-d instance feature before classification


![这里写图片描述](http://img.blog.csdn.net/2018030914560454?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzAxMDg4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)