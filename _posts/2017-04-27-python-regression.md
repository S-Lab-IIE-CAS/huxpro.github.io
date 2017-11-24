---
layout:     post
title:      "Python利用梯度下降求多元线性回归"
subtitle:   " \"公式+代码\""
date:       2017-04-27 12:00:00+0800
author:     "Sundrops"
header-img: "img/home-bg-faye.png"
catalog: true
tags:
    - deep learning基础学习
---

之前一直看Ng的课程，以为掌握了，结果自己动手实现发现问题很多。
> **多元线性回归**
$向量形式：Y=W*X$
$$展开：y = w_0*x_0+w_1*x_1+...+w_n*x_n$$
$$参数:W: w_0,w_1,...w_n$$
$$代价函数：J(w_0,w_1,...w_n) = \frac{1}{2m}\sum_{i=0}^m(w_i*x_i-y_i)$$

``` python
# 批量创造一些数据
# y = w0*x0+w1*x1+w2*x2+w3*x3...wn*xn
# 我们要回归的w先提前随机生成
# 然后给出一些x，和w相乘后得到一些y，再加一些高斯噪声
# 即 y=wx+noise
# data_num生成的数据量，weight_num w的维度
def loadDataSet(data_num,weight_num):
    x = np.random.random((data_num,weight_num))
    w = np.random.random((weight_num,1))
    mu, sigma = 0, 0.1  # 均值与标准差
    noise = np.random.normal(mu, sigma, (data_num, 1))
    y = x.dot(w)+ noise
    print 'groundtruth_weight:'
    print w
    return x, y
```
##  1. 批梯度下降 ##
> 优化的代价函数是关于所有数据样本的loss，计算整个样本的loss后才更新权值，推导参考我的博客[梯度下降法](http://blog.csdn.net/u013010889/article/details/61658311)
> $$参数更新向量形式：W=W-\frac{1}{m}*(X^T(W*X-Y) * lr$$
$$展开：w_j = w_j - \frac{1}{m}\sum_{i=0}^m(w_i*x_i-y_i)*x_i*lr$$
**m**是所有样本的数量和

``` python
# x:(data_num,weight_num)
# init_w:(weight_num,1)
# y:(data_num,1)
def bgd(x,init_w,y,iter_size,lr):
	start_time = timeit.default_timer()
    w = init_w
    m = x.shape[0]
    for i in range(iter_size):
        predict = x.dot(w) # predict:(data_num,1)
        # x:(data_num,weight_num) x.T:(weight_num,data_num) y-predit:(data_num,1)
        # grad:(weight_num,1)
        grad = x.T.dot((predcit - y)) / m * lr
        w -= grad
    print w
    end_time = timeit.default_timer()
    print 'the time of cost: ',end_time - start_time
```
##  2. 随机梯度下降 ##
>优化的代价函数是单个数据样本的loss，计算每个样本的loss后就立即更新权值
 $$参数更新向量形式：W=W-\frac{1}{m}*(X^T(W*X-Y) * lr$$
$$展开：w_j = w_j - \frac{1}{m}\sum_{i=0}^m(w_i*x_i-y_i)*x_i*lr$$
**只是：m=1**

``` python
# 随机梯度下降
# x:(data_num,weight_num)
# init_w:(weight_num,1)
# y:(data_num,1)
def sgd(x, y, init_w, iter_size, lr):
    start_time = timeit.default_timer()
    w = init_w
    for i in range(iter_size):
        for j in range(x.shape[0]):
            temp_x = x[np.newaxis,j]
            temp_y = y[np.newaxis, j]
            predict = temp_x.dot(w)
            # x:(1,weight_num) x.T:(weight_num,1) y-predit:(1,1)
	        # grad:(weight_num,1)
            grad = temp_x.T.dot((predict - temp_y)) / 1 * lr
            w -= grad
    print w
    end_time = timeit.default_timer()
    print 'the time of cost: ', end_time - start_time
```
##  3. 小批量随机梯度下降 ##
>优化的代价函数是每块(batch_size)数据样本的loss，计算每快样本的loss后就更新权值
 $$参数更新向量形式：W=W-\frac{1}{m}*(X^T(W*X-Y) * lr$$
$$展开：w_j = w_j - \frac{1}{m}\sum_{i=0}^m(w_i*x_i-y_i)*x_i*lr$$
**只是：m=batch_size**

``` python
# 随机梯度下降
# x:(data_num,weight_num)
# init_w:(weight_num,1)
# y:(data_num,1)
ef minibgd(x,y,init_w,iter_size,lr,batch_size):
    start_time = timeit.default_timer()
    w = init_w
    batch_predict = np.zeros((batch_size, 1))
    batch_x = np.zeros((batch_size, init_w.shape[0]))
    batch_y = np.zeros((batch_size, 1))
    m = batch_size
    batch_num = 0
    for i in range(iter_size):
        for j in range(x.shape[0]):
            batch_x[batch_num] = x[np.newaxis, j]
            batch_predict[batch_num][0] = batch_x[batch_num].dot(w)
            batch_y[batch_num][0] = y[np.newaxis, j]
            batch_num += 1
            if batch_num==batch_size:
                batch_num = 0
                # x:(batch_size,weight_num) x.T:(weight_num,batch_size)
                # y-predit:(batch_size,1)
		        # grad:(weight_num,1)
                grad = batch_x.T.dot((batch_predict - batch_y)) / m *lr
                w -= grad
    print w
    end_time = timeit.default_timer()
    print 'the time of cost: ', end_time - start_time
```

## 4. 测试 ##

```python
data_num = 2000
weight_num = 5
x, y = loadDataSet(data_num,weight_num)
mu, sigma = 0, 0.1  # 均值与标准差
w = np.random.normal(mu, sigma, (weight_num,1))
bgd(x,y,w.copy(),100,0.1)
sgd(x,y,w.copy(),100,0.1)
minibgd(x,y,w.copy(),100,0.1,20)

```
由于批梯度下降是整个样本，可以利用numpy里面的矩阵乘法，速度较快。而随机梯度下降需要每个样本单独乘速度慢些，小批量梯度是保存到一个batch_size的样本后算乘法，但是由于多了赋值保存等操作还是比批梯度下降慢了些。
```
groundtruth_weight:
[[ 0.79517256]
 [ 0.3429605 ]
 [ 0.92893851]
 [ 0.28832528]
 [ 0.84102092]]
 ------------------------
 批梯度下降：
[[ 0.79504909]
 [ 0.33444026]
 [ 0.92603806]
 [ 0.28099184]
 [ 0.85376685]]
the time of cost:  0.0421590805054
随机梯度下降：
[[ 0.81632613]
 [ 0.33844041]
 [ 0.94321311]
 [ 0.30843523]
 [ 0.87303639]]
the time of cost:  10.9770891666
小批量随机梯度：
[[ 0.79647115]
 [ 0.33604063]
 [ 0.92717676]
 [ 0.28293051]
 [ 0.85364327]]
the time of cost:  4.24483799934
```
参考了两篇博客，十分感谢
[ Andrew Ng机器学习课程（一） - Python梯度下降实战](http://blog.csdn.net/mango_badnot/article/details/52328740?locationNum=10)
[机器学习公开课笔记(2)：多元线性回归](http://www.cnblogs.com/python27/p/MachineLearningWeek02.html)
