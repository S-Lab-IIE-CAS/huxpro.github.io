---
layout:     post
title:      "nvidia-docker快速迁移caffe环境"
subtitle:   " \"GPU+VNCserver+lxde桌面\""
date:       2017-02-22 12:00:00+0800
author:     "Sundrops"
header-img: "img/home-bg-faye.png"
catalog: true
tags:
    - docker
---

> 有时候我们自己在本机上费力搭建了很复杂的环境（cuda(5,6,7,8)+cudnn+opencv(2,3)+matlab等等其他依赖），我们想迁移这个环境到另一台机器上就需要重新安装一遍，如果另一台机器不是那么“干净”装了一些可能和你不一致的东西比如opencv，所以需要更换版本这时候需要彻底卸载干净，否则又会出现乱七八糟的问题，总之一句话迁移成本很高，风险很大。

回到正题，首先简单介绍下虚拟机和docker的区别


 1. vm与docker框架，直观上来讲vm多了一层guest OS，同时Hypervisor会对硬件资源进行虚拟化，docker直接使用硬件资源，所以资源利用率相对docker低也是比较容易理解的
 2. 让我们假设你有一个容器镜像（image）容量是1GB，如果你想用一个完整的虚拟机来装载，你得需要容量的大小是1GB乘上你需要虚拟机的数量。但使用Linux容器虚拟化技术（LXC）和AuFS，你可以共享1GB容量，如果你需要1000个容器，假设他们都运行在同样的系统影像上，你仍然可以用稍微比1GB多一点的空间来给容器系统，一个完整的虚拟化系统得到了分给它的自有全部资源，只有最小的共享。你获得了更多的隔离，但是这是很庞大的（需要更多的资源）使用Linux容器虚拟化技术（LXC），隔离性方面有所缺失，但是他们更加轻量，而且需要更少资源。所以你可以轻松运行1000个容器在一个宿主机器上，甚至眼都不眨。

## docker和nvidia-docker的区别 ##

> 由于我们深度学习需要用到GPU，使用docker时，需要映射设备等等，docker容器对宿主机的依赖就会很多也就失去了便捷，并不能让我们很舒服的迁移环境，nvidia-docker则很好的封装了这些，只需要容器内的cuda版本和宿主机相同就行（这个要求很低了，而且这个要求现在也基本可以通过docker hub上别人做好的带有各种cuda版本的镜像来满足，所以几乎无要求）
> 其实nvidia-docker只是run 和 exec命令和docker执行不同，其余的和docker执行的一模一样

## nvidia-docker安装(ubuntu为例) ##

 https://docs.docker.com/engine/installation/linux/ubuntu/ 详情参阅这个官方的指导


```
# **首先安装docker**
 1. sudo apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
 2. curl -fsSL https://apt.dockerproject.org/gpg | sudo apt-key add -
    #如果此处出现认证错误，curl后面-k忽略认证
 3.  sudo add-apt-repository \
       "deb https://apt.dockerproject.org/repo/ \
       ubuntu-$(lsb_release -cs) \
       main"
 4. sudo apt-get update
 5. sudo apt-get -y install docker-engine
 6. sudo docker run hello-world
    #它会自动下载这个hello-world镜像，然后运行，成功出现hello world就是docker就是装好了
 # **然后安装nvidia-docker**
 7. 下载nvidia-docker的安装包 https://github.com/NVIDIA/nvidia-docker/releases
   #deb类型: https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0/nvidia-docker_1.0.0-1_amd64.deb
 8. sudo dpkg -i nvidia-docker_1.0.0-1_amd64.deb
 ### 有一点需要注意docker默然使用root所以docker的每个命令需要sudo官方解释是安全，我们自己用则无妨
 # 创建docker组
    sudo groupadd docker
 # 将当前用户加入docker组
    sudo gpasswd -a ${USER} docker
 # 重新启动docker服务（下面是CentOS7的命令）
    sudo systemctl restart docker
 # .刷新Docker成员
    newgrp - docker
 # 之后使用docker就不用sudo了，稍微方便点
```
## docker hub ##

https://hub.docker.com 里面有很多别人做好的镜像，这里推荐一个https://hub.docker.com/r/kaixhin/cuda-caffe/

> cuda-caffe
---Ubuntu Core 14.04 + CUDA 8.0 + cuDNN v5 + Caffe. Includes Python interface.
Requirements
---NVIDIA Docker - see requirements for more details.
Usage
---Use NVIDIA Docker: nvidia-docker run -it kaixhin/cuda-caffe.
这个镜像的tags有6.5 7.0 7.5 8.0的cuda版本，基本满足使用

```   
 # 8.0意为cuda8.0，根据自己的需要修改，这个下载比较大所以很慢有空我会把我下好的镜像放到国内的百度云盘
   docker pull kaixhin/cuda-caffe:8.0
 # 下载后使用下面命令就会看到你已load的镜像，注意docker默认在/目录，所以注意/分区的大小
   docker images
 # REPOSITORY   TAG   IMAGE   ID   CREATED   SIZE
 # xxx          8.0    xxx     x     xxx       x
   nvidia-docker run -ti -p 宿主端口:docker容器端口 -v 宿主机地址:docker容器内地址 image名字(或者ID):8.0
   #不加tag默认是latest
   # -t 以为tty -i 意为可交互 如果-d就直接进后台 -v 就是文件映射可以用于宿主机和docker容器文件传输，-p是端口映射待会会用到
```

**到此处基本已经可以用终端来在容器中来跑你的网络了，但是有时候需要matlab或者需要个桌面环境比较顺手，这个时候就需要给容器里的linux装个桌面然后传输出来**
##安装LXDE桌面 VNCserver##

```
 # 先映射好端口进入容器
 nvidia-docker run -ti -p 5901:5901 kaixhin/cuda-caffe:8.0
 # 安装lxde vncserver
 sudo apt-get update
 sudo apt-get install xorg lxde-core tightvncserver
 # 或者到https://www.realvnc.com/download/vnc/linux/下载vncserver的deb包安装
 # 此镜像默认root用户，且没有设置USER和HOME环境变量启动vncserver时会有错误，所以先执行以下命令
  vim /root/.bashrc
 # 在最后加入以下代码后保存
  export USER=root
  export HOME=/root
 # 然后立即生效该环境变量
  source /root/.bashrc
 # 此处需要设置密码，还有一个view-only密码可选否
  vncserver -geometry 1024x768 :1 # 如果你启动时映射的是590n，那么此处就是vncserver :n
 # 其他(重新启动vncserver)
  vncserver -kill :1 && rm /tmp/.X1-lock && rm /tmp/.X1-lock
  vncserver -geometry 2400x1300 :1
```
## 通过VNCviewer连接容器内桌面 ##

下载VNCviwer https://www.realvnc.com/download/viewer/
然后输入地址：127.0.0.1:5901 然后输入你刚才设置的密码就成功了
lxde桌面调节分辨率
1. sudo vim /etc/xdg/lxsession/LXDE/autostart
最后一行添加
2. @xrandr --auto --output DVI-1 --primary --mode 1680x1050 --left-of DVI-0
## docker的其他 ##



```
 # 查看正在运行的容器
  docker ps
 # 删除所有容器
  docker rm `docker ps -a |awk '{print $1}' | grep [0-9a-z]`
 #  保存容器当前的状态到一个镜像
  docker commit -m "test" 容器ID 新镜像名字
 # 保存镜像到一个文件
  docker save 镜像名字>xxx.tar
 # 加载一个镜像文件tar
  docker load < xxx.tar
 # 查看加载的镜像
  docker images
 # 进入挂起的容器
  nvidia-docker exec -it 9f /bin/bash
 # 重启退出的容器
  nvidia-docker restart f9f
```
