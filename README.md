## 0. 前言

我们希望利用多台机器的多张显卡做一件事情（分布式训练），使用Uber开发的Horovod框架。本文介绍该框架从安装到应用的相关细节。



## 1. 环境

- Ubuntu 20.04 机器若干
- Nvidia 2080Ti 若干
- CUDA 11.0



## 2. 安装Horovod步骤

> 参考 https://github.com/horovod/horovod/blob/master/docs/gpus.rst

### 2.0 cmake

```bash
sudo apt install -y cmake
```

### 2.1 安装NCCL2

参考 https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html，一共有4步，仔细阅读下图

![image-20210716165354396](https://gitee.com/qiangzibro/uPic/raw/master/uPic/image-20210716165354396.png)

第1步，根据自己的环境，操作：加一个ubuntu key

```bash
sudo apt-key add /var/nccl-local-repo-ubuntu2004-2.10.3-cuda11.0/7fa2af80.pub
```

第2步，`nccl-repo-<version>.deb`需要到官网 https://developer.nvidia.com/nccl下载。（注意要注册账号，NVIDIA官网很慢，梯子开全局才打开）

​	![image-20210716155002045](https://gitee.com/qiangzibro/uPic/raw/master/uPic/image-20210716155002045.png)



分别选择下列项![image-20210716155142838](https://gitee.com/qiangzibro/uPic/raw/master/uPic/image-20210716155142838.png)	

下载，安装

```bash
sudo dpkg -i nccl-local-repo-ubuntu2004-2.10.3-cuda11.0_1.0-1_amd64.deb
```

第3、4步

```bash
sudo apt update
sudo apt install libnccl2 libnccl-dev
```



### 2.2 安装MPI

根据文档提示得知

```text
Note: Open MPI 3.1.3 has an issue that may cause hangs. The recommended fix is to downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.
```

于是我们安装MPI 4.0.0，谷歌到它的软件包网址：https://www.open-mpi.org/software/ompi/v4.0/

![image-20210716171356679](https://gitee.com/qiangzibro/uPic/raw/master/uPic/image-20210716171356679.png)

点击下载或者用下列方式

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
```

安装（注意安装的时候需要管理员权限）

```bash
tar -xvzf openmpi-4.0.0.tar.gz
cd openmpi-4.0.0
./configure --prefix=/usr/local
make all install
```

:beers::beers:  搞定 ！

### 2.3 pip安装horovod

因为我们需要GPU版本，使用下面命令：

```bash
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
```

编译花了好一会儿时间，最终安装

![image-20210716160517662](https://gitee.com/qiangzibro/uPic/raw/master/uPic/image-20210716160517662.png)

:beers::beers:  搞定 ！



## 3. 分布式训练——以MNIST为例

### 3.1 代码、数据准备

我们直接跑一跑Uber官方提供的[例子](https://github.com/horovod/horovod/tree/master/examples)，再去研究相关细节

```bash
git clone https://github.com/horovod/horovod --depth 1
cd horovod/examples/pytorch
```

这里我们关注`pytorch_mnist.py` 这个脚本，注意到这个脚本用了一个第三方库，先安装之

```bash
pip install filelock
```

> 这个脚本对MNIST数据集的下载有些问题，第一次运行脚本下载好之后，报了一个`urllib.error.HTTPError: HTTP Error 503: Service Unavailable`的错误，解决方法是142行的download参数设为False即可。

### 3.2 单机多卡运行

在单台机器、双显卡的**本机**下运行:rocket::rocket:

```bash
horovodrun -np 2 -H localhost:2 python pytorch_mnist.py
```

成功运行，结果日志如下:beers::beers:

![image-20210716162636235](imgs/基于Horovod的多机多卡训练/image-20210716162636235.png)

### 3.3 多机多卡运行

仍然以mnist为例，假设我们有两台机器`192.168.3.3`,`192.168.3.4`，这两台机器：

- 都安装了Horovod
- anaconda安装位置、环境一样

直接在其中一个ip上运行，比如`192.168.3.3`上：

```bash
horovodrun --gloo --start-timeout 600 -np 4 -H 192.168.3.3:2,192.168.3.4:2 python pytorch_mnist.py
```



## 问题

- 安装Horovod需要步骤繁多，特别是不同机器环境不同的情况，比如我发现实验室大多CUDA驱动版本为11.1，但是Horovod的依赖，NCCL，其对应的CUDA只有11.0，11.4，10.2几个选择。自己的测试，CUDA11.1，安装了依赖CUDA11.0的NCCL，Horovod安装**成功**。

- `ORTE_ERROR_LOG: Data unpack would read past end of buffer in file grpcomm_direct.c at line 355`

  报如下错误

  ```text
  [GPU-2-3080-M5:11115] [[27185,0],1] ORTE_ERROR_LOG: Data unpack would read past end of buffer in file grpcomm_direct.c at line 355
  --------------------------------------------------------------------------
  An internal error has occurred in ORTE:
  
  [[27185,0],1] FORCE-TERMINATE AT Data unpack would read past end of buffer:-26 - error grpcomm_direct.c(359)
  
  This is something that should be reported to the developers.
  --------------------------------------------------------------------------
  ```

  解决方案：选择如下两种方式之一

  （1）增加`—gloo`参数，[参考](https://github.com/horovod/horovod/issues/2156#issuecomment-668090235)

  （2）hwloc版本有问题，清除（purge）掉`hwloc*`, [参考](https://github.com/horovod/horovod/blob/master/docs/troubleshooting.rst#force-terminate-at-data-unpack-would-read-past-end-of-buffer)。

  使用如下命令可以清除已安装的hwloc库。

  ```bash
  apt list | grep hwloc | grep installed | awk -F',' '{print $1}' | xargs -I{} apt purge -y {}
  ```

  

  

## 参考

[1] horovod的官方教学，值得参考 https://github.com/horovod/tutorials