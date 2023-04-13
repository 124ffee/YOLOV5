# YOLOV5
YOLOV5的识别使用YOLOV5来训练一个自己检测模型。
要实现的效果，我们将会通过数据来训练一个检测的模型，并用pyqt5进行封装，实现图片检测、视频检测和摄像头实时检测的功能。
一.以下使用Windows系统环境配置
  1.1 为满足对不同python版本环境和可视化编程需求，这里安装pycharm和anaconda虚拟环境（当然你也可以使用vs看你个人使用习惯）。
  配置虚拟环境需要通过anaconda来完成，anaconda的下载地址为：https://www.anaconda.com/
  
  windows用户下载python3.8的miniconda也可下载地址为：https://docs.conda.io/en/latest/miniconda.html，
  下载完毕之后双击安装即可，注意一点这些一定要选中
  
![114f0bfcbd637b396a431782a5349858](https://user-images.githubusercontent.com/130628227/231653265-b0f147e8-1530-4e6b-af36-94590106f87b.png)
![df4ea285735dd5efa4382b2e35704459](https://user-images.githubusercontent.com/130628227/231653286-6232123d-43bc-4559-86b2-3f66b596a295.png)

（这里以anaconda为例）下载安装包进行安装

![capture_20230413125357230](https://user-images.githubusercontent.com/130628227/231659454-1973bc2b-aef9-475c-9dba-a9225d31f4ab.jpg)
![capture_20230413133053595](https://user-images.githubusercontent.com/130628227/231662515-06758baf-cded-44df-a62f-fedb2444dc3f.jpg)

程序安装完毕之后打开windows的命令行（cmd），输入conda，出现下列信息则表示conda已完成安装

![capture_20230413133309564](https://user-images.githubusercontent.com/130628227/231662778-4ff44fdd-02f7-4c8d-8bc4-f57d46eec0eb.jpg)

然后在命令行中输入下列指令创建虚拟环境，这里使用3.8.5版本（3.9.x以上的版本可能会导致opencv无法读图等问题）
conda create -n yolo python==3.8.5
这条指令的含义是创建python版本为3.8.5，名称为yolo的虚拟环境

![capture_20230413133721490](https://user-images.githubusercontent.com/130628227/231663539-b41dc7c4-4ed4-46af-9302-a7a5897f33bc.jpg)
![capture_20230413133738681](https://user-images.githubusercontent.com/130628227/231663549-312c72e5-4354-48f5-aa05-c7782655d903.jpg)

安装结束之后输入下列指令激活虚拟环境，出现小括号(yolo)表示环境激活成功
conda activate yolo

为了加速下载添加anaconda国内镜像配置
清华TUNA提供了 Anaconda 仓库的镜像,运行以下三个命令:
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

添加USTC仓库镜像：
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple

Conda 附加库:
Conda Forge
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/

msys2
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/

bioconda
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/

menpo
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/

  1.2 为了方便查看和调试代码，我们这里使用pycharm，pycharm的下载地址为：https://www.jetbrains.com/pycharm/download/#section=windows
  
  1.3 pytorch安装（gpu版本和cpu版本的安装）
实际测试情况是YOLOv5在CPU和GPU的情况下均可使用，不过在CPU的条件下训练那个速度会令人发指，所以有条件的小伙伴一定要安装GPU版本的Pytorch，没有条件的小伙伴最好是租服务器来使用。
GPU版本安装的具体步骤
最近比特币的热潮慢慢褪去，显卡的价格也下来了，所以小伙伴们可以观察一下最近的行情，合适的时候可以入手几块显卡来搞深度学习。关注我的朋友大多数来自大作业怎么搞系列教程，大作业怎么搞系列使用tensorlfow训练了几个物体分类模型，在开源出的代码中我也基本给了大家我训练好的模型，有的朋友反应自己跑训练的过程的时候速度比较慢，这个原因大多都是因为大家是在CPU环境下来跑的，所以速度不是很快，如果使用GPU来跑训练的话，效率上差不多是CPU的10倍。

！！！注：深度学习的小伙伴请选择Nvidia（英伟达）的显卡

显卡的品牌基本可以分为两大阵营：AMD和Nvidia。如果是深度学习使用的话大家务必要选择Nvidia的显卡，AMD显卡性价比虽然比较高，但是其对深度学习的支持不是很好，所以大家购买之前一定要看清是不是英伟达的显卡.

认识显卡（GPU）
购买显卡之前，我们以Nvidia的显卡为例，说明一下显卡的型号上面的数字都表示什么意思，影驰表示的是显卡的制造商，Geforce是显卡系列名称，GTX表示显卡的档次，数字一般是四位，前两位表示显卡的代数，16就表示是第16代显卡，中间的第三位数字表示的是显卡的性能级别，数字越大显卡的性能级别就相对越好，最后一位一般都是0，不用去管，有的显卡还有英文的后缀，SE表示阉割版、TI和Super表示增强的版本，比如1660TI就是在1660的基础上进行了增强，6G表示显卡的显存是6个GB，最后的比较炫酷的名字是制造商定的名字，这个一般就是个噱头，不需要太关注。

![fda701f7dfa577c30bb1de5f27b0ab72](https://user-images.githubusercontent.com/130628227/231666653-70e66be6-d166-466d-be48-730d0ce68437.jpg)


架构：相当于运行布局，布局越好跑的越流畅。
工艺：制程越小精度越高，越能发挥更多性能。
光栅以及流处理器：相当于劳动力，人越多执行力越强。
核心频率：反应速度，相当于跑车百米提速效率。
显存频率：相当于限速标志，决定了最大运行速度。
显存位宽：相当于划线，决定了最大运行通道。
显存容量：相当于道路限宽，决定了最大承载量。

1.3.1安装显卡驱动
去官网下载对应你显卡的驱动程序下载之后执行程序并进行安装即可，为了防止不必要的情况出现，安装的时候请按照默认选项执行。
安装完毕之后重启电脑在cmd中输入nvidia-smi，输出下列信息则表示显卡驱动安装成功。

![963ce2373379b560189f973275b4f978](https://user-images.githubusercontent.com/130628227/231666960-cb497a78-538a-4c2f-958c-311d107d2ea8.png)

1.3.2安装GPU版本的Tensorflow
GPU版本Tensorflow的安装

创建并激活虚拟环境
打开cmd，首先创建虚拟环境，分别输入下面两条命令，完成虚拟环境的创建和激活
安装
我们首先需要使用conda安装cuda和cudnn
conda install cudatoolkit=10.1
conda install cudnn==7.6.5

然后使用pip指令安装gpu版本的tensorflow
pip install tensorflow-gpu==2.3.0

测试GPU是否可用
现在在命令行中测试一下GPU是否可用，首先输入python进入python的解释器中
输入下面两条指令，如果输出为True则表示GPU可以使用
import tensorflow as tf
print(tf.test.is_gpu_available())

tensorflow如果安装的是GPU版本的则默认使用GPU，大家不需要在代码中指定，直接使用即可

1.3.3安装GPU版本的Pytorch
GPU版本Pytorch的安装

创建并激活虚拟环境
打开cmd，首先创建虚拟环境，分别输入下面两条命令，完成虚拟环境的创建和激活

安装
我们这里使用conda进行gpu版本pytorch的安装，非常方便，直接在激活的虚拟环境中输入下列命令即可，这里安装的是最新版本的Pytorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2

30系的朋友需要cuda11的支持，请执行下列命令
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

如果需要指定版本号，请这样执行
conda install pytorch==1.5.0 torchvision==0.6.1 cudatoolkit=10.2

测试GPU是否可用
现在在命令行中测试一下GPU是否可用，首先输入python进入python的解释器中输入下面两条指令，如果输出为True则表示GPU可以使用
import torch
print(torch.cuda.is_available())

不过Pytorch会讲究使用device来指定GPU，需要大家通过to()方法做下转移

