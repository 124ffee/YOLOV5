# YOLOV5
YOLOV5的识别使用YOLOV5来训练一个自己检测模型。
要实现的效果，我们将会通过数据来训练一个检测的模型，并用pyqt5进行封装，实现图片检测、视频检测和摄像头实时检测的功能。
一.以下使用Windows系统环境配置
  1.1 为满足对不同python版本环境和可视化编程需求，这里安装pycharm和anaconda虚拟环境（当然你也可以使用vs看你个人使用习惯）。
  配置虚拟环境需要通过anaconda来完成，anaconda的下载地址为：https://www.anaconda.com/
  
  windows用户下载python3.8的miniconda也可下载地址为：https://docs.conda.io/en/latest/miniconda.html，
  下载完毕之后双击安装即可，注意一点这些一定要选中

（这里以anaconda为例）下载安装包进行安装

![capture_20230413125357230](https://user-images.githubusercontent.com/130628227/231659454-1973bc2b-aef9-475c-9dba-a9225d31f4ab.jpg)
![capture_20230413133053595](https://user-images.githubusercontent.com/130628227/231662515-06758baf-cded-44df-a62f-fedb2444dc3f.jpg)

程序安装完毕之后打开windows的命令行（cmd），输入conda，出现下列信息则表示conda已完成安装

![capture_20230413133309564](https://user-images.githubusercontent.com/130628227/231662778-4ff44fdd-02f7-4c8d-8bc4-f57d46eec0eb.jpg)

然后在命令行中输入下列指令创建虚拟环境，这里使用3.8.5版本（3.9.x以上的版本可能会导致opencv无法读图等问题）

      conda create -n yolo python==3.8.5


  1.2 为了方便查看和调试代码，我们这里使用pycharm，pycharm的下载地址为：https://www.jetbrains.com/pycharm/download/#section=windows
  
  1.3pytorch安装（gpu版本和cpu版本的安装）
实际测试情况是YOLOv5在CPU和GPU的情况下均可使用，不过在CPU的条件下训练那个速度会令人发指，所以有条件的小伙伴一定要安装GPU版本的Pytorch，没有条件的小伙伴最好是租服务器来使用。

需要注意以下几点：

安装之前一定要先更新你的显卡驱动，去官网下载对应型号的驱动安装
30系显卡只能使用cuda11的版本
一定要创建虚拟环境，这样的话各个深度学习框架之间不发生冲突
我这里创建的是python3.8的环境，安装的Pytorch的版本是1.8.0，命令如下：

      conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本

      conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可

pycocotools的安装
后面我发现了windows下更简单的安装方法，大家可以使用下面这个指令来直接进行安装，不需要下载之后再来安装

      pip install pycocotools-windows

其他包的安装
另外的话大家还需要安装程序其他所需的包，包括opencv，matplotlib这些包，不过这些包的安装比较简单，直接通过pip指令执行即可，我们cd到yolov5代码的目录下，直接执行下列指令即可完成包的安装。

      pip install -r requirements.txt
      pip install pyqt5
      pip install labelme

测试一下
在yolov5目录下执行下列代码

       python detect.py --source data/images/bus.jpg --weights pretrained/yolov5s.pt

执行完毕之后将会输出下列信息

![image-20210610111308496](https://user-images.githubusercontent.com/130628227/231671751-06baa587-bfe8-412f-bf4c-a1d255123d84.png)


在runs目录下可以找到检测之后的结果

![image-20210610111426144](https://user-images.githubusercontent.com/130628227/231671774-8600a0bb-4ef3-47b7-9c5d-86bfe0bdf863.png)

按照官方给出的指令，这里的检测代码功能十分强大，是支持对多种图像和视频流进行检测的，具体的使用方法如下：

         python detect.py --source 0  # webcam
 
                            file.jpg  # image 
                            
                            file.mp4  # video
                            
                            path/  # directory
                            
                            path/*.jpg  # glob
                            
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

数据处理
这里改成yolo的标注形式，之后专门出一期数据转换的内容。

数据标注这里推荐的软件是labelimg，通过pip指令即可安装

在你的虚拟环境下执行

      pip install labelimg -i https://mirror.baidu.com/pypi/simple命令进行安装，然后在命令行中直接执行labelimg软件即可启动数据标注软件。

![image-20210609172156067](https://user-images.githubusercontent.com/130628227/231672120-609f6764-133f-47bf-b986-6d82db4dc77a.png)

软件启动后的界面如下：

![image-20210609172557286](https://user-images.githubusercontent.com/130628227/231672323-7a74b48f-9f4f-4b42-b665-234215890860.png)

数据标注
虽然是yolo的模型训练，但是这里我们还是选择进行voc格式的标注，一是方便在其他的代码中使用数据集，二是我提供了数据格式转化

标注的过程是：

1.打开图片目录

![image-20210610004158135](https://user-images.githubusercontent.com/130628227/231672697-607f1e2f-1fb8-425d-9ab1-f617eb8a7cff.png)

2.设置标注文件保存的目录并设置自动保存

![image-20210610004215206](https://user-images.githubusercontent.com/130628227/231673761-23246185-3b94-454a-8c53-7faa0263f260.png)

3.开始标注，画框，标记目标的label，crtl+s保存，然后d切换到下一张继续标注，不断重复重复

![image-20211212201302682](https://user-images.githubusercontent.com/130628227/231673854-e9912d40-4cf4-4ad7-b847-f5f2e8d8ef7e.png)

labelimg的快捷键如下，学会快捷键可以帮助你提高数据标注的效率。

![image-20210609171855504](https://user-images.githubusercontent.com/130628227/231673933-ea80f1f9-dfc2-40b8-82aa-b0a5b1b80c72.png)

标注完成之后你会得到一系列的txt文件，这里的txt文件就是目标检测的标注文件，其中txt文件和图片文件的名称是一一对应的，如下图所示：

![image-20211212170509714](https://user-images.githubusercontent.com/130628227/231674034-2b1e72c4-57bd-4b7e-a0f1-7400e867f4d6.png)

打开具体的标注文件，你将会看到下面的内容，txt文件中每一行表示一个目标，以空格进行区分，分别表示目标的类别id，归一化处理之后的中心点x坐标、y坐标、目标框的w和h。

![image-20211212170853677](https://user-images.githubusercontent.com/130628227/231674119-7163f4ca-fccc-4059-a8ee-281538326cd2.png)

4.修改数据集配置文件

标记完成的数据请按照下面的格式进行放置，方便程序进行索引。

        YOLO_Mask
            └─ score
                 ├─ images
                 │    ├─ test # 下面放测试集图片
                 │    ├─ train # 下面放训练集图片
                 │    └─ val # 下面放验证集图片
                 └─ labels
                      ├─ test # 下面放测试集标签
                      ├─ train # 下面放训练集标签
                      ├─ val # 下面放验证集标签
                      
 这里的配置文件是为了方便我们后期训练使用，我们需要在data目录下创建一个mask_data.yaml的文件，如下图所示：
 
 ![image-20211212174510070](https://user-images.githubusercontent.com/130628227/231674548-90a35369-44d6-4a5c-941f-7aff2ef050fa.png)


到这里，数据集处理部分基本完结撒花了，下面的内容将会是模型训练！

模型训练
模型的基本训练
在models下建立一个mask_yolov5s.yaml的模型配置文件

模型训练之前，请确保代码目录下有以下文件

![image-20211212174920551](https://user-images.githubusercontent.com/130628227/231674717-b3e651e2-79a9-44af-8956-d43ba2690789.png)

 执行下列代码运行程序即可：
 
          python train.py --data mask_data.yaml --cfg mask_yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 100 --batch-size 4 --device cpu
 
 
 ![image-20210610113348751](https://user-images.githubusercontent.com/130628227/231674966-e92eb00d-3bba-487f-a92f-f4390fdf5f14.png)

训练代码成功执行之后会在命令行中输出下列信息，接下来就是安心等待模型训练结束即可。

![image-20210610112655726](https://user-images.githubusercontent.com/130628227/231675179-d0186290-c369-4443-a2f0-41abab57ddc4.png)

根据数据集的大小和设备的性能，经过漫长的等待之后模型就训练完了，输出如下：

![image-20210610134412258](https://user-images.githubusercontent.com/130628227/231675200-8cbc23de-86a2-4008-84ef-7d8c094be831.png)

在train/runs/exp3的目录下可以找到训练得到的模型和日志文件

![image-20210610145140340](https://user-images.githubusercontent.com/130628227/231675260-0f53b834-1d48-46ed-8c02-db5d5079daaf.png)

当然还有一些骚操作，比如模型训练到一半可以从中断点继续训练，这些就交给大家下去自行探索喽。

模型评估
出了在博客一开头你就能看到的检测效果之外，还有一些学术上的评价指标用来表示我们模型的性能，其中目标检测最常用的评价指标是mAP，mAP是介于0到1之间的一个数字，这个数字越接近于1，就表示你的模型的性能更好。

一般我们会接触到两个指标，分别是召回率recall和精度precision，两个指标p和r都是简单地从一个角度来判断模型的好坏，均是介于0到1之间的数值，其中接近于1表示模型的性能越好，接近于0表示模型的性能越差，为了综合评价目标检测的性能，一般采用均值平均密度map来进一步评估模型的好坏。我们通过设定不同的置信度的阈值，可以得到在模型在不同的阈值下所计算出的p值和r值，一般情况下，p值和r值是负相关的，绘制出来可以得到如下图所示的曲线，其中曲线的面积我们称AP，目标检测模型中每种目标可计算出一个AP值，对所有的AP值求平均则可以得到模型的mAP值，以本文为例，我们可以计算佩戴安全帽和未佩戴安全帽的两个目标的AP值，我们对两组AP值求平均，可以得到整个模型的mAP值，该值越接近1表示模型的性能越好。

关于更加学术的定义大家可以在知乎或者csdn上自行查阅，以我们本次训练的模型为例，在模型结束之后你会找到三张图像，分别表示我们模型在验证集上的召回率、准确率和均值平均密度。

![image-20211212175851524](https://user-images.githubusercontent.com/130628227/231675420-dec944d1-a6a5-4aa5-bc48-331e3c0ff9db.png)

以PR-curve为例，你可以看到我们的模型在验证集上的均值平均密度为0.832。

![PR_curve](https://user-images.githubusercontent.com/130628227/231675468-49d1f74b-357b-4e3a-a6d8-9cbc9bedbce9.png)

如果你的目录下没有这样的曲线，可能是因为你的模型训练一半就停止了，没有执行验证的过程，你可以通过下面的命令来生成这些图片。

        python val.py --data data/mask_data.yaml --weights runs/train/exp_yolov5s/weights/best.pt --img 640
        
 最后，这里是一张详细的评价指标的解释清单，可以说是最原始的定义了。
 
 ![20200411141530456](https://user-images.githubusercontent.com/130628227/231675603-0160040c-d446-40c4-9a43-1b604494096e.png)


模型使用
模型的使用全部集成在了detect.py目录下，你按照下面的指令指你要检测的内容即可

         # 检测摄像头
          python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source 0  # webcam
         # 检测图片文件
           python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source file.jpg  # image 
         # 检测视频文件
           python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source file.mp4  # video
         # 检测一个目录下的文件
           python detect.py --weights runs/train/exp_yolov5s/weights/best.pt path/  # directory
         # 检测网络视频
          python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'https://youtu.be/NUsoVlDFqZg'  # YouTube video
         # 检测流媒体
           python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream                   
           
比如以我们的口罩模型为例，如果我们执行

      python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source data/images/fishman.jpg

的命令便可以得到这样的一张检测结果。  
    
 ![fishman](https://user-images.githubusercontent.com/130628227/231675984-ff29ac7a-15fb-4b83-915e-6ec394ca8345.jpg)

构建可视化界面
可视化界面的部分在window.py文件中，是通过pyqt5完成的界面设计，在启动界面前，你需要将模型替换成你训练好的模型，替换的位置在window.py的第60行，修改成你的模型地址即可，如果你有GPU的话，可以将device设置为0，表示使用第0行GPU，这样可以加快模型的识别速度嗷。

![image-20211212194547804](https://user-images.githubusercontent.com/130628227/231676071-a9b1c11c-cd20-4909-9874-fc95826a1f32.png)

替换之后直接右键run即可启动图形化界面了，快去自己测试一下看看效果吧
