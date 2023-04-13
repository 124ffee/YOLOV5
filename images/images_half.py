#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yolov5-mask-42 
@File    ：images_half.py
@Author  ：这段代码使用了OpenCV（cv2）库来读取指定路径下的所有图像，并将它们的大小缩小到原来的一半，最后保存这些修改后的图像。
具体地，这行代码 `img_paths =[osp.join("yiqing/", x) for x in os.listdir("yiqing/")]` 根据指定路径 "yiqing/" 下的文件列表，生成了一个包含所有图像文件路径的列表 img_paths。然后，使用一个循环遍历这个列表中的所有图像文件路径并加载每个图像，调用 cv2.imread(img_path) 读取图像，接着将图像的大小调整为原来的一半（0.5），并使用 cv2.imwrite(img_path, img) 将缩小后的图像保存到原始图像所在的路径。
@Date    ：2022/3/2 12:03 
@Description：
'''
import cv2
import os
import os.path as osp

img_paths =[osp.join("yiqing/", x) for x in os.listdir("yiqing/")]
for img_path in img_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite(img_path, img)