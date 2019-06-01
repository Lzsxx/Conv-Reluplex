# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
import pylab
import tensorflow as tf
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

folder = p.file_base + "/s2_parameter/"
folder_divided = p.file_base + "/s2_parameter_divided/"

w = p.w
h = p.h
c = p.c


# 读取图片及其标签函数
def read_image(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    idx = 0
    for index,folder in enumerate(label_dir):
        folderName = folder[-1:]
        for img in glob.glob(folder+'/*.png'):
            print(str(idx) + ": reading the image:%s"%img)
            image = io.imread(img)
            image = transform.resize(image,(w,h,c), mode='constant')
            images.append(image)
            labels.append(folderName)
            idx = idx + 1
            # show_image_label(image, index)
    return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32)


# 每次获取batch_size个样本进行训练或测试
def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]


# 读取图片及其标签函数
def show_image_label(img, label):
    # print("before restore:")
    # print(img)
    # 将图片的取值范围改成（0~255）
    img = img * 255
    img = img.astype(np.uint8)
    # print("after restore:")
    # print(img)

    # print("label:", label)
    # 将形状更改回图片格式，画出图片
    img_restore = img.reshape(w, h)
    io.imshow(img_restore)
    pylab.show()


def mkdir_if_not_exit():
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(folder_divided):
        os.mkdir(folder_divided)


def compute_map_num(data):
    # 计算有多少个map
    for layer_1 in data:
        for layer_2 in layer_1:
            for layer_3 in layer_2:
                for layer_4 in layer_3:
                    return len(layer_4)

# feature_map_save私有调用
def feature_map_save_divided(filename, data):
    feature_map_num = compute_map_num(data)
    for i in range(feature_map_num):
        curr_folder = folder_divided + filename + "/"
        if not os.path.exists(curr_folder):
            os.mkdir(curr_folder)

        name = curr_folder + filename + "_" + str(i)
        file = open(name, 'w+')  # 可读可写可新建，覆盖方式
        for layer_1 in data:
            file.write("[\n")
            for layer_2 in layer_1:
                file.write("\t[\n")
                for layer_3 in layer_2:
                    file.write("\t\t[")
                    for layer_4 in layer_3:
                        item = layer_4[i]
                        s = '{:<15}'.format(str(item))
                        s = "[ " + s + " ], "
                        file.write(s)
                    file.write("],\n")
                file.write("\t]")
            file.write("\n]")
        file.close()

    print("feature_map_save_divided - 保存文件成功")

# 存储Input、每层的conv_result，after_relu，pool值
def feature_map_save(filename, data, divided_flag):
    mkdir_if_not_exit()
    file = open(folder + filename, 'w+')  # 可读可写可新建，覆盖方式
    for layer_1 in data:
        file.write("[\n")
        for layer_2 in layer_1:
            file.write("\t[\n")
            for layer_3 in layer_2:
                file.write("\t\t[")
                for layer_4 in layer_3:
                    s = ",\t".join('{:<15}'.format(str(i)) for i in layer_4)
                    s = "[ " + s + " ],  "
                    file.write(s)
                file.write("],\n")
            file.write("\t]")
        file.write("\n]")

    file.close()
    print("feature_map_save - 保存文件成功")
    # 根据参数决定是否进行分开存储
    if divided_flag:
        feature_map_save_divided(filename, data)


def weight_save_divided(filename, data):
    feature_map_num = compute_map_num(data)

    for i in range(feature_map_num):
        curr_folder = folder_divided + filename + "/"
        if not os.path.exists(curr_folder):
            os.mkdir(curr_folder)

        name = curr_folder + filename + "_" + str(i)
        file = open(name, 'w+')  # 可读可写可新建，覆盖方式
        for layer_1 in data:
            file.write("[\n")
            for layer_2 in layer_1:
                file.write("\t[")
                for layer_3 in layer_2:
                    file.write("[")
                    for layer_4 in layer_3:
                        item = layer_4[i]
                        s = '{:<15}'.format(str(item))
                        s = "[ " + s + " ], "
                        file.write(s)
                    file.write("],  ")
                file.write("],\n")
            file.write("]\n")
        file.close()

    print("weight_save_divided - 保存文件成功")


def weight_save(filename, data, divided_flag):
    mkdir_if_not_exit()
    file = open(folder + filename, 'w+')  # 可读可写可新建，覆盖方式
    for layer_1 in data:
        file.write("[\n")
        for layer_2 in layer_1:
            file.write("\t[")
            for layer_3 in layer_2:
                file.write("[")
                for layer_4 in layer_3:
                    # s = ",  ".join(str(i) for i in layer_4)
                    s = ",\t".join('{:<15}'.format(str(i)) for i in layer_4)
                    s = "[ " + s + " ], "
                    file.write(s)
                file.write("],  ")
            file.write("],\n")
        file.write("]\n")
    file.close()
    print("weight_save - 保存文件成功")
    # 开始分开保存feature的参数
    if divided_flag:
        weight_save_divided(filename, data)


def fc_weight_save(filename, data):
    mkdir_if_not_exit()
    file = open(folder + filename, 'w+')  # 可读可写可新建，覆盖方式
    for layer_1 in data:
        file.write("[\n")
        for layer_2 in layer_1:
            # s = ",  ".join(str(i) for i in layer_2)
            s = ",\t".join('{:<15}'.format(str(i)) for i in layer_2)
            s = "\t[ " + s + " ],\n"
            file.write(s)
        file.write("]\n")
    file.close()
    print("fc_weight_save - 保存文件成功")


def fc_after_relu_save(filename, data):
    mkdir_if_not_exit()
    file = open(folder + filename, 'w+')  # 可读可写可新建，覆盖方式
    for layer_1 in data:
        file.write("[\n")
        for layer_2 in layer_1:
            s = ",  ".join(str(i) for i in layer_2)
            s = "\t[ " + s + " ]\n"
            file.write(s)
        file.write("]\n")
    file.close()
    print("fc_after_relu_save - 保存文件成功")


def biases_save(filename, data):
    mkdir_if_not_exit()
    file = open(folder + filename, 'w+')  # 可读可写可新建，覆盖方式
    for item in data:
        s = ",  ".join(str(i) for i in item)
        s = "[ " + s + " ]\n"
        file.write(s)
    file.close()
    print("biases_save - 保存文件成功")


