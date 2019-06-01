# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p


path = p.file_base + "transform_nnet_parameter/s3_param_file/"


def build_folder():
    if not os.path.exists(path):
        os.mkdir(path)

def build_folder(path_sepical):
    if not os.path.exists(path_sepical):
        os.mkdir(path_sepical)


def transform_input_special_name(weight, name, path_sepical):
    build_folder(path_sepical)
    file = open(path_sepical + name, "w+")  # 设置文件对象
    for layer_1 in weight:
        for layer_2 in layer_1:
            for layer_3 in layer_2:
                s = ",".join(str(i) for i in layer_3)
                s = s + " ,"
                file.write(s)
    file.close()

def transform_input(weight, name):
    build_folder()
    file = open(path + name, "w+")  # 设置文件对象
    for layer_1 in weight:
        for layer_2 in layer_1:
            for layer_3 in layer_2:
                s = ",".join(str(i) for i in layer_3)
                s = s + " ,"
                file.write(s)
    file.close()


def transform_weight_special_name_line(weight, original_row, original_col, name, path_sepical):
    build_folder(path_sepical)
    file = open(path_sepical + name, "w+")  # 设置文件对象
    result = []
    for i in range(original_col):  # 遍历列
        arr_row = []
        for j in range(original_row):  # 遍历行
            arr_row.append(weight[j][i])
        # 取得了一列，加入逗号转化成行，写入文件
        s = ",".join(str(i) for i in arr_row)
        s = s + ","
        file.write(s)
    file.close()
    return result

def transform_weight_special_name(weight, original_row, original_col, name, path_sepical):
    build_folder(path_sepical)
    file = open(path_sepical + name, "w+")  # 设置文件对象
    result = []
    for i in range(original_col):  # 遍历列
        arr_row = []
        for j in range(original_row):  # 遍历行
            arr_row.append(weight[j][i])
        # 取得了一列，加入逗号转化成行，写入文件
        s = ",".join(str(i) for i in arr_row)
        s = s + ",\n"
        file.write(s)
    file.close()
    return result
# 输入权重矩阵，以及原始矩阵的行和列，将其反转
def transform_weight(weight, original_row, original_col, name):
    build_folder()
    file = open(path + name, "w+")  # 设置文件对象
    result = []
    for i in range(original_col):  # 遍历列
        arr_row = []
        for j in range(original_row):  # 遍历行
            arr_row.append(weight[j][i])
        # 取得了一列，加入逗号转化成行，写入文件
        s = ",".join(str(i) for i in arr_row)
        s = s + ",\n"
        file.write(s)
    file.close()
    return result


def transform_biases_special_name(biases, name, path_sepical):
    build_folder(path_sepical)
    file = open(path_sepical + name, "w+")  # 设置文件对象
    for item in biases:
        s = str(item) + ",\n"
        file.write(s)
    file.close()

def transform_biases(biases, name):
    build_folder()
    file = open(path + name, "w+")  # 设置文件对象
    for item in biases:
        s = str(item) + ",\n"
        file.write(s)
    file.close()
