# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p



# ------------- 读取总参数部分 -------------------

# folder = "../bk_parameter/"
# folder_divided = "../bk_parameter_divided/"

folder = p.file_base + "/s2_parameter/"
folder_divided = p.file_base + "/s2_parameter_divided/"
ae_folder = p.file_base + "/reluplex_to_ae/s5_ae_divided_parameter_temp/"

def get_list_from_file(path):
    file = open(path, "r")  # 设置文件对象
    string = file.read()  # 将txt文件的所有内容读入到字符串str中
    file.close()  # 将文件关闭

    string = string.replace("\n", "").replace("\t", "")
    string = string.replace("[", "").replace("]", "").replace(" ", "").replace(",,", ",")
    arr = string.split(",")
    while "" in arr:
        arr.remove("")  # 把数组内的""这玩意清理掉
    return arr

def read_ae_x(myfolder, filename):
    lst = get_list_from_file(myfolder + filename)
    # print("x len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.w, p.h, p.c)   # 28 * 28 * 1
    # fs.show_image_label(x, "5")
    return arr

def read_x(filename):
    lst = get_list_from_file(folder + filename)
    # print("x len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.w, p.h, p.c)   # 28 * 28 * 1
    # fs.show_image_label(x, "5")
    return arr


def read_layer1_conv_weight(filename):
    lst = get_list_from_file(folder + filename)
    print("layer1_conv_weight len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.layer1_conv_size, p.layer1_conv_size,
                                                   p.c, p.layer1_conv_amount)  # 9 * 9 * 1 * 3
    return arr


def read_layer3_conv_weight(filename):
    lst = get_list_from_file(folder + filename)
    print("layer3_conv_weight len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.layer3_conv_size, p.layer3_conv_size,
                                                   p.layer1_conv_amount, p.layer3_conv_amount) # 5, 5, 3, 3
    return arr


def read_layer1_conv_result(filename):
    lst = get_list_from_file(folder + filename)
    print("layer1_conv_result len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(1, p.layer1_conv_result_size, p.layer1_conv_result_size,
                                                   p.layer1_conv_amount) # 1,20,20,3
    return arr


def read_layer3_conv_result(filename):
    lst = get_list_from_file(folder + filename)
    print("layer3_conv_result len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(1, p.layer3_conv_result_size, p.layer3_conv_result_size,
                                                   p.layer3_conv_amount) # 1, 6, 6, 3
    return arr


def read_layer1_conv_biases(filename):
    lst = get_list_from_file(folder + filename)
    print("layer1_conv_biases len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.layer1_conv_amount)    # 3
    return arr


def read_layer3_conv_biases(filename):
    lst = get_list_from_file(folder + filename)
    print("layer3_conv_biases len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.layer3_conv_amount)
    return arr


def read_layer1_after_relu(filename):
    lst = get_list_from_file(folder + filename)
    print("layer1_after_relu len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(1, p.layer1_conv_result_size, p.layer1_conv_result_size,
                                                   p.layer1_conv_amount)  # 1,20,20,3
    return arr


def read_layer3_after_relu(filename):
    lst = get_list_from_file(folder + filename)
    print("layer3_after_relu len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(1, p.layer3_conv_result_size, p.layer3_conv_result_size,
                                                   p.layer3_conv_amount)  # 1, 6, 6, 3
    return arr


def read_layer2_pool(filename):
    lst = get_list_from_file(folder + filename)
    print("layer2_pool len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(1, p.layer2_pool_result_size, p.layer2_pool_result_size,
                                                   p.layer1_conv_amount)  # 1,10,10,3
    return arr


def read_layer4_pool(filename):
    lst = get_list_from_file(folder + filename)
    print("layer4_pool len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(1, p.layer4_pool_result_size, p.layer4_pool_result_size,
                                                      p.layer3_conv_amount)  # 1, 3, 3, 3
    return arr


def read_fc1_weights(filename):
    lst = get_list_from_file(folder + filename)
    print("fc1_weights len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.fc_input, p.fc1_amount)
    return arr


def read_fc2_weights(filename):
    lst = get_list_from_file(folder + filename)
    print("fc2_weights len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.fc1_amount, p.fc2_amount)
    return arr


def read_fc3_weights(filename):
    lst = get_list_from_file(folder + filename)
    print("fc3_weights len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.fc2_amount, p.fc3_amount)
    return arr


def read_fc1_biases(filename):
    lst = get_list_from_file(folder + filename)
    print("fc1_biases len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.fc1_amount)
    return arr


def read_fc2_biases(filename):
    lst = get_list_from_file(folder + filename)
    print("fc2_biases len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.fc2_amount)
    return arr


def read_fc3_biases(filename):
    lst = get_list_from_file(folder + filename)
    print("fc3_biases len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.fc3_amount)
    return arr

# read_x("x")
# read_layer1_conv_weight("layer1_conv_weights")
# read_layer1_conv_result("layer1_conv_result")
# read_layer1_conv_biases("layer1_conv_biases")
# read_layer1_after_relu("layer1_after_relu")
# read_layer2_pool("layer2_pool")
# read_layer3_conv_weight("layer3_conv_weights")
# read_layer3_conv_result("layer3_conv_result")
# read_layer3_conv_biases("layer3_conv_biases")
# read_layer3_after_relu("layer3_after_relu")
# read_layer4_pool("layer4_pool")


# ------- 读取单独某个特征map -----------
def read_layer1_conv_weight_divided(filename):
    lst = get_list_from_file(folder_divided + "layer1_conv_weights/" + filename)
    arr = np.array(lst).astype(np.float64).reshape(p.layer1_conv_size, p.layer1_conv_size, p.c)
    return arr


def read_layer1_after_relu_divided(filename):
    lst = get_list_from_file(folder_divided + "layer1_after_relu/" + filename)
    arr = np.array(lst).astype(np.float64).reshape(p.layer1_conv_result_size, p.layer1_conv_result_size, 1)
    return arr

def read_ae_layer1_after_relu_divided(filename):
    lst = get_list_from_file(ae_folder + "ae_layer1_after_relu/" + filename)
    print("layer1_after_relu len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.layer1_conv_result_size, p.layer1_conv_result_size, 1)  # 6,6,1
    return arr


def read_layer3_conv_weight_divided(filename):
    lst = get_list_from_file(folder_divided + "layer3_conv_weights/" + filename)
    arr = np.array(lst).astype(np.float64).reshape(p.layer3_conv_size, p.layer3_conv_size, p.layer1_conv_amount)     # 5*5*6表示前面的6个特征map
    return arr


def read_layer3_after_relu_divided(filename):
    lst = get_list_from_file(folder_divided + "layer3_after_relu/" + filename)
    arr = np.array(lst).astype(np.float64).reshape(p.layer3_conv_result_size, p.layer3_conv_result_size, 1)  # 6,6,1
    return arr


def read_ae_layer3_after_relu_divided(filename):
    lst = get_list_from_file(ae_folder + "ae_layer3_after_relu/" + filename)
    print("layer3_after_relu len: ", len(lst))
    arr = np.array(lst).astype(np.float64).reshape(p.layer3_conv_result_size, p.layer3_conv_result_size, 1)  # 6,6,1
    return arr
# read_layer1_conv_weight_divided("layer1_conv_weights_0")
# read_layer3_conv_weight_divided("layer3_conv_weights_0")
