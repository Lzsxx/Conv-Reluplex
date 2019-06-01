# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import numpy as np
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import simulation_function as sfc
import matplotlib.image as mpimg

# predict use my simulation network


# original img file
original_input_x = mpimg.imread(p.original_file)

def ten_step():
    folder = p.file_base + "z_pulp_application/ae_result/"
    file = "ae_file.txt"

    input_x = rd.read_ae_x(folder, file)

    layer1_conv_biases = rd.read_layer1_conv_biases("layer1_conv_biases")

    fc1_weights = rd.read_fc1_weights("fc1_weights")
    fc1_biases = rd.read_fc1_biases("fc1_biases")
    fc2_weights = rd.read_fc2_weights("fc2_weights")
    fc2_biases = rd.read_fc2_biases("fc2_biases")
    fc3_weights = rd.read_fc3_weights("fc3_weights")
    fc3_biases = rd.read_fc3_biases("fc3_biases")

    # ------- 读取单独某个特征map -----------
    layer1_conv_weights_0 = rd.read_layer1_conv_weight_divided("layer1_conv_weights_0")

    #  第一层：卷积计算
    layer1_conv_result_0 = sfc.layer1_conv_compute(input_x, layer1_conv_weights_0)

    # 第一层：biases和relu计算
    layer1_after_relu_0 = sfc.layer1_biased_relu_compute(layer1_conv_result_0, layer1_conv_biases[0])

    # 第二层：池化计算
    layer2_pool_0 = sfc.layer2_max_pool_compute(layer1_after_relu_0)

    # 由于第一次卷积后产生了3个map，相当于3个通道，这里要将前面分开的feature map合并起来，才能进行计算
    # layer2_pool_all = sfc.layer2_merge_divided_pool(layer2_pool_0, layer2_pool_1, layer2_pool_2)
    layer2_pool_all = layer2_pool_0

    # 第五层前合并所有pool
    # layer4_pool_all = sfc.layer4_merge_divided_pool(layer4_pool_0, layer4_pool_1, layer4_pool_2)
    layer4_pool_all = layer2_pool_all

    # 第五层，全连接层，
    fc1_after_relu = sfc.fc1_multiply_biases_relu(layer4_pool_all, fc1_weights, fc1_biases)

    # 第六层，全连接层，
    fc2_after_relu = sfc.fc2_multiply_biases_relu(fc1_after_relu, fc2_weights, fc2_biases)

    # 第七层，全连接层，
    fc3_after_relu = sfc.fc3_multiply_biases_relu(fc2_after_relu, fc3_weights, fc3_biases)

    # 根据哪个更大，预测label
    predict = sfc.get_max_index(fc3_after_relu)

    print("\nfc3_after_relu: ", fc3_after_relu)
    print("\npredict: ", predict)
