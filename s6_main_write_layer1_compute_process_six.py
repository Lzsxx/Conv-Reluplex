# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import simulation_function as sfc

#
# input = rd.read_x("x")
#
# layer1_conv_biases = rd.read_layer1_conv_biases("layer1_conv_biases")
#
# # ------ change! 第三层有4个Map,所以这里有4个，后面也都要有4个 ------
#
# layer1_conv_weights_0 = rd.read_layer1_conv_weight_divided("layer1_conv_weights_0")
#
# # 读取第3层的AE结果，用于存入详细计算等式
# ae_layer1_after_relu_0 = rd.read_ae_layer1_after_relu_divided("ae_layer1_after_relu_0")
#
# # 一次性计算第3层，并将计算那结果写入文件
# write_layer1_compute_process_0 = sfc.layer1_all_compute_i(input, layer1_conv_weights_0,
#                                                           layer1_conv_biases[0], 0, ae_layer1_after_relu_0)
#
# file_name_list = []
# file_name_list.append("compute_process_layer1_with_x_ae_0.param")
# sfc.merge_all_file(file_name_list, "compute_process_layer1_with_x_ae_all_idx" + ".param")
# sfc.merge_all_file(file_name_list, "compute_process_layer1_with_x_ae_all_idx=" + str(idx) + ".param")


def six_step(idx):

    input = rd.read_x("x")

    layer1_conv_biases = rd.read_layer1_conv_biases("layer1_conv_biases")

    # ------ change! 第三层有4个Map,所以这里有4个，后面也都要有4个 ------

    layer1_conv_weights_0 = rd.read_layer1_conv_weight_divided("layer1_conv_weights_0")

    # 读取第3层的AE结果，用于存入详细计算等式
    ae_layer1_after_relu_0 = rd.read_ae_layer1_after_relu_divided("ae_layer1_after_relu_0")

    # 一次性计算第3层，并将计算那结果写入文件
    write_layer1_compute_process_0 = sfc.layer1_all_compute_i(input, layer1_conv_weights_0,
                                                              layer1_conv_biases[0], 0, ae_layer1_after_relu_0)

    file_name_list = []
    file_name_list.append("compute_process_layer1_with_x_ae_0.param")
    # sfc.merge_all_file(file_name_list, "compute_process_layer1_with_x_ae_all_idx" + ".param")
    sfc.merge_all_file(file_name_list, "compute_process_layer1_with_x_ae_all_idx=" + str(idx) + ".param")