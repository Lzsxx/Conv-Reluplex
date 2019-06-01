# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import simulation_function as sfc

# layer2_pool_all = rd.read_layer2_pool("layer2_pool")
#
# layer3_conv_biases = rd.read_layer3_conv_biases("layer3_conv_biases")
#
# # ------ change! 第三层有4个Map,所以这里有4个，后面也都要有4个 ------
#
# layer3_conv_weights_0 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_0")
# layer3_conv_weights_1 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_1")
# layer3_conv_weights_2 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_2")
# layer3_conv_weights_3 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_3")
#
#
# # 读取第3层的AE结果，用于存入详细计算等式
# ae_layer3_after_relu_0 = rd.read_ae_layer3_after_relu_divided("ae_layer3_after_relu_0")
# ae_layer3_after_relu_1 = rd.read_ae_layer3_after_relu_divided("ae_layer3_after_relu_1")
# ae_layer3_after_relu_2 = rd.read_ae_layer3_after_relu_divided("ae_layer3_after_relu_2")
# ae_layer3_after_relu_3 = rd.read_ae_layer3_after_relu_divided("ae_layer3_after_relu_3")
#
#
# # 一次性计算第3层，并将计算那结果写入文件
# write_layer3_compute_process_0 = sfc.layer3_all_compute_i(layer2_pool_all, layer3_conv_weights_0, layer3_conv_biases[0], 0, ae_layer3_after_relu_0)
# write_layer3_compute_process_1 = sfc.layer3_all_compute_i(layer2_pool_all, layer3_conv_weights_1, layer3_conv_biases[1], 1, ae_layer3_after_relu_1)
# write_layer3_compute_process_2 = sfc.layer3_all_compute_i(layer2_pool_all, layer3_conv_weights_2, layer3_conv_biases[2], 2, ae_layer3_after_relu_2)
# write_layer3_compute_process_3 = sfc.layer3_all_compute_i(layer2_pool_all, layer3_conv_weights_3, layer3_conv_biases[3], 3, ae_layer3_after_relu_3)
#
#
# file_name_list = []
# file_name_list.append("compute_process_layer3_with_x_ae_0.param")
# file_name_list.append("compute_process_layer3_with_x_ae_1.param")
# file_name_list.append("compute_process_layer3_with_x_ae_2.param")
# file_name_list.append("compute_process_layer3_with_x_ae_3.param")
# sfc.merge_all_file(file_name_list, "compute_process_layer3_with_x_ae_all.param")


def six_step(idx):
    layer2_pool_all = rd.read_layer2_pool("layer2_pool")

    layer3_conv_biases = rd.read_layer3_conv_biases("layer3_conv_biases")

    # ------ change! 第三层有4个Map,所以这里有4个，后面也都要有4个 ------

    layer3_conv_weights_0 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_0")
    layer3_conv_weights_1 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_1")
    layer3_conv_weights_2 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_2")
    layer3_conv_weights_3 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_3")

    # 读取第3层的AE结果，用于存入详细计算等式
    ae_layer3_after_relu_0 = rd.read_ae_layer3_after_relu_divided("ae_layer3_after_relu_0")
    ae_layer3_after_relu_1 = rd.read_ae_layer3_after_relu_divided("ae_layer3_after_relu_1")
    ae_layer3_after_relu_2 = rd.read_ae_layer3_after_relu_divided("ae_layer3_after_relu_2")
    ae_layer3_after_relu_3 = rd.read_ae_layer3_after_relu_divided("ae_layer3_after_relu_3")

    # 一次性计算第3层，并将计算那结果写入文件
    write_layer3_compute_process_0 = sfc.layer3_all_compute_i(layer2_pool_all, layer3_conv_weights_0,
                                                              layer3_conv_biases[0], 0, ae_layer3_after_relu_0)
    write_layer3_compute_process_1 = sfc.layer3_all_compute_i(layer2_pool_all, layer3_conv_weights_1,
                                                              layer3_conv_biases[1], 1, ae_layer3_after_relu_1)
    write_layer3_compute_process_2 = sfc.layer3_all_compute_i(layer2_pool_all, layer3_conv_weights_2,
                                                              layer3_conv_biases[2], 2, ae_layer3_after_relu_2)
    write_layer3_compute_process_3 = sfc.layer3_all_compute_i(layer2_pool_all, layer3_conv_weights_3,
                                                              layer3_conv_biases[3], 3, ae_layer3_after_relu_3)

    file_name_list = []
    file_name_list.append("compute_process_layer3_with_x_ae_0.param")
    file_name_list.append("compute_process_layer3_with_x_ae_1.param")
    file_name_list.append("compute_process_layer3_with_x_ae_2.param")
    file_name_list.append("compute_process_layer3_with_x_ae_3.param")
    sfc.merge_all_file(file_name_list, "compute_process_layer3_with_x_ae_all_idx=" + str(idx) + ".param")