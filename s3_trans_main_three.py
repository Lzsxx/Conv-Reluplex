# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.transform_nnet_parameter import trans_function as ts
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p

def third_step_special(speical_name):
    # 读入参数
    # input = rd.read_layer4_pool("layer4_pool")

    # change to one map
    input = rd.read_layer2_pool("layer2_pool")
    # change end

    temp = input.reshape(p.fc_input)
    print("layer4_pool from read file:")
    print(",".join(str(item) for item in temp))  #

    fc1_weights = rd.read_fc1_weights("fc1_weights")
    fc1_biases = rd.read_fc1_biases("fc1_biases")

    fc2_weights = rd.read_fc2_weights("fc2_weights")
    fc2_biases = rd.read_fc2_biases("fc2_biases")

    fc3_weights = rd.read_fc3_weights("fc3_weights")
    fc3_biases = rd.read_fc3_biases("fc3_biases")

    # 参数格式化写入文件
    ts.transform_input_special_name(input, "input", speical_name)

    ts.transform_weight_special_name(fc1_weights, p.fc_input, p.fc1_amount, "layer1_weight", speical_name)
    ts.transform_weight_special_name_line(fc1_weights, p.fc_input, p.fc1_amount, "layer1_weight_line", speical_name)
    ts.transform_biases_special_name(fc1_biases, "layer1_biases", speical_name)

    ts.transform_weight_special_name(fc2_weights, p.fc1_amount, p.fc2_amount, "layer2_weight", speical_name)
    ts.transform_weight_special_name_line(fc2_weights, p.fc1_amount, p.fc2_amount, "layer2_weight_line", speical_name)
    ts.transform_biases_special_name(fc2_biases, "layer2_biases", speical_name)

    ts.transform_weight_special_name(fc3_weights, p.fc2_amount, p.fc3_amount, "layer3_weight", speical_name)
    ts.transform_weight_special_name_line(fc3_weights, p.fc2_amount, p.fc3_amount, "layer3_weight_line", speical_name)
    ts.transform_biases_special_name(fc3_biases, "layer3_biases", speical_name)



if __name__ == "__main__":
    path = p.file_base + "transform_nnet_parameter/s3_param_file_" +str(p.original_label) + "_" + p.original_file_name + "/"
    third_step_special(path)




# def third_step():
#
#     # 读入参数
#     # input = rd.read_layer4_pool("layer4_pool")
#
#     # change to one map
#     input = rd.read_layer2_pool("layer2_pool")
#     # change end
#
#     temp = input.reshape(p.fc_input)
#     print("layer4_pool from read file:")
#     print(",".join(str(item) for item in temp)) #
#
#     fc1_weights = rd.read_fc1_weights("fc1_weights")
#     fc1_biases = rd.read_fc1_biases("fc1_biases")
#
#     fc2_weights = rd.read_fc2_weights("fc2_weights")
#     fc2_biases = rd.read_fc2_biases("fc2_biases")
#
#     fc3_weights = rd.read_fc3_weights("fc3_weights")
#     fc3_biases = rd.read_fc3_biases("fc3_biases")
#
#     # 参数格式化写入文件
#     ts.transform_input(input, "input")
#
#     ts.transform_weight(fc1_weights, p.fc_input, p.fc1_amount, "layer1_weight")
#     ts.transform_biases(fc1_biases, "layer1_biases")
#
#     ts.transform_weight(fc2_weights, p.fc1_amount, p.fc2_amount, "layer2_weight")
#     ts.transform_biases(fc2_biases, "layer2_biases")
#
#     ts.transform_weight(fc3_weights, p.fc2_amount, p.fc3_amount, "layer3_weight")
#     ts.transform_biases(fc3_biases, "layer3_biases")



