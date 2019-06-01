# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import simulation_function as sfc
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p


# ------- 读取输入值input -----------
x = rd.read_x("x")

# ------- 读取所有参数 -----------
# layer1_conv_weights = rd.read_layer1_conv_weight("layer1_conv_weights")
# layer1_conv_result = rd.read_layer1_conv_result("layer1_conv_result")
layer1_conv_biases = rd.read_layer1_conv_biases("layer1_conv_biases")
# layer1_after_relu = rd.read_layer1_after_relu("layer1_after_relu")
# layer2_pool = rd.read_layer2_pool("layer2_pool")
# layer3_conv_weights = rd.read_layer3_conv_weight("layer3_conv_weights")
# layer3_conv_result = rd.read_layer3_conv_result("layer3_conv_result")
layer3_conv_biases = rd.read_layer3_conv_biases("layer3_conv_biases")
# layer3_after_relu = rd.read_layer3_after_relu("layer3_after_relu")
layer4_pool = rd.read_layer4_pool("layer4_pool")


fc1_weights = rd.read_fc1_weights("fc1_weights")
fc1_biases = rd.read_fc1_biases("fc1_biases")
fc2_weights = rd.read_fc2_weights("fc2_weights")
fc2_biases = rd.read_fc2_biases("fc2_biases")
fc3_weights = rd.read_fc3_weights("fc3_weights")
fc3_biases = rd.read_fc3_biases("fc3_biases")

# ------- 读取单独某个特征map -----------
layer1_conv_weights_0 = rd.read_layer1_conv_weight_divided("layer1_conv_weights_0")
layer1_conv_weights_1 = rd.read_layer1_conv_weight_divided("layer1_conv_weights_1")
layer1_conv_weights_2 = rd.read_layer1_conv_weight_divided("layer1_conv_weights_2")

layer3_conv_weights_0 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_0")
layer3_conv_weights_1 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_1")
layer3_conv_weights_2 = rd.read_layer3_conv_weight_divided("layer3_conv_weights_2")


#  第一层：卷积计算
layer1_conv_result_0 = sfc.layer1_conv_compute(x, layer1_conv_weights_0)
layer1_conv_result_1 = sfc.layer1_conv_compute(x, layer1_conv_weights_1)
layer1_conv_result_2 = sfc.layer1_conv_compute(x, layer1_conv_weights_2)

# 第一层：biases和relu计算
layer1_after_relu_0 = sfc.layer1_biased_relu_compute(layer1_conv_result_0, layer1_conv_biases[0])
layer1_after_relu_1 = sfc.layer1_biased_relu_compute(layer1_conv_result_1, layer1_conv_biases[1])
layer1_after_relu_2 = sfc.layer1_biased_relu_compute(layer1_conv_result_2, layer1_conv_biases[2])

# 第二层：池化计算
layer2_pool_0 = sfc.layer2_max_pool_compute(layer1_after_relu_0)
layer2_pool_1 = sfc.layer2_max_pool_compute(layer1_after_relu_1)
layer2_pool_2 = sfc.layer2_max_pool_compute(layer1_after_relu_2)

# 由于第一次卷积后产生了3个map，相当于3个通道，这里要将前面分开的feature map合并起来，才能进行计算
layer2_pool_all = sfc.layer2_merge_divided_pool(layer2_pool_0, layer2_pool_1, layer2_pool_2)

# 第三层：卷积计算
layer3_conv_result_0 = sfc.layer3_conv_compute(layer2_pool_all, layer3_conv_weights_0)
layer3_conv_result_1 = sfc.layer3_conv_compute(layer2_pool_all, layer3_conv_weights_1)
layer3_conv_result_2 = sfc.layer3_conv_compute(layer2_pool_all, layer3_conv_weights_2)

# 第三层：biases和relu计算
layer3_after_relu_0 = sfc.layer3_biased_relu_compute(layer3_conv_result_0, layer3_conv_biases[0])
layer3_after_relu_1 = sfc.layer3_biased_relu_compute(layer3_conv_result_1, layer3_conv_biases[1])
layer3_after_relu_2 = sfc.layer3_biased_relu_compute(layer3_conv_result_2, layer3_conv_biases[2])


# 第四层：池化计算
layer4_pool_0 = sfc.layer4_max_pool_compute(layer3_after_relu_0)
layer4_pool_1 = sfc.layer4_max_pool_compute(layer3_after_relu_1)
layer4_pool_2 = sfc.layer4_max_pool_compute(layer3_after_relu_2)

# 第五层前合并所有pool
layer4_pool_all = sfc.layer4_merge_divided_pool(layer4_pool_0, layer4_pool_1, layer4_pool_2)

# temp = layer4_pool_all.reshape(p.fc_input)
# print("layer4_pool:")
# for item in temp:
#     print(str(item) + ",")

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