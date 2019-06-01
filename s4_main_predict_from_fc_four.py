# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import simulation_function as mfc
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p



fc1_weights = rd.read_fc1_weights("fc1_weights")
fc1_biases = rd.read_fc1_biases("fc1_biases")
fc2_weights = rd.read_fc2_weights("fc2_weights")
fc2_biases = rd.read_fc2_biases("fc2_biases")
fc3_weights = rd.read_fc3_weights("fc3_weights")
fc3_biases = rd.read_fc3_biases("fc3_biases")

# 原始正确参数，来自原生网络的输出文件
# layer4_pool_all = np.array([0.000000,5.079190,0.165285,4.453590,0.000000,5.979640,3.611440,6.032680,1.721550,1.149610,7.262340,8.199760,2.942560,0.700000,6.051890,8.784380,0.000000,5.132390,2.015710,4.615820,4.031360,1.240003,1.369638,4.812940,0.000000,0.700000,0.571240,7.653942,0.000000,0.700000,0.844290,9.404860,0.700000,4.363070,7.687401,1.418500,3.831680,1.530243,7.909810,1.393540,0.000000,2.168310,4.347360,0.869540,0.000000,0.700000,1.095910,2.240520,1.874480,3.715800,4.131680,5.517060,2.314540,2.744650,2.416730,9.198290,0.000000,2.287520,4.195990,11.246800,0.000000,3.167600,3.039120,9.944660 ])
layer4_pool_all = p.ae_fc_input

# 第五层，全连接层，
fc1_after_relu = mfc.fc1_multiply_biases_relu(layer4_pool_all, fc1_weights, fc1_biases)

# 第六层，全连接层，
fc2_after_relu = mfc.fc2_multiply_biases_relu(fc1_after_relu, fc2_weights, fc2_biases)

# 第七层，全连接层，
fc3_after_relu = mfc.fc3_multiply_biases_relu(fc2_after_relu, fc3_weights, fc3_biases)

# 根据哪个更大，预测label
predict = mfc.get_max_index(fc3_after_relu)

print("\nfc3_after_relu: ", fc3_after_relu)
print("\npredict: ", predict)