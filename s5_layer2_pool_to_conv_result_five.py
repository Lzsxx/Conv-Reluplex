# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import simulation_function as mfc
from mycode.mnist_all_minish_one_map_9_9.reluplex_to_ae import ae_function as aefc
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p



# ae_fc_input = p.ae_fc_input
# ae_layer2_pool_all = ae_fc_input.reshape(p.layer2_pool_result_size, p.layer2_pool_result_size,
#                                          p.layer1_conv_amount)  # 4 * 4 * 4
# aefc.feature_map_save("ae_layer2_pool", ae_layer2_pool_all, True)
#
# # ------ change! 第2层有1个Map,所以这里有1个，后面也都要有1个 ------
#
# # 将反向输入的第2层 pool后的值按照每一层分隔开。
# ae_layer2_pool_0 = aefc.divided_layer2_pool_all(ae_layer2_pool_all, 0)
#
# # 读取原输入在第2层 pool前的值，按每一层分开
# original_layer1_after_relu_0 = rd.read_layer1_after_relu_divided("layer1_after_relu_0")
#
# # ------------ 第2层：输出值到输入值的反推，同时也是第1层的结果after_relu的值 ---------------
#
# # 反向演算，以原有的第三层pool前的结果为模板，推导AE如果经过最小的变动反推到第三层after_relu后的值
# ae_layer1_after_relu_0 = aefc.reverse_layer2_pool_to_layer1_after_relu(ae_layer2_pool_0,
#                                                                        original_layer1_after_relu_0)
#
# # 将计算结果，手动一个个写入文件
# aefc.feature_map_save_divided_i("ae_layer1_after_relu", 0, ae_layer1_after_relu_0)  # 第一个参数为文件夹名，第二个参数指定要存储第几个map的值
#

def five_step(ae_fc_input_from_p):

    ae_fc_input = ae_fc_input_from_p
    ae_layer2_pool_all = ae_fc_input.reshape(p.layer2_pool_result_size, p.layer2_pool_result_size,
                                             p.layer1_conv_amount)  # 4 * 4 * 4
    aefc.feature_map_save("ae_layer2_pool", ae_layer2_pool_all, True)

    # ------ change! 第2层有1个Map,所以这里有1个，后面也都要有1个 ------

    # 将反向输入的第2层 pool后的值按照每一层分隔开。
    ae_layer2_pool_0 = aefc.divided_layer2_pool_all(ae_layer2_pool_all, 0)

    # 读取原输入在第2层 pool前的值，按每一层分开
    original_layer1_after_relu_0 = rd.read_layer1_after_relu_divided("layer1_after_relu_0")

    # ------------ 第2层：输出值到输入值的反推，同时也是第1层的结果after_relu的值 ---------------

    # 反向演算，以原有的第三层pool前的结果为模板，推导AE如果经过最小的变动反推到第三层after_relu后的值
    ae_layer1_after_relu_0 = aefc.reverse_layer2_pool_to_layer1_after_relu(ae_layer2_pool_0,
                                                                           original_layer1_after_relu_0)

    # 将计算结果，手动一个个写入文件
    aefc.feature_map_save_divided_i("ae_layer1_after_relu", 0, ae_layer1_after_relu_0)  # 第一个参数为文件夹名，第二个参数指定要存储第几个map的值
