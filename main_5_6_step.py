# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import numpy as np
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
# from mycode.mnist_all_minish_one_map_9_9 import train_28_minish_one as one
# from mycode.mnist_all_minish_one_map_9_9 import predict_write_28_minish_two as two
# from mycode.mnist_all_minish_one_map_9_9.transform_nnet_parameter import trans_main_three as three

from mycode.mnist_all_minish_one_map_9_9 import s5_layer2_pool_to_conv_result_five as five
# from mycode.mnist_all_minish_one_map_9_9.reluplex_to_ae import layer4_pool_to_conv_result_five_old as five
from mycode.mnist_all_minish_one_map_9_9 import s6_main_write_layer1_compute_process_six as six
# from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import main_write_layer3_compute_process_six_old as six


# print("five step")
# five.five_step(p.ae_fc_input)
#
# print("six step")
# six.six_step(0)



list_len = len(p.ae_fc_input_list)
for i in range(list_len):

    ae_fc_input =p.ae_fc_input_list[i]
    ae_fc_input = ae_fc_input[0:p.fc_input]

    five.five_step(np.array(ae_fc_input))
    six.six_step(i)

