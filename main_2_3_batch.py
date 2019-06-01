

# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import numpy as np
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
from mycode.mnist_all_minish_one_map_9_9 import s2_predict_write_28_minish_two as two
from mycode.mnist_all_minish_one_map_9_9 import s3_trans_main_three as third
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.transform_nnet_parameter import trans_function as ts



# ------------- divid ae training -----------------

original_label = 4
original_file_name_list = ["02208","02220","02392","02596","02689"
                            ]

# original_label = 1
# original_file_name_list = ["00031", "00040", "00104", "00112", "00184",
#                            "00231", "00276", "00310", "00572", "01200"
#                             ]
original_file_name = ""

for idx in range(len(original_file_name_list)):
    original_file_name = original_file_name_list[idx]
    original_folder = p.file_base + "mnist_predict_write_example/" + str(original_label) +"/"
    original_file = original_folder + original_file_name + ".png"
    path = p.file_base + "transform_nnet_parameter/s3_param_file_" +str(original_label) + "_" + original_file_name + "/"

    two.second_step_special(original_file, original_label)
    third.third_step_special(path)

# ------------- divid end -----------------