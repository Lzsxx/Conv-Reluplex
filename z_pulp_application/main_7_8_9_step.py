# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import numpy as np
import time
import matplotlib.image as mpimg
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
from mycode.mnist_all_minish_one_map_9_9.z_pulp_application import s7_one_map_81_400_seven as seven
from mycode.mnist_all_minish_one_map_9_9.z_pulp_application import s8_print_ae_eight as eight
from mycode.mnist_all_minish_one_map_9_9.z_pulp_application import s9_predict_ae_use_net_nine as nine
from mycode.mnist_all_minish_one_map_9_9.z_pulp_application import parameter as curr_p
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd


target_label = p.original_label
flag = str(target_label) + "to" + str(p.false_label)
skip_num = 0


# log file
log_folder = p.file_base + "z_pulp_application/run_logs/"+ str(target_label) + "/"
ae_img_collection_folder = p.file_base + "/z_pulp_application/s8_ae_img_collection/" + str(target_label) + "/"
def mkdir_if_not_exist():
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    if not os.path.exists(ae_img_collection_folder):
        os.mkdir(ae_img_collection_folder)

mkdir_if_not_exist()

# open log file
log_file = open(p.file_base + "z_pulp_application/run_logs/" + str(target_label) + "/" +"log-" + flag + "-" + p.original_file_name + "-" +str(int(round(time.time() * 1000))) + ".txt", "w")

# at first, predict the original picture
nine.nine_step_use_original(p.original_file, p.test_label, log_file)

for i in range(len(p.ae_fc_input_list) - skip_num):
    ae_index = i

    # if use the instance folder, use this  # change!
    pulp_compute_source_file = curr_p.source_folder + "compute_process_layer1_with_x_ae_all_idx="+ str(ae_index + skip_num) +".param"
    solve_result = seven.seven_step(pulp_compute_source_file, log_file)

    # if rename the folder name, use this  # change!
    # temp_pulp_compute_source_file = p.file_base + "conv_network_simulation/bk_ae_compute_process_delta_0.6_gap_0.5_5to3/compute_process_layer1_with_x_ae_all_idx="+ str(ae_index) +".param"
    # solve_result = seven.seven_step(temp_pulp_compute_source_file, log_file)

    if solve_result:
        eight.eight_step(ae_index + skip_num, ae_img_collection_folder, flag)
        ae_collection_file = ae_img_collection_folder + flag + "_delta_" + p.delta + "_gap_" + p.gap + "_" + p.original_file_name + "_" + str(ae_index) +".png"
        nine.nine_step_use_img(ae_collection_file, p.test_label, log_file, "ae")


log_file.close()


