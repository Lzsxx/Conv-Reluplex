# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import time
import numpy as np
import pylab
import matplotlib.image as mpimg
import tensorflow as tf
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import simulation_function as sfc
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
from mycode.mnist_all_minish_one_map_9_9.z_pulp_application import parameter as curr_p
from mycode.mnist_all_minish_one_map_9_9.z_pulp_application import s9_predict_ae_use_net_nine as s9
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd



# change picture by code
ae_collection_file = curr_p.ae_collection_folder + "change.png"

ae_input_x = mpimg.imread(ae_collection_file)
ae_one_dim = []
for i in range(28):
    for j in range(28):
        ae_one_dim.append(ae_input_x[i][j][0])

ae_one_dim = np.array(ae_one_dim)
save_img = ae_one_dim.reshape(p.w, p.h) * 255
save_img = save_img.astype(np.uint8)
io.imsave(curr_p.ae_collection_folder + "change_by_code.png", save_img)

# predict the changed picture
log_file = open(p.file_base + "z_pulp_application/run_logs/log-"+str(int(round(time.time() * 1000))) + ".txt", "w")
test_label = np.array([5])

s9.nine_step_use_original(p.original_file, p.test_label, log_file)
s9.nine_step_use_img(curr_p.ae_collection_folder + "ae3_0.6_original.png", p.test_label, log_file, "ae")
s9.nine_step_use_img(curr_p.ae_collection_folder + "change_by_code.png", p.test_label, log_file, "changed")

log_file.close()