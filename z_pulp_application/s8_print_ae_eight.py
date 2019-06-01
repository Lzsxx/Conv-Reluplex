# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
import pylab
import re
import tensorflow as tf
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
from mycode.mnist_all_minish_one_map_9_9.z_pulp_application import parameter as curr_p
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd

# ---- description ----
# read the txt file and save it as image
#

def eight_step(ae_index, ae_collection_folder, flag ):
    ae_result_folder = curr_p.ae_result_folder
    ae_result_file = "ae_txt_file.txt"

    input_x = rd.read_x("x")
    ae_input_x = rd.read_ae_x(ae_result_folder, ae_result_file)

    fs.show_image_label(input_x, 5)
    fs.show_image_label(ae_input_x, "ae5")

    save_img = ae_input_x.reshape(p.w, p.h) * 255
    save_img = save_img.astype(np.uint8)

    # ae_index = re.split(r"\.|=", curr_p.source_file)[1]
    io.imsave(ae_collection_folder + flag + "_delta_" + p.delta + "_gap_" + p.gap + "_" + p.original_file_name + "_" + str(ae_index) + ".png", save_img)
