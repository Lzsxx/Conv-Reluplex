# -*- coding: utf-8 -*-

from pulp import *
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
import re


# the file to store the ae paramater before reverse
source_folder = p.file_base + "conv_network_simulation/s6_compute_process_replace_result_with_ae/"

# the file to store the reverse result
ae_result_folder = p.file_base + "z_pulp_application/s7_ae_txt_result_temp/"
ae_result_file = ae_result_folder + "ae_txt_file.txt"

ae_collection_folder = p.file_base + "z_pulp_application/s8_ae_img_collection/"
