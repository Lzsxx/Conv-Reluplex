# Conv-Reluplex

1) Use ./s0_parameter_all.py to set the parameter of CNN
2) Use ./s1_train_28_minish_one.py to train a CNN
3) Use ./s2_predict_write_28_minish_two.py to predict a sample, choose a correctly classified sample. And extract the weight parameter of CNN to the file ./s2_parameter and ./s2_parameter_divided
4) Use ./s3_trans_main_three.py to transform the weight parameter to the file ./transform_nnet_parameter
5) From the file ./transform_nnet_parameter get the parameter, and encode to Reluplex (https://github.com/guykatzz/ReluplexCav2017) or Leaky-Reluplex (https://github.com/Lzsxx/Leaky-Reluplex), and set corresponding verification parameters, start verification. If verification success and return SAT, we can find adversarial example from the log file. Extract the specific value of the adversarial example from the log file.
6) Use ./s4_main_predict_from_fc_four.py to predict the classification of the adversarial example. Make sure this is an adversarial example with misclassification.
7) Use ./s5_layer2_pool_to_conv_result_five.py to run unpooling algorithm, restore the intermediate adversarial example to the potential rough features. The rough features are stored in the file ./reluplex_to_ae temporarily.
8) Use ./s6_main_write_layer1_compute_process_six.py to extract the weight parameters of convolution layer, which are stored in the file ./conv_network_simulation. And the calculation of convolution layer is transformed into inequality system.
9) Use ./z_pulp_application/s7_one_map_81_400_seven.py to call the tool pulp to solve the inequality group. The solution is stored in the file ./z_pulp_application/s7_ae_txt_result_temp/ae_txt_file.txt
10) Use ./z_pulp_application/s8_print_ae_eight.py to transform the solution to image type, and store in the  folder ./z_pulp_application/s8_ae_img_collection/
