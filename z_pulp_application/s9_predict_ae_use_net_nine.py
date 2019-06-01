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
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9 import functions as fs



# ---- description ----
# predict the ae img by txt file or png format
#


def nine_step_use_img(img_file, label, log_file, category):

    input_x = mpimg.imread(img_file)
    input_x = input_x.reshape(1, 28, 28, 1)
    with tf.Session() as sess:
        # 载入已有模型
        saver = tf.train.import_meta_graph(p.model + '/model.ckpt.meta')
        saver.restore(sess, p.model + '/model.ckpt')

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")

        # fs.show_image_label(input_x, label)

        # 将输入数据填充进去
        feed_dict = {x: input_x, y_: label}

        # 模型中间层的参数
        # 第一层参数
        layer1_conv_weights = tf.get_collection('layer1_conv_weights')
        layer1_conv_biases = tf.get_collection('layer1_conv_biases')
        layer1_conv_result = tf.get_collection('layer1_conv_result')
        layer1_after_relu = tf.get_collection('layer1_after_relu')

        # 第二层参数
        layer2_pool = tf.get_collection('layer2_pool')

        # # 第三层参数
        # layer3_conv_weights = tf.get_collection('layer3_conv_weights')
        # layer3_conv_biases = tf.get_collection('layer3_conv_biases')
        # layer3_conv_result = tf.get_collection('layer3_conv_result')
        # layer3_after_relu = tf.get_collection('layer3_after_relu')
        #
        # # 第四层参数
        # layer4_pool = tf.get_collection('layer4_pool')

        # 第五层参数
        fc1_weights = tf.get_collection('fc1_weights')
        fc1_biases = tf.get_collection('fc1_biases')
        fc1_after_relu = tf.get_collection('fc1_after_relu')

        # 第六层参数
        fc2_weights = tf.get_collection('fc2_weights')
        fc2_biases = tf.get_collection('fc2_biases')
        fc2_after_relu = tf.get_collection('fc2_after_relu')

        # 第七层参数
        fc3_weights = tf.get_collection('fc3_weights')
        fc3_biases = tf.get_collection('fc3_biases')
        fc3_result = tf.get_collection('fc3_result')

        # 取得结果Tensor
        x = tf.get_collection('x')
        y = tf.get_collection('y')
        y_ = tf.get_collection('y_')
        accuracy = graph.get_tensor_by_name("accuracy:0")  # 准确率
        correct_prediction = graph.get_tensor_by_name("correct_prediction:0")  # 预测是否正确的List，正确为True，错误表示为False
        # cross_entropy = graph.get_tensor_by_name("cross_entropy:0")   # 交叉熵
        # cross_entropy_mean = graph.get_tensor_by_name("cross_entropy_mean:0")   # 交叉熵的平均值
        # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))   # 损失值

        # Add more to the current graph
        # add_on_op = tf.multiply(op_to_restore, 2)

        result = sess.run([layer1_conv_weights, layer1_conv_biases, layer1_conv_result, layer1_after_relu,
                           layer2_pool,
                           # layer3_conv_weights, layer3_conv_biases, layer3_conv_result, layer3_after_relu,
                           # layer4_pool,
                           fc1_weights, fc1_biases, fc1_after_relu,
                           fc2_weights, fc2_biases, fc2_after_relu,
                           fc3_weights, fc3_biases, fc3_result,
                           correct_prediction, accuracy, x, y, y_],  # y_是真实标签，y是预测标签
                          # feed_dict=feed_dict)
                          feed_dict=feed_dict)

        if category == "ae":
            print("\nthe ae from img : \n")
            print("y:", result[17][0])  # this will repeat if predict more than one , so just get the first result
            print("y_:", result[18][0])
            print("max index:", sfc.get_max_index(result[17][0][0]))
            print("correct_prediction:", result[14])
            print("accuracy:", str(result[15]) )

            # --- write to log file -----
            log_file.write("\nthe ae from img :\n")
            log_file.write("\ny:" + str(result[17][0]))
            log_file.write("\ny_:" + str(result[18][0]))
            log_file.write("\nmax index:" + str(sfc.get_max_index(result[17][0][0])))
            log_file.write("\ncorrect_prediction:" + str(result[14]))
            log_file.write("\naccuracy:" + str(result[15]) + "\n")
        elif category == "original":
            print("\nthe original img : \n")
            print("y:", result[17][0])
            print("y_:", result[18][0])
            print("max index:", sfc.get_max_index(result[17][0][0]))
            print("correct_prediction:", result[14])
            print("accuracy:", result[15])

            # --- write to log file -----
            log_file.write("\nthe original img :\n")
            log_file.write("\ny:" + str(result[17][0]))
            log_file.write("\ny_:" + str(result[18][0]))
            log_file.write("\nmax index:" + str(sfc.get_max_index(result[17][0][0])))
            log_file.write("\ncorrect_prediction:" + str(result[14]))
            log_file.write("\naccuracy:" + str(result[15]) + "\n")
        elif category == "changed":
            print("\nthe changed img : \n")
            print("y:", result[17][0])
            print("y_:", result[18][0])
            print("max index:", sfc.get_max_index(result[17][0][0]))
            print("correct_prediction:", result[14])
            print("accuracy:", result[15])

            # --- write to log file -----
            log_file.write("\nthe changed img :\n")
            log_file.write("\ny:" + str(result[17][0]))
            log_file.write("\ny_:" + str(result[18][0]))
            log_file.write("\nmax index:" + str(sfc.get_max_index(result[17][0][0])))
            log_file.write("\ncorrect_prediction:" + str(result[14]))
            log_file.write("\naccuracy:" + str(result[15]) + "\n")


def nine_step_use_original(original_file, original_label, log_file):
    nine_step_use_img(original_file, original_label, log_file, "original")




def nine_step_use_ae_txt(original_label):
    ae_result_folder = curr_p.ae_result_folder
    ae_result_file = "ae_txt_file.txt"

    ae_input_x = rd.read_ae_x(ae_result_folder, ae_result_file)
    ae_input_x = ae_input_x.reshape(1, 28, 28, 1)

    with tf.Session() as sess:
        # 载入已有模型
        saver = tf.train.import_meta_graph(p.model + '/model.ckpt.meta')
        saver.restore(sess, p.model + '/model.ckpt')

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")

        # show_image_label(test_data[10], test_label[10])

        # 将输入数据填充进去
        feed_dict = {x: ae_input_x, y_: original_label}

        # 模型中间层的参数
        # 第一层参数
        layer1_conv_weights = tf.get_collection('layer1_conv_weights')
        layer1_conv_biases = tf.get_collection('layer1_conv_biases')
        layer1_conv_result = tf.get_collection('layer1_conv_result')
        layer1_after_relu = tf.get_collection('layer1_after_relu')

        # 第二层参数
        layer2_pool = tf.get_collection('layer2_pool')

        # # 第三层参数
        # layer3_conv_weights = tf.get_collection('layer3_conv_weights')
        # layer3_conv_biases = tf.get_collection('layer3_conv_biases')
        # layer3_conv_result = tf.get_collection('layer3_conv_result')
        # layer3_after_relu = tf.get_collection('layer3_after_relu')
        #
        # # 第四层参数
        # layer4_pool = tf.get_collection('layer4_pool')

        # 第五层参数
        fc1_weights = tf.get_collection('fc1_weights')
        fc1_biases = tf.get_collection('fc1_biases')
        fc1_after_relu = tf.get_collection('fc1_after_relu')

        # 第六层参数
        fc2_weights = tf.get_collection('fc2_weights')
        fc2_biases = tf.get_collection('fc2_biases')
        fc2_after_relu = tf.get_collection('fc2_after_relu')

        # 第七层参数
        fc3_weights = tf.get_collection('fc3_weights')
        fc3_biases = tf.get_collection('fc3_biases')
        fc3_result = tf.get_collection('fc3_result')

        # 取得结果Tensor
        x = tf.get_collection('x')
        y = tf.get_collection('y')
        y_ = tf.get_collection('y_')
        accuracy = graph.get_tensor_by_name("accuracy:0")  # 准确率
        correct_prediction = graph.get_tensor_by_name("correct_prediction:0")  # 预测是否正确的List，正确为True，错误表示为False
        # cross_entropy = graph.get_tensor_by_name("cross_entropy:0")   # 交叉熵
        # cross_entropy_mean = graph.get_tensor_by_name("cross_entropy_mean:0")   # 交叉熵的平均值
        # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))   # 损失值

        # Add more to the current graph
        # add_on_op = tf.multiply(op_to_restore, 2)

        result = sess.run([layer1_conv_weights, layer1_conv_biases, layer1_conv_result, layer1_after_relu,
                           layer2_pool,
                           # layer3_conv_weights, layer3_conv_biases, layer3_conv_result, layer3_after_relu,
                           # layer4_pool,
                           fc1_weights, fc1_biases, fc1_after_relu,
                           fc2_weights, fc2_biases, fc2_after_relu,
                           fc3_weights, fc3_biases, fc3_result,
                           correct_prediction, accuracy, x, y, y_],  # y_是真实标签，y是预测标签
                           feed_dict=feed_dict)
        print("\nthe ae from txt : \n")
        print("y:", result[17][0])
        print("y_:", result[18][0])
        print("max index:", sfc.get_max_index(result[17][0][0]))
        print("correct_prediction:", result[14])
        print("accuracy:", result[15])

if __name__ == "__main__":
    # ae_collection_file =  p.file_base + "mnist_predict_write_example/5/z_pulp_application/s8_ae_img_collection/5_to_3/ae3_0.6_0.01_original.png"
    ae_collection_file =  curr_p.ae_collection_folder + "adversarial_example_0.png"
    test_label = np.array([6])  # change!
    log_file = open(p.file_base + "z_pulp_application/run_logs/log-" + str(int(round(time.time() * 1000))) + ".txt",
                    "w")

    # nine_step_use_img(ae_collection_file, test_label, log_file, "ae")
    nine_step_use_img(curr_p.ae_collection_folder + "adversarial_example_2.png", test_label, log_file, "ae")
    nine_step_use_img(curr_p.ae_collection_folder + "adversarial_example_6.png", test_label, log_file, "ae")
    nine_step_use_img(curr_p.ae_collection_folder + "adversarial_example_11.png", test_label, log_file, "ae")
    nine_step_use_img(curr_p.ae_collection_folder + "adversarial_example_12.png", test_label, log_file, "ae")
    nine_step_use_img(curr_p.ae_collection_folder + "adversarial_example_13.png", test_label, log_file, "ae")
    nine_step_use_img(curr_p.ae_collection_folder + "adversarial_example_14.png", test_label, log_file, "ae")
    nine_step_use_img(curr_p.ae_collection_folder + "adversarial_example_15.png", test_label, log_file, "ae")


    log_file.close()
