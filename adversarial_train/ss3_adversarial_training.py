# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
import pylab
import time
import tensorflow as tf
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
import matplotlib.image as mpimg



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#mnist数据集中训练数据和测试数据保存地址
train_path = '/home/lzs/Documents/my_image_net/mycode/mnist_all_minish_one_map_9_9/adversarial_train_data_train/'
test_path = "/home/lzs/Documents/my_image_net/mycode/mnist_all_minish_one_map_9_9/adversarial_train_data_test/"

test_original_path = '/home/lzs/Documents/my_image_net/mycode/data_set/mnist_data/test/'

# 读取训练数据及测试数据
train_data,train_label = fs.read_image(train_path)
test_data,test_label = fs.read_image(test_path)
test_original_data,test_original_label = fs.read_image(test_original_path)

# 打乱训练数据及测试数据
train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label = train_label[train_image_index]

test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]


test_original_image_num = len(test_original_data)
test_original_image_index = np.arange(test_original_image_num)
np.random.shuffle(test_original_image_index)
test_original_data = test_original_data[test_original_image_index]
test_original_label = test_original_label[test_original_image_index]

def adversarial_train():
    with tf.Session() as sess:

        # 载入已有模型
        saver = tf.train.import_meta_graph('../model_9_9/model.ckpt.meta')
        saver.restore(sess, '../model_9_9/model.ckpt')

        graph = tf.get_default_graph()

        # x = tf.placeholder(tf.float32, [None, fs.w, fs.h, fs.c], name='x')
        # y_ = tf.placeholder(tf.int32, [None], name='y_')

        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")


        accuracy = graph.get_tensor_by_name("accuracy:0")     # 准确率
        correct_prediction = graph.get_tensor_by_name("correct_prediction:0")   # 预测是否正确的List，正确为True，错误表示为False
        # cross_entropy = graph.get_tensor_by_name("cross_entropy")   # 交叉熵
        cross_entropy_mean = graph.get_tensor_by_name("cross_entropy_mean:0")   # 交叉熵的平均值
        loss = cross_entropy_mean   # 损失值
        # ae_train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
        ae_train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean)
        # tf.train.GradientDescentOptimizer(0.02)



        # ---------- before ae training ---------
        err, acc = sess.run([loss, accuracy], feed_dict={x: test_data, y_: test_label})
        print("***** before ae training *****")
        print("test ae loss:", err)
        print("test ae acc:", acc)

        log_file = open("./logs/" + "ss3_adversarial_training" + str(int(round(time.time() * 1000))) + ".txt", "w")
        log_file.write("\n\n***** before ae training *****")
        log_file.write("\ntest ae loss: " + str(err))
        log_file.write("\ntest ae acc: " + str(acc))


        err, acc = sess.run([loss, accuracy], feed_dict={x: test_original_data, y_: test_original_label})
        print("test original loss:", err)
        print("test original acc:", acc)

        log_file.write("\ntest original loss: " + str(err))
        log_file.write("\ntest original acc: " + str(acc))

        err, acc = sess.run([loss, accuracy], feed_dict={x: train_data, y_: train_label})
        print("test train loss:", err)
        print("test train acc:", acc)

        log_file.write("\ntest train loss: " + str(err))
        log_file.write("\ntest train acc: " + str(acc))


        # ---------- begin ae training ---------
        # 将所有样本训练10次，每次训练中以64个为一组训练完所有样本。
        # train_num可以设置大一些。
        train_num = 10
        batch_size = 16

        print("***** begin ae training *****")
        log_file.write("\n\n***** begin ae training *****")

        arr_train_acc = []
        arr_train_loss = []

        arr_test_adv_acc = []
        arr_test_adv_loss = []

        arr_test_original_acc = []
        arr_test_original_loss = []

        for i in range(train_num):
            # --- train ---------
            train_loss, train_acc, batch_num = 0, 0, 0
            for train_data_batch, train_label_batch in fs.get_batch(train_data, train_label, batch_size):
                _, err, acc = sess.run([ae_train_op, loss, accuracy],
                                       feed_dict={x: train_data_batch, y_: train_label_batch})
                train_loss += err
                train_acc += acc
                batch_num += 1

            print("in turn : ", i)
            print("train loss:", train_loss / batch_num)
            print("train acc:", train_acc / batch_num)

            log_file.write("\nin turn : "+ str(i))
            log_file.write("\ntrain loss:" + str(train_loss / batch_num))
            log_file.write("\ntrain acc:" + str(train_acc / batch_num))

            arr_train_loss.append(train_loss / batch_num)
            arr_train_acc.append(train_acc / batch_num)


            # --- test ae ---------
            test_loss, test_acc, batch_num = 0, 0, 0
            for test_data_batch, test_label_batch in fs.get_batch(test_data, test_label, batch_size):
                err, acc = sess.run([loss, accuracy], feed_dict={x: test_data_batch, y_: test_label_batch})
                test_loss += err
                test_acc += acc
                batch_num += 1

            print("in turn : ", i)
            print("test ae loss:", test_loss / batch_num)
            print("test ae acc:", test_acc / batch_num)

            log_file.write("\nin turn : " + str(i))
            log_file.write("\ntest ae loss:" + str(test_loss / batch_num))
            log_file.write("\ntest ae acc:" + str(test_acc / batch_num))

            arr_test_adv_loss.append(test_loss / batch_num)
            arr_test_adv_acc.append(test_acc / batch_num)


            # --- test original ---------
            test_original_loss, test_original_acc, batch_num = 0, 0, 0
            for test_original_data_batch, test_original_label_batch in fs.get_batch(test_original_data, test_original_label, batch_size):
                err, acc = sess.run([loss, accuracy], feed_dict={x: test_original_data_batch, y_: test_original_label_batch})
                test_original_loss += err
                test_original_acc += acc
                batch_num += 1

            print("in turn : ", i)
            print("test original loss:", test_original_loss / batch_num)
            print("test original acc:", test_original_acc / batch_num)

            log_file.write("\nin turn : " + str(i))
            log_file.write("\ntest original loss:" + str(test_original_loss / batch_num))
            log_file.write("\ntest original acc:" + str(test_original_acc / batch_num))

            arr_test_original_loss.append(test_original_loss / batch_num)
            arr_test_original_acc.append(test_original_acc / batch_num)


        # ---------- after ae training ---------
        err, acc = sess.run([loss, accuracy], feed_dict={x: test_data, y_: test_label})
        print("***** after ae training *****")
        print("test ae loss:", err)
        print("test ae acc:", acc)

        log_file.write("\n\n***** after ae training *****")
        log_file.write("\ntest ae loss: " + str(err))
        log_file.write("\ntest ae acc: " + str(acc))

        err, acc = sess.run([loss, accuracy], feed_dict={x: test_original_data, y_: test_original_label})
        print("test original loss:", err)
        print("test original acc:", acc)

        log_file.write("\ntest original loss: " + str(err))
        log_file.write("\ntest original acc: " + str(acc))



        log_file.write("\n\narr_train_loss:\n" + ",  ".join(str(i) for i in arr_train_loss))
        log_file.write("\n\narr_test_adv_loss:\n" + ",  ".join(str(i) for i in arr_test_adv_loss))
        log_file.write("\n\narr_test_original_loss:\n" + ",  ".join(str(i) for i in arr_test_original_loss))

        log_file.write("\n\narr_train_acc:\n" + ",  ".join(str(i) for i in arr_train_acc))
        log_file.write("\n\narr_test_adv_acc:\n" + ",  ".join(str(i) for i in arr_test_adv_acc))
        log_file.write("\n\narr_test_original_acc:\n" + ",  ".join(str(i) for i in arr_test_original_acc))

        log_file.close()



        # 存储模型所有参数
        folder = "./temp-model/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        saver.save(sess, "./temp-model/model.ckpt")


        # change end
if __name__ == "__main__":
    adversarial_train()