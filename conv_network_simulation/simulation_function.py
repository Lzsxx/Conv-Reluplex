# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p



# 第一层：卷积计算
def layer1_conv_compute(x, layer1_conv_weights_divided):

    # 输入层：28*28*1，卷积核：5*5*1
    x = x.reshape(p.w, p.h) # 28, 28
    layer1_conv_weights_divided = layer1_conv_weights_divided.reshape(p.layer1_conv_size * p.layer1_conv_size) # 9 * 9 = 81

    # 每次移动步长为1
    row_start_base = 0
    row_end_base = p.layer1_conv_size # 5
    col_start_base = 0
    col_end_base = p.layer1_conv_size
    length = p.w - p.layer1_conv_size + 1     # 20

    # 存储运算结果的list
    conv_result = []

    for i in range(length): # 0 - 20
        # 向下移动
        row_start = row_start_base + i
        row_end = row_end_base + i
        rows = x[row_start: row_end]

        conv_result_row = []     # 存储计算结果的一行

        for j in range(length):     # 0 - 20
            # 向右移动，提取本次要计算的卷积内容
            col_start = col_start_base + j
            col_end = col_end_base + j
            input_lst = []  # 存储本此计算所需要截取的元素，也就是一个田字格的元素

            for row in rows:
                arr = row[col_start: col_end]
                input_lst.append(arr)

            # 提取了5*5的方块以后，要和卷积核计算
            input_arr = np.reshape(np.array(input_lst), p.layer1_conv_size * p.layer1_conv_size)
            sum_value = sum(input_arr * layer1_conv_weights_divided)
            conv_result_row.append(sum_value)

        conv_result.append(conv_result_row)  # 将计算结果的一行存入总的结果数组

    conv_result = np.reshape(np.array(conv_result), (p.layer1_conv_result_size, p.layer1_conv_result_size, 1))
    return conv_result


# 第一层：biases和relu计算
def layer1_biased_relu_compute(conv_result, biases):
    biases_result = conv_result + biases
    biases_result = biases_result.reshape(p.layer1_conv_result_size * p.layer1_conv_result_size)
    for idx in range(len(biases_result)):
        if biases_result[idx] < 0:
            biases_result[idx] = 0

    relu_result = biases_result.reshape(p.layer1_conv_result_size, p.layer1_conv_result_size, 1)
    return relu_result


# 第二层：池化计算
def layer2_max_pool_compute(relu_result):

    # 输入层：20*20，核：2*2
    relu_result = relu_result.reshape(p.layer1_conv_result_size, p.layer1_conv_result_size)

    # 每次移动步长为2
    row_start_base = 0
    row_end_base = 2
    col_start_base = 0
    col_end_base = 2
    length = int(p.layer1_conv_result_size / 2)     # 计算完后一行是10个元素

    # 存储运算结果的list
    conv_result = []

    # 竖向移动
    for i in range(length):  # 0 - 11
        row_start = row_start_base + 2 * i      # 步长为2，所以乘以2
        row_end = row_end_base + 2 * i
        rows = relu_result[row_start: row_end]

        conv_result_row = []     # 存储计算结果的一行

        # 横向移动
        for j in range(length):     # 0 - 11
            col_start = col_start_base + 2 * j
            col_end = col_end_base + 2 * j
            input_lst = []  # 存储本此计算所需要截取的元素，也就是一个田字格的元素

            # 提取要进行池化的2*2方块
            for row in rows:
                arr = row[col_start: col_end]
                input_lst.append(arr)

            # 提取了2*2的方块以后，要进行max_pool计算
            input_arr = np.reshape(np.array(input_lst), 4)
            max_value = input_arr[0]
            for item in input_arr:
                if item > max_value:
                    max_value = item
            conv_result_row.append(max_value)

        # 将计算结果的一行存入总的结果数组
        conv_result.append(conv_result_row)

    conv_result = np.reshape(np.array(conv_result), (p.layer2_pool_result_size, p.layer2_pool_result_size, 1))
    return conv_result


# 第二次池化前，合并前面所有的array
def layer2_merge_divided_pool(pool_0, pool_1, pool_2):
    # 合并成12*12*6的shape
    pool_0 = pool_0.reshape(p.layer2_pool_result_size*p.layer2_pool_result_size)
    pool_1 = pool_1.reshape(p.layer2_pool_result_size*p.layer2_pool_result_size)
    pool_2 = pool_2.reshape(p.layer2_pool_result_size*p.layer2_pool_result_size)

    lst = []
    for i in range(p.layer2_pool_result_size * p.layer2_pool_result_size):
        lst.append(pool_0[i])
        lst.append(pool_1[i])
        lst.append(pool_2[i])

    pool_all = np.reshape(np.array(lst), (p.layer2_pool_result_size, p.layer2_pool_result_size, p.layer1_conv_amount))
    return pool_all


# 第三层：卷积计算
def layer3_conv_compute(input, layer3_conv_weights_divided):

    # 输入层：10*10*3
    input = input.reshape(p.layer2_pool_result_size, p.layer2_pool_result_size, p.layer1_conv_amount)
    # 卷积核：5*5*3
    all_conv_compute_size = p.layer3_conv_size * p.layer3_conv_size * p.layer1_conv_amount
    layer3_conv_weights_divided = layer3_conv_weights_divided.reshape(all_conv_compute_size)

    # 每次移动步长为1
    row_start_base = 0
    row_end_base = p.layer3_conv_size
    col_start_base = 0
    col_end_base = p.layer3_conv_size
    length = p.layer2_pool_result_size - p.layer3_conv_size + 1     # 10 - 5 + 1 = 6

    # 存储运算结果的list
    conv_result = []

    for i in range(length):  # 0 - 5
        # 向下移动
        row_start = row_start_base + i
        row_end = row_end_base + i
        rows = input[row_start: row_end]

        conv_result_row = []     # 存储计算结果的一行

        for j in range(length):     # 0 - 5
            # 向右移动，提取本次要计算的卷积内容
            col_start = col_start_base + j
            col_end = col_end_base + j
            input_lst = []  # 存储本此计算所需要截取的元素，也就是一个田字格的元素

            for row in rows:
                arr = row[col_start: col_end]
                input_lst.append(arr)

            # 提取了5*5*6的方块以后，要和卷积核计算
            temp = np.array(input_lst)
            input_arr = np.reshape(np.array(input_lst), all_conv_compute_size)
            sum_value = sum(input_arr * layer3_conv_weights_divided)
            conv_result_row.append(sum_value)

        conv_result.append(conv_result_row)  # 将计算结果的一行存入总的结果数组

    conv_result = np.reshape(np.array(conv_result), (p.layer3_conv_result_size, p.layer3_conv_result_size, 1))
    return conv_result


# 第三层：biases和relu计算
def layer3_biased_relu_compute(conv_result, biases):
    biases_result = conv_result + biases
    biases_result = biases_result.reshape(p.layer3_conv_result_size * p.layer3_conv_result_size)  # 6,6
    for idx in range(len(biases_result)):
        if biases_result[idx] < 0:
            biases_result[idx] = 0

    relu_result = biases_result.reshape(p.layer3_conv_result_size, p.layer3_conv_result_size, 1)  # 6, 6
    return relu_result


# 第四层：池化计算
def layer4_max_pool_compute(relu_result):

    # 输入层：6*6，核：2*2
    relu_result = relu_result.reshape(p.layer3_conv_result_size, p.layer3_conv_result_size)

    # 每次移动步长为2
    row_start_base = 0
    row_end_base = 2
    col_start_base = 0
    col_end_base = 2
    # 计算完后一行是3个元素
    length = int(p.layer3_conv_result_size / 2)

    # 存储运算结果的list
    conv_result = []

    # 竖向移动
    for i in range(length):  # 0 - 2
        row_start = row_start_base + 2 * i      # 步长为2，所以乘以2
        row_end = row_end_base + 2 * i
        rows = relu_result[row_start: row_end]

        conv_result_row = []     # 存储计算结果的一行

        # 横向移动
        for j in range(length):     # 0 - 2
            col_start = col_start_base + 2 * j
            col_end = col_end_base + 2 * j
            input_lst = []  # 存储本此计算所需要截取的元素，也就是一个田字格的元素

            # 提取要进行池化的2*2方块
            for row in rows:
                arr = row[col_start: col_end]
                input_lst.append(arr)

            # 提取了2*2的方块以后，要进行max_pool计算
            input_arr = np.reshape(np.array(input_lst), 4)
            max_value = input_arr[0]
            for item in input_arr:
                if item > max_value:
                    max_value = item
            conv_result_row.append(max_value)

        # 将计算结果的一行存入总的结果数组
        conv_result.append(conv_result_row)

    conv_result = np.reshape(np.array(conv_result), (p.layer4_pool_result_size, p.layer4_pool_result_size, 1))
    return conv_result


# 全连接层前，合并前面所有的array
def layer4_merge_divided_pool(pool_0, pool_1, pool_2):
    # 转成3*3的shape
    shp = p.layer4_pool_result_size * p.layer4_pool_result_size
    pool_0 = pool_0.reshape(shp)
    pool_1 = pool_1.reshape(shp)
    pool_2 = pool_2.reshape(shp)

    # 合并成3*3*3的shape
    lst = []
    for i in range(shp):
        lst.append(pool_0[i])
        lst.append(pool_1[i])
        lst.append(pool_2[i])

    pool_all = np.reshape(np.array(lst), (p.layer4_pool_result_size, p.layer4_pool_result_size, p.layer3_conv_amount))
    return pool_all


# 第五层：全连接层
def fc1_multiply_biases_relu(input, weight, biases):
    input = input.reshape(p.fc_input)

    result = []
    for i in range(p.fc1_amount):    # 遍历列 48
        arr_row = []
        for j in range(p.fc_input):    # 遍历行 64
            arr_row.append(weight[j][i])
        # 取得了一列，开始用于计算
        col = np.reshape(np.array(arr_row), p.fc_input) # 64
        # 相乘相加
        multiply_sum = sum(col * input) + biases[i]
        # Relu计算
        if multiply_sum < 0:
            multiply_sum = 0
        result.append(multiply_sum)

    result = np.reshape(np.array(result), p.fc1_amount) # 48
    return result


# 第六层：全连接层
def fc2_multiply_biases_relu(input, weight, biases):
    input = input.reshape(p.fc1_amount) # 48

    result = []
    for i in range(p.fc2_amount):    # 遍历列 24
        arr_row = []
        for j in range(p.fc1_amount):    # 遍历行 48
            arr_row.append(weight[j][i])
        # 取得了一列，开始用于计算
        col = np.reshape(np.array(arr_row), p.fc1_amount) # 48
        # 相乘相加
        multiply_sum = sum(col * input) + biases[i]
        # Relu计算
        if multiply_sum < 0:
            multiply_sum = 0
        result.append(multiply_sum)

    result = np.reshape(np.array(result), p.fc2_amount) # 24
    return result

# 第七层：全连接层
def fc3_multiply_biases_relu(input, weight, biases):
    input = input.reshape(p.fc2_amount) # 24

    result = []
    for i in range(p.fc3_amount):    # 遍历列 10
        arr_row = []
        for j in range(p.fc2_amount):    # 遍历行 24
            arr_row.append(weight[j][i])
        # 取得了一列，开始用于计算
        col = np.reshape(np.array(arr_row), p.fc2_amount) # 24
        # 相乘相加
        multiply_sum = sum(col * input) + biases[i]
        # Relu计算，第三层不需要
        # if multiply_sum < 0:
        #     multiply_sum = 0
        result.append(multiply_sum)

    result = np.reshape(np.array(result), p.fc3_amount) # 10
    return result

# 预测
def get_max_index(result):
    max = result[0]
    index = 0
    for i in range(len(result)):
        if(result[i] > max):
            max = result[i]
            index = i

    return index

# ----------- 分析计算过程并写入文件 ----------

folder = p.file_base + "/conv_network_simulation/s6_compute_process_temp/"
floder_with_x = p.file_base + "/conv_network_simulation/s6_compute_process_with_x_temp/"
floder_with_x_ae = p.file_base + "/conv_network_simulation/s6_compute_process_replace_result_with_ae/"

def mkdir_if_not_exit():
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(floder_with_x):
        os.mkdir(floder_with_x)
    if not os.path.exists(floder_with_x_ae):
        os.mkdir(floder_with_x_ae)

# 第1层：卷积 + biases + relu 计算
def layer1_all_compute_i(input, layer1_conv_weights_divided, biases, i, ae_result):  # i用于表示这是第几个feature map，方便存储

    # 为自己的写入做准备
    mkdir_if_not_exit()
    file = open(folder + "compute_process_layer1_" + str(i), 'w+')  # 可读可写可新建，覆盖方式
    file_wite_x = open(floder_with_x + "compute_process_layer1_with_x_" + str(i), 'w+')  # 可读可写可新建，覆盖方式
    file_wite_x_ae = open(floder_with_x_ae + "compute_process_layer1_with_x_ae_" + str(i) + ".param",
                          'w+')  # 可读可写可新建，覆盖方式

    ae_result = ae_result.reshape(p.layer1_conv_result_size, p.layer1_conv_result_size) # 6,6
    # 输入层：28*28*1
    input = input.reshape(p.w, p.h, p.c)
    # 每次计算参数的个数 9 * 9 * 1
    param_length = p.layer1_conv_size * p.layer1_conv_size *  p.c
    # 卷积核：5*5*3
    layer1_conv_weights_divided = layer1_conv_weights_divided.reshape(p.layer1_conv_size, p.layer1_conv_size, p.c) # 9, 9, 1

    # 每次移动步长为1
    row_start_base = 0
    row_end_base = p.layer1_conv_size # 9
    col_start_base = 0
    col_end_base = p.layer1_conv_size # 9
    length = p.w - p.layer1_conv_size + 1  # 20

    # 存储运算结果的list
    conv_result = []

    for i in range(length):  # 0 - 19
        # 向下移动
        row_start = row_start_base + i
        row_end = row_end_base + i
        rows = input[row_start: row_end]

        conv_result_row = []  # 存储计算结果的一行

        for j in range(length):  # 0 - 19
            # 向右移动，提取本次要计算的卷积内容
            col_start = col_start_base + j
            col_end = col_end_base + j
            # input_lst = []  # 存储本此计算所需要截取的元素，也就是一个田字格的元素

            # for row in rows:
            #     arr = row[col_start: col_end]
            #     input_lst.append(arr)

            # 提取了5*5*6的方块以后，要和卷积核计算
            # temp = np.array(input_lst)
            # input_arr = np.reshape(np.array(input_lst), (5, 5, 6))

            # 手动计算代替sum函数：sum_value = sum(input_arr * layer3_conv_weights_divided)
            sum_value = 0
            relu_value = 0
            relu_result_str = ""
            relu_result_str_with_x = ""
            relu_result_str_with_x_ae = ""
            # 记录每一个相乘相加的细节部分
            for k in range(p.layer1_conv_size): # 9
                for m in range(p.layer1_conv_size): # 9
                    for n in range(p.layer1_conv_amount): # 3
                        sum_value = sum_value + input[row_start + k][col_start + m][n] * layer1_conv_weights_divided[k][m][n]
                        relu_result_str = relu_result_str + '{:<16}'.format(str(input[row_start + k][col_start + m][n])) + "*  " + '{:<16}'.format(str(layer1_conv_weights_divided[k][m][n])) + "+  "
                        relu_result_str_with_x = relu_result_str_with_x + '{:<10}'.format("x_" + str(row_start + k) + "_" + str(col_start + m) + "_" + str(n)) + "   *   " + '{:<16}'.format(str(layer1_conv_weights_divided[k][m][n])) + "+  "
                        relu_result_str_with_x_ae = relu_result_str_with_x_ae + "x_" + str(row_start + k) + "_" + str(col_start + m) + "_" + str(n) + "*" + str(layer1_conv_weights_divided[k][m][n]) + "+"

            sum_value = sum_value + biases
            # 进行relu操作
            if sum_value < 0:
                relu_value = 0
            elif sum_value >= 0:
                relu_value = sum_value

            relu_result_str = relu_result_str + '{:<16}'.format(str(biases)) + "= " + '{:<16}'.format(
                str(sum_value)) + " + Relu = " + '{:<16}'.format(str(relu_value))
            relu_result_str_with_x = relu_result_str_with_x + '{:<16}'.format(str(biases)) + "= " + '{:<16}'.format(
                str(sum_value)) + " + Relu = " + '{:<16}'.format(str(relu_value))
            relu_result_str_with_x_ae = relu_result_str_with_x_ae + str(biases) + "=" + str(ae_result[i][j])

            file.write(relu_result_str + "\n")
            file_wite_x.write(relu_result_str_with_x + "\n")
            file_wite_x_ae.write(relu_result_str_with_x_ae + "\n")

            conv_result_row.append(relu_value)

        conv_result.append(conv_result_row)  # 将计算结果的一行存入总的结果数组

    conv_result = np.reshape(np.array(conv_result), (p.layer1_conv_result_size, p.layer1_conv_result_size, 1))
    file.close()
    file_wite_x.close()
    file_wite_x_ae.close()
    return conv_result


# 第3层：卷积 + biases + relu 计算
def layer3_all_compute_i(input, layer3_conv_weights_divided, biases, i, ae_result):  # i用于表示这是第几个feature map，方便存储

    # 为自己的写入做准备
    mkdir_if_not_exit()
    file = open(folder + "compute_process_layer3_" + str(i), 'w+')  # 可读可写可新建，覆盖方式
    file_wite_x = open(floder_with_x + "compute_process_layer3_with_x_" + str(i), 'w+')  # 可读可写可新建，覆盖方式
    file_wite_x_ae = open(floder_with_x_ae + "compute_process_layer3_with_x_ae_" + str(i) + ".param",
                          'w+')  # 可读可写可新建，覆盖方式

    ae_result = ae_result.reshape(p.layer3_conv_result_size, p.layer3_conv_result_size) # 6,6
    # 输入层：10*10*3
    input = input.reshape(p.layer2_pool_result_size, p.layer2_pool_result_size, p.layer1_conv_amount)
    # 每次计算参数的个数
    param_length = p.layer3_conv_size * p.layer3_conv_size *  p.layer1_conv_amount
    # 卷积核：5*5*3
    layer3_conv_weights_divided = layer3_conv_weights_divided.reshape(p.layer3_conv_size, p.layer3_conv_size, p.layer1_conv_amount) # 5, 5, 3

    # 每次移动步长为1
    row_start_base = 0
    row_end_base = p.layer3_conv_size # 5
    col_start_base = 0
    col_end_base = p.layer3_conv_size # 5
    length = p.layer2_pool_result_size - p.layer3_conv_size + 1  # 6

    # 存储运算结果的list
    conv_result = []

    for i in range(length):  # 0 - 5
        # 向下移动
        row_start = row_start_base + i
        row_end = row_end_base + i
        rows = input[row_start: row_end]

        conv_result_row = []  # 存储计算结果的一行

        for j in range(length):  # 0 - 5
            # 向右移动，提取本次要计算的卷积内容
            col_start = col_start_base + j
            col_end = col_end_base + j
            # input_lst = []  # 存储本此计算所需要截取的元素，也就是一个田字格的元素

            # for row in rows:
            #     arr = row[col_start: col_end]
            #     input_lst.append(arr)

            # 提取了5*5*6的方块以后，要和卷积核计算
            # temp = np.array(input_lst)
            # input_arr = np.reshape(np.array(input_lst), (5, 5, 6))

            # 手动计算代替sum函数：sum_value = sum(input_arr * layer3_conv_weights_divided)
            sum_value = 0
            relu_value = 0
            relu_result_str = ""
            relu_result_str_with_x = ""
            relu_result_str_with_x_ae = ""
            # 记录每一个相乘相加的细节部分
            for k in range(p.layer3_conv_size): # 5
                for m in range(p.layer3_conv_size): # 5
                    for n in range(p.layer1_conv_amount): # 3
                        sum_value = sum_value + input[row_start + k][col_start + m][n] * layer3_conv_weights_divided[k][m][n]
                        relu_result_str = relu_result_str + '{:<16}'.format(str(input[row_start + k][col_start + m][n])) + "*  " + '{:<16}'.format(str(layer3_conv_weights_divided[k][m][n])) + "+  "
                        relu_result_str_with_x = relu_result_str_with_x + '{:<10}'.format("x_" + str(row_start + k) + "_" + str(col_start + m) + "_" + str(n)) + "   *   " + '{:<16}'.format(str(layer3_conv_weights_divided[k][m][n])) + "+  "
                        relu_result_str_with_x_ae = relu_result_str_with_x_ae + "x_" + str(row_start + k) + "_" + str(col_start + m) + "_" + str(n) + "*" + str(layer3_conv_weights_divided[k][m][n]) + "+"

            sum_value = sum_value + biases
            # 进行relu操作
            if sum_value < 0:
                relu_value = 0
            elif sum_value >= 0:
                relu_value = sum_value

            relu_result_str = relu_result_str + '{:<16}'.format(str(biases)) + "= " + '{:<16}'.format(
                str(sum_value)) + " + Relu = " + '{:<16}'.format(str(relu_value))
            relu_result_str_with_x = relu_result_str_with_x + '{:<16}'.format(str(biases)) + "= " + '{:<16}'.format(
                str(sum_value)) + " + Relu = " + '{:<16}'.format(str(relu_value))
            relu_result_str_with_x_ae = relu_result_str_with_x_ae + str(biases) + "=" + str(ae_result[i][j])

            file.write(relu_result_str + "\n")
            file_wite_x.write(relu_result_str_with_x + "\n")
            file_wite_x_ae.write(relu_result_str_with_x_ae + "\n")

            conv_result_row.append(relu_value)

        conv_result.append(conv_result_row)  # 将计算结果的一行存入总的结果数组

    conv_result = np.reshape(np.array(conv_result), (p.layer3_conv_result_size, p.layer3_conv_result_size, 1))
    file.close()
    file_wite_x.close()
    file_wite_x_ae.close()
    return conv_result

def merge_all_file(file_name_list, targetName):
    file_all = open(floder_with_x_ae + targetName, 'w+')  # 可读可写可新建，覆盖方式
    for name in file_name_list:
        file = open(floder_with_x_ae + name, 'r')
        line = file.readline()
        while line:
            file_all.write(line)
            line = file.readline()
        file.close()
    file_all.close()
