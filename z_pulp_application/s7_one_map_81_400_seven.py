# -*- coding: utf-8 -*-

from pulp import *
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
from mycode.mnist_all_minish_one_map_9_9.z_pulp_application import parameter as curr_p
import re

# ---- description ----
# solve the linear equation by pulp, write the parameter result to the txt file
#


# source_folder = p.file_base + "/conv_network_simulation/compute_process_replace_result_with_ae/"
# source_file = source_folder + "compute_process_layer1_with_x_ae_all_idx=29.param"
#
# ae_result_folder = p.file_base + "/z_pulp_application/ae_result/"
# ae_result_file = ae_result_folder + "ae_file.txt"

input_layer_1 = 28  # 输入矩阵的大小，由此计算总的输入节点个数
input_layer_2 = 28
input_layer_3 = 1

name_to_idx_map = {}

# 1. 定义决策变量，并设置变量最小取值
class params_nodes():
    def __init__(self):
        # input variable
        idx = 0
        for j in range(input_layer_1):
            for k in range(input_layer_2):
                for m in range(input_layer_3):
                    name = "x_" + str(j) + "_" + str(k) + "_" + str(m)
                    name_to_idx_map[name] = idx
                    idx = idx + 1

                    # ------- version : 1 --------------

                    # lp_var = LpVariable(name, lowBound=0, upBound=1.0)

                    # ------------ 1 end ---------------


                    # # ------- version : 2 : for number 4 - 00024 --------------
                    # low_bound_value = 0.0
                    # up_bound_value = 0.1
                    # gap_left = 0
                    # gap_right = 0
                    #
                    # if j == 18 and (k >= 16 and k <= 17):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 19 and (k >= 16 and k <= 17):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 20 and (k >= 16 and k <= 17):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 21 and (k >= 16 and k <= 17):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 22 and (k >= 16 and k <= 16):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 23 and (k >= 16 and k <= 16):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 24 and (k >= 16 and k <= 16):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # else:
                    #     lp_var = LpVariable(name, lowBound=0, upBound=1.0)
                    # ------------ 2 end ---------------


                    # # ------- version : 3 : for number 5 --------------
                    #
                    # low_bound_value = 0.0  # 0.6
                    # up_bound_value = 1.0
                    # # min valid value is 1
                    # gap_left = 1
                    # gap_right = 1
                    #
                    # org_low_bound_value = 0.0
                    # org_up_bound_value = 1.0
                    # # min valid value is 0
                    # org_gap_left = 4
                    # org_gap_right = 4
                    #
                    # if j == 4 and ((k >= 12 - gap_left and k < 12) or (k > 23 and k <= 23 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 4 and ( k >= 12 + org_gap_left and k <= 23 - org_gap_right):
                    #     lp_var = LpVariable(name, lowBound=org_low_bound_value, upBound=org_up_bound_value)
                    #
                    #
                    # elif j == 5 and ((k >= 8 - gap_left and k < 8) or (k > 20 and k <= 20 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 5 and (k >= 8 + org_gap_left and k <= 20 - org_gap_right):
                    #     lp_var = LpVariable(name, lowBound=org_low_bound_value, upBound=org_up_bound_value)
                    #
                    # elif j == 6 and ((k >= 8 - gap_left and k < 8) or (k > 17 and k <= 17 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 6 and (k >= 8 + org_gap_left and k <= 17 - org_gap_right):
                    #     lp_var = LpVariable(name, lowBound=org_low_bound_value, upBound=org_up_bound_value)
                    #
                    #
                    # elif j == 7 and ((k >= 8 - gap_left and k < 8) or (k > 13 and k <= 13 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 8 and ((k >= 8 - gap_left and k < 8) or (k > 14 and k <= 14 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 9 and ((k >= 9 - gap_left and k < 9) or (k > 14 and k <= 14 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 10 and ((k >= 9 - gap_left and k < 9) or (k > 18 and k <= 18 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 11 and ((k >= 8 - gap_left and k < 8) or (k > 19 and k <= 19 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 12 and ((k >= 7 - gap_left and k < 7) or (k > 22 and k <= 22 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 13 and ((k >= 7 - gap_left and k < 7) or (k > 12 and k <= 12 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 14 and ((k >= 18 - gap_left and k < 18) or (k > 24 and k <= 24 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 15 and ((k >= 19 - gap_left and k < 19) or (k > 25 and k <= 25 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 16 and ((k >= 20 - gap_left and k < 20) or (k > 26 and k <= 26 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 17 and ((k >= 21 - gap_left and k < 21) or (k > 26 and k <= 26 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 18 and ((k >= 22 - gap_left and k < 22) or (k > 26 and k <= 26 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 19 and ((k >= 21 - gap_left and k < 21) or (k > 26 and k <= 26 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # elif j == 20 and ((k >= 8 - gap_left and k < 8) or (k > 12 and k <= 12 + gap_right)):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # else:
                    #     lp_var = LpVariable(name, lowBound=0, upBound=1.0)
                    #
                    # # ------------ 3 end ---------------


                    # ------- version : 4 : for number 9 --------------
                    # low_bound_value = 0.0
                    # up_bound_value = 0.2
                    # gap_left = 0
                    # gap_right = 0
                    #
                    # if j == 17 and (k >= 16 and k <= 17):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 15 and (k >= 18 and k <= 18):
                    #     lp_var = LpVariable(name, lowBound=1.0, upBound=1.0)
                    #
                    # elif j == 16 and (k >= 18 and k <= 18):
                    #     lp_var = LpVariable(name, lowBound=1.0, upBound=1.0)
                    #
                    # elif j == 17 and (k >= 18 and k <= 18):
                    #     lp_var = LpVariable(name, lowBound=1.0, upBound=1.0)
                    #
                    # elif j == 18 and (k >= 18 and k <= 18):
                    #     lp_var = LpVariable(name, lowBound=1.0, upBound=1.0)
                    #
                    # elif j == 18 and (k >= 16 and k <= 17):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 19 and (k >= 15 - gap_left and k <= 17 + gap_right):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 20 and (k >= 15 - gap_left and k <= 17 + gap_right):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 21 and (k >= 15 - gap_left and k <= 16 + gap_right):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 22 and (k >= 14 - gap_left and k <= 16 + gap_right):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 23 and (k >= 14 - gap_left and k <= 15 + gap_right):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 24 and (k >= 14 - gap_left and k <= 15 + gap_right):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # elif j == 25 and (k >= 14 - gap_left and k <= 15 + gap_right):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    #
                    # else:
                    #     lp_var = LpVariable(name, lowBound=0, upBound=1.0)
                    # ------------ 4 end ---------------

                    # ------- version : 5:  for number :3: 02076 --------------
                    # low_bound_value = 0.0
                    # up_bound_value = 0.2
                    # gap_left = 0
                    # gap_right = 0
                    #
                    # if j == 5 and (k >= 13 and k <= 19):
                    #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # # elif j == 4 and (k >= 12 and k <= 15):
                    # #     lp_var = LpVariable(name, lowBound=low_bound_value, upBound=up_bound_value)
                    # else:
                    #     lp_var = LpVariable(name, lowBound=0, upBound=1.0)


                    lp_var = LpVariable(name, lowBound=0, upBound=1.0)
                    setattr(self, name, lp_var)


def seven_step(source_file, log_file):
    param = params_nodes()
    X = []
    for j in range(input_layer_1):
        for k in range(input_layer_2):
            for m in range(input_layer_3):
                name = "x_" + str(j) + "_" + str(k) + "_" + str(m)
                X.append(param.__getattribute__(name))

    # 2. 设置对象
    prob = LpProblem('mySolve', LpMaximize)
    # prob = LpProblem('mySolve', LpMinimize)

    # 3. 添加目标函数
    z = 0
    for i in range(len(X)):
        z += X[i]
    print(z)
    prob += z

    # 4. 载入约束变量
    idx = 1
    file = open(source_file, "r")  # 设置文件对象
    for line in file.readlines():
        print(str(idx) + ":" + line)
        idx = idx + 1
        arr = re.split(r"\+|\*|=", line)
        # 0-161 is the number and  coefficient, 162 is biases, 163 is the result
        # 163 is the result, if it greater than 0 ,it is the real value, else it should be less than or equal to 0
        result = float(arr[163])
        if result == 0.0:
            prob += (param.__getattribute__(arr[0]) * float(arr[1]) + param.__getattribute__(arr[2]) * float(arr[3])
                     + param.__getattribute__(arr[4]) * float(arr[5]) + param.__getattribute__(arr[6]) * float(arr[7])
                     + param.__getattribute__(arr[8]) * float(arr[9]) + param.__getattribute__(arr[10]) * float(arr[11])
                     + param.__getattribute__(arr[12]) * float(arr[13]) + param.__getattribute__(arr[14]) * float(
                        arr[15])
                     + param.__getattribute__(arr[16]) * float(arr[17]) + param.__getattribute__(arr[18]) * float(
                        arr[19])
                     + param.__getattribute__(arr[20]) * float(arr[21]) + param.__getattribute__(arr[22]) * float(
                        arr[23])
                     + param.__getattribute__(arr[24]) * float(arr[25]) + param.__getattribute__(arr[26]) * float(
                        arr[27])
                     + param.__getattribute__(arr[28]) * float(arr[29]) + param.__getattribute__(arr[30]) * float(
                        arr[31])
                     + param.__getattribute__(arr[32]) * float(arr[33]) + param.__getattribute__(arr[34]) * float(
                        arr[35])
                     + param.__getattribute__(arr[36]) * float(arr[37]) + param.__getattribute__(arr[38]) * float(
                        arr[39])
                     + param.__getattribute__(arr[40]) * float(arr[41]) + param.__getattribute__(arr[42]) * float(
                        arr[43])
                     + param.__getattribute__(arr[44]) * float(arr[45]) + param.__getattribute__(arr[46]) * float(
                        arr[47])
                     + param.__getattribute__(arr[48]) * float(arr[49]) + param.__getattribute__(arr[50]) * float(
                        arr[51])
                     + param.__getattribute__(arr[52]) * float(arr[53]) + param.__getattribute__(arr[54]) * float(
                        arr[55])
                     + param.__getattribute__(arr[56]) * float(arr[57]) + param.__getattribute__(arr[58]) * float(
                        arr[59])
                     + param.__getattribute__(arr[60]) * float(arr[61]) + param.__getattribute__(arr[62]) * float(
                        arr[63])
                     + param.__getattribute__(arr[64]) * float(arr[65]) + param.__getattribute__(arr[66]) * float(
                        arr[67])
                     + param.__getattribute__(arr[68]) * float(arr[69]) + param.__getattribute__(arr[70]) * float(
                        arr[71])
                     + param.__getattribute__(arr[72]) * float(arr[73]) + param.__getattribute__(arr[74]) * float(
                        arr[75])
                     + param.__getattribute__(arr[76]) * float(arr[77]) + param.__getattribute__(arr[78]) * float(
                        arr[79])
                     + param.__getattribute__(arr[80]) * float(arr[81]) + param.__getattribute__(arr[82]) * float(
                        arr[83])
                     + param.__getattribute__(arr[84]) * float(arr[85]) + param.__getattribute__(arr[86]) * float(
                        arr[87])
                     + param.__getattribute__(arr[88]) * float(arr[89]) + param.__getattribute__(arr[90]) * float(
                        arr[91])
                     + param.__getattribute__(arr[92]) * float(arr[93]) + param.__getattribute__(arr[94]) * float(
                        arr[95])
                     + param.__getattribute__(arr[96]) * float(arr[97]) + param.__getattribute__(arr[98]) * float(
                        arr[99])
                     + param.__getattribute__(arr[100]) * float(arr[101]) + param.__getattribute__(arr[102]) * float(
                        arr[103])
                     + param.__getattribute__(arr[104]) * float(arr[105]) + param.__getattribute__(arr[106]) * float(
                        arr[107])
                     + param.__getattribute__(arr[108]) * float(arr[109]) + param.__getattribute__(arr[110]) * float(
                        arr[111])
                     + param.__getattribute__(arr[112]) * float(arr[113]) + param.__getattribute__(arr[114]) * float(
                        arr[115])
                     + param.__getattribute__(arr[116]) * float(arr[117]) + param.__getattribute__(arr[118]) * float(
                        arr[119])
                     + param.__getattribute__(arr[120]) * float(arr[121]) + param.__getattribute__(arr[122]) * float(
                        arr[123])
                     + param.__getattribute__(arr[124]) * float(arr[125]) + param.__getattribute__(arr[126]) * float(
                        arr[127])
                     + param.__getattribute__(arr[128]) * float(arr[129]) + param.__getattribute__(arr[130]) * float(
                        arr[131])
                     + param.__getattribute__(arr[132]) * float(arr[133]) + param.__getattribute__(arr[134]) * float(
                        arr[135])
                     + param.__getattribute__(arr[136]) * float(arr[137]) + param.__getattribute__(arr[138]) * float(
                        arr[139])
                     + param.__getattribute__(arr[140]) * float(arr[141]) + param.__getattribute__(arr[142]) * float(
                        arr[143])
                     + param.__getattribute__(arr[144]) * float(arr[145]) + param.__getattribute__(arr[146]) * float(
                        arr[147])
                     + param.__getattribute__(arr[148]) * float(arr[149]) + param.__getattribute__(arr[150]) * float(
                        arr[151])
                     + param.__getattribute__(arr[152]) * float(arr[153]) + param.__getattribute__(arr[154]) * float(
                        arr[155])
                     + param.__getattribute__(arr[156]) * float(arr[157]) + param.__getattribute__(arr[158]) * float(
                        arr[159])
                     + param.__getattribute__(arr[160]) * float(arr[161])
                     + float(arr[162])) <= 0
        else:
            prob += (param.__getattribute__(arr[0]) * float(arr[1]) + param.__getattribute__(arr[2]) * float(arr[3])
                     + param.__getattribute__(arr[4]) * float(arr[5]) + param.__getattribute__(arr[6]) * float(arr[7])
                     + param.__getattribute__(arr[8]) * float(arr[9]) + param.__getattribute__(arr[10]) * float(arr[11])
                     + param.__getattribute__(arr[12]) * float(arr[13]) + param.__getattribute__(arr[14]) * float(
                        arr[15])
                     + param.__getattribute__(arr[16]) * float(arr[17]) + param.__getattribute__(arr[18]) * float(
                        arr[19])
                     + param.__getattribute__(arr[20]) * float(arr[21]) + param.__getattribute__(arr[22]) * float(
                        arr[23])
                     + param.__getattribute__(arr[24]) * float(arr[25]) + param.__getattribute__(arr[26]) * float(
                        arr[27])
                     + param.__getattribute__(arr[28]) * float(arr[29]) + param.__getattribute__(arr[30]) * float(
                        arr[31])
                     + param.__getattribute__(arr[32]) * float(arr[33]) + param.__getattribute__(arr[34]) * float(
                        arr[35])
                     + param.__getattribute__(arr[36]) * float(arr[37]) + param.__getattribute__(arr[38]) * float(
                        arr[39])
                     + param.__getattribute__(arr[40]) * float(arr[41]) + param.__getattribute__(arr[42]) * float(
                        arr[43])
                     + param.__getattribute__(arr[44]) * float(arr[45]) + param.__getattribute__(arr[46]) * float(
                        arr[47])
                     + param.__getattribute__(arr[48]) * float(arr[49]) + param.__getattribute__(arr[50]) * float(
                        arr[51])
                     + param.__getattribute__(arr[52]) * float(arr[53]) + param.__getattribute__(arr[54]) * float(
                        arr[55])
                     + param.__getattribute__(arr[56]) * float(arr[57]) + param.__getattribute__(arr[58]) * float(
                        arr[59])
                     + param.__getattribute__(arr[60]) * float(arr[61]) + param.__getattribute__(arr[62]) * float(
                        arr[63])
                     + param.__getattribute__(arr[64]) * float(arr[65]) + param.__getattribute__(arr[66]) * float(
                        arr[67])
                     + param.__getattribute__(arr[68]) * float(arr[69]) + param.__getattribute__(arr[70]) * float(
                        arr[71])
                     + param.__getattribute__(arr[72]) * float(arr[73]) + param.__getattribute__(arr[74]) * float(
                        arr[75])
                     + param.__getattribute__(arr[76]) * float(arr[77]) + param.__getattribute__(arr[78]) * float(
                        arr[79])
                     + param.__getattribute__(arr[80]) * float(arr[81]) + param.__getattribute__(arr[82]) * float(
                        arr[83])
                     + param.__getattribute__(arr[84]) * float(arr[85]) + param.__getattribute__(arr[86]) * float(
                        arr[87])
                     + param.__getattribute__(arr[88]) * float(arr[89]) + param.__getattribute__(arr[90]) * float(
                        arr[91])
                     + param.__getattribute__(arr[92]) * float(arr[93]) + param.__getattribute__(arr[94]) * float(
                        arr[95])
                     + param.__getattribute__(arr[96]) * float(arr[97]) + param.__getattribute__(arr[98]) * float(
                        arr[99])
                     + param.__getattribute__(arr[100]) * float(arr[101]) + param.__getattribute__(arr[102]) * float(
                        arr[103])
                     + param.__getattribute__(arr[104]) * float(arr[105]) + param.__getattribute__(arr[106]) * float(
                        arr[107])
                     + param.__getattribute__(arr[108]) * float(arr[109]) + param.__getattribute__(arr[110]) * float(
                        arr[111])
                     + param.__getattribute__(arr[112]) * float(arr[113]) + param.__getattribute__(arr[114]) * float(
                        arr[115])
                     + param.__getattribute__(arr[116]) * float(arr[117]) + param.__getattribute__(arr[118]) * float(
                        arr[119])
                     + param.__getattribute__(arr[120]) * float(arr[121]) + param.__getattribute__(arr[122]) * float(
                        arr[123])
                     + param.__getattribute__(arr[124]) * float(arr[125]) + param.__getattribute__(arr[126]) * float(
                        arr[127])
                     + param.__getattribute__(arr[128]) * float(arr[129]) + param.__getattribute__(arr[130]) * float(
                        arr[131])
                     + param.__getattribute__(arr[132]) * float(arr[133]) + param.__getattribute__(arr[134]) * float(
                        arr[135])
                     + param.__getattribute__(arr[136]) * float(arr[137]) + param.__getattribute__(arr[138]) * float(
                        arr[139])
                     + param.__getattribute__(arr[140]) * float(arr[141]) + param.__getattribute__(arr[142]) * float(
                        arr[143])
                     + param.__getattribute__(arr[144]) * float(arr[145]) + param.__getattribute__(arr[146]) * float(
                        arr[147])
                     + param.__getattribute__(arr[148]) * float(arr[149]) + param.__getattribute__(arr[150]) * float(
                        arr[151])
                     + param.__getattribute__(arr[152]) * float(arr[153]) + param.__getattribute__(arr[154]) * float(
                        arr[155])
                     + param.__getattribute__(arr[156]) * float(arr[157]) + param.__getattribute__(arr[158]) * float(
                        arr[159])
                     + param.__getattribute__(arr[160]) * float(arr[161])
                     + float(arr[162])) == result
    file.close()  # 将文件关闭

    # 5. 求解
    status = prob.solve()

    idx_to_value_map = {}

    # 显示结果
    for i in prob.variables():
        print(i.name + "=" + str(i.varValue))
        print(name_to_idx_map[i.name])
        idx_to_value_map[name_to_idx_map[i.name]] = i.varValue

    print("\nstatus: " + str(status))
    print("LpStatus[status]: " + LpStatus[status])
    print("result: " + str(value(prob.objective)) + "\n")  # 计算结果

    # ----- write to log file ----
    param_file_index = re.split(r"\.|=", source_file)[1]
    log_file.write("\n\n\n----------------- use param file: "+ param_file_index + " -------------------")
    log_file.write("\nstatus: " + str(status))
    log_file.write("\nLpStatus[status]: " + LpStatus[status])
    log_file.write("\nresult: " + str(value(prob.objective)) + "\n")


    if status == 1:
        # if the status is Optimal,  write the result to file
        if not os.path.exists(curr_p.ae_result_folder):
            os.mkdir(curr_p.ae_result_folder)

        file = open(curr_p.ae_result_file, 'w+')  # 可读可写可新建，覆盖方式

        for i in range(input_layer_1 * input_layer_2 * input_layer_3):
            file.write(str(idx_to_value_map[i]) + ",")
            # after write one line , there should be a '\n' to anther line
            if ((i + 1) % input_layer_2 == 0):
                file.write("\n")

        file.close()
        return True
    else:
        return False







