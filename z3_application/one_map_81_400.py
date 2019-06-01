# -*- coding: utf-8 -*-

from z3 import *
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
import re

# set_option(":pp.decimal-precision", 50)
# set_option(":pp.decimal", True)
set_option(rational_to_decimal=True)
set_param(verbose=10)


source_folder = p.file_base + "/conv_network_simulation/compute_process_replace_result_with_ae/"
source_file = source_folder + "compute_process_layer1_with_x_ae_0.param"
# equation_lines = 400  # 文件中总共有多少行
# param_num_in_line = 81  # 参与计算的一个卷积里有多少个参数
input_layer_1 = 28  # 输入矩阵的大小，由此计算总的输入节点个数
input_layer_2 = 28
input_layer_3 = 1

# store the result
result_folder = p.file_base + "z3_application/check_result/"
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
result_file = open(result_folder + "check_result", 'w+')


s = Solver()

# build the z3 variable
class params_nodes():
    def __init__(self):
        # input variable
        for j in range(input_layer_1):
            for k in range(input_layer_2):
                for m in range(input_layer_3):
                    name = "x_" + str(j) + "_" + str(k) + "_" + str(m)
                    z3_var = Real(name)
                    s.add(z3_var >= 0)
                    setattr(self, name, z3_var)


param = params_nodes()

# print(param.x_27_27_0)
# print(is_real(param.x_27_27_0))
# print("x_27_27_0".find(str(param.x_27_27_0)))

idx = 1
file = open(source_file , "r")  # 设置文件对象
for line in file.readlines():
    print(str(idx) + ":" + line)
    idx = idx + 1
    arr = re.split(r"\+|\*|=", line)
    # 0-161 is the number and  coefficient, 162 is biases, 163 is the result
    # 163 is the result, if it greater than 0 ,it is the real value, else it should be less than or equal to 0
    result = float(arr[163])
    if result == 0.0:
        s.add(param.__getattribute__(arr[0]) * float(arr[1]) + param.__getattribute__(arr[2]) * float(arr[3])
              + param.__getattribute__(arr[4]) * float(arr[5]) + param.__getattribute__(arr[6]) * float(arr[7])
              + param.__getattribute__(arr[8]) * float(arr[9]) + param.__getattribute__(arr[10]) * float(arr[11])
              + param.__getattribute__(arr[12]) * float(arr[13]) + param.__getattribute__(arr[14]) * float(arr[15])
              + param.__getattribute__(arr[16]) * float(arr[17]) + param.__getattribute__(arr[18]) * float(arr[19])
              + param.__getattribute__(arr[20]) * float(arr[21]) + param.__getattribute__(arr[22]) * float(arr[23])
              + param.__getattribute__(arr[24]) * float(arr[25]) + param.__getattribute__(arr[26]) * float(arr[27])
              + param.__getattribute__(arr[28]) * float(arr[29]) + param.__getattribute__(arr[30]) * float(arr[31])
              + param.__getattribute__(arr[32]) * float(arr[33]) + param.__getattribute__(arr[34]) * float(arr[35])
              + param.__getattribute__(arr[36]) * float(arr[37]) + param.__getattribute__(arr[38]) * float(arr[39])
              + param.__getattribute__(arr[40]) * float(arr[41]) + param.__getattribute__(arr[42]) * float(arr[43])
              + param.__getattribute__(arr[44]) * float(arr[45]) + param.__getattribute__(arr[46]) * float(arr[47])
              + param.__getattribute__(arr[48]) * float(arr[49]) + param.__getattribute__(arr[50]) * float(arr[51])
              + param.__getattribute__(arr[52]) * float(arr[53]) + param.__getattribute__(arr[54]) * float(arr[55])
              + param.__getattribute__(arr[56]) * float(arr[57]) + param.__getattribute__(arr[58]) * float(arr[59])
              + param.__getattribute__(arr[60]) * float(arr[61]) + param.__getattribute__(arr[62]) * float(arr[63])
              + param.__getattribute__(arr[64]) * float(arr[65]) + param.__getattribute__(arr[66]) * float(arr[67])
              + param.__getattribute__(arr[68]) * float(arr[69]) + param.__getattribute__(arr[70]) * float(arr[71])
              + param.__getattribute__(arr[72]) * float(arr[73]) + param.__getattribute__(arr[74]) * float(arr[75])
              + param.__getattribute__(arr[76]) * float(arr[77]) + param.__getattribute__(arr[78]) * float(arr[79])
              + param.__getattribute__(arr[80]) * float(arr[81]) + param.__getattribute__(arr[82]) * float(arr[83])
              + param.__getattribute__(arr[84]) * float(arr[85]) + param.__getattribute__(arr[86]) * float(arr[87])
              + param.__getattribute__(arr[88]) * float(arr[89]) + param.__getattribute__(arr[90]) * float(arr[91])
              + param.__getattribute__(arr[92]) * float(arr[93]) + param.__getattribute__(arr[94]) * float(arr[95])
              + param.__getattribute__(arr[96]) * float(arr[97]) + param.__getattribute__(arr[98]) * float(arr[99])
              + param.__getattribute__(arr[100]) * float(arr[101]) + param.__getattribute__(arr[102]) * float(arr[103])
              + param.__getattribute__(arr[104]) * float(arr[105]) + param.__getattribute__(arr[106]) * float(arr[107])
              + param.__getattribute__(arr[108]) * float(arr[109]) + param.__getattribute__(arr[110]) * float(arr[111])
              + param.__getattribute__(arr[112]) * float(arr[113]) + param.__getattribute__(arr[114]) * float(arr[115])
              + param.__getattribute__(arr[116]) * float(arr[117]) + param.__getattribute__(arr[118]) * float(arr[119])
              + param.__getattribute__(arr[120]) * float(arr[121]) + param.__getattribute__(arr[122]) * float(arr[123])
              + param.__getattribute__(arr[124]) * float(arr[125]) + param.__getattribute__(arr[126]) * float(arr[127])
              + param.__getattribute__(arr[128]) * float(arr[129]) + param.__getattribute__(arr[130]) * float(arr[131])
              + param.__getattribute__(arr[132]) * float(arr[133]) + param.__getattribute__(arr[134]) * float(arr[135])
              + param.__getattribute__(arr[136]) * float(arr[137]) + param.__getattribute__(arr[138]) * float(arr[139])
              + param.__getattribute__(arr[140]) * float(arr[141]) + param.__getattribute__(arr[142]) * float(arr[143])
              + param.__getattribute__(arr[144]) * float(arr[145]) + param.__getattribute__(arr[146]) * float(arr[147])
              + param.__getattribute__(arr[148]) * float(arr[149]) + param.__getattribute__(arr[150]) * float(arr[151])
              + param.__getattribute__(arr[152]) * float(arr[153]) + param.__getattribute__(arr[154]) * float(arr[155])
              + param.__getattribute__(arr[156]) * float(arr[157]) + param.__getattribute__(arr[158]) * float(arr[159])
              + param.__getattribute__(arr[160]) * float(arr[161])
              + float(arr[162]) <= 0)
    else:
        s.add(param.__getattribute__(arr[0]) * float(arr[1]) + param.__getattribute__(arr[2]) * float(arr[3])
              + param.__getattribute__(arr[4]) * float(arr[5]) + param.__getattribute__(arr[6]) * float(arr[7])
              + param.__getattribute__(arr[8]) * float(arr[9]) + param.__getattribute__(arr[10]) * float(arr[11])
              + param.__getattribute__(arr[12]) * float(arr[13]) + param.__getattribute__(arr[14]) * float(arr[15])
              + param.__getattribute__(arr[16]) * float(arr[17]) + param.__getattribute__(arr[18]) * float(arr[19])
              + param.__getattribute__(arr[20]) * float(arr[21]) + param.__getattribute__(arr[22]) * float(arr[23])
              + param.__getattribute__(arr[24]) * float(arr[25]) + param.__getattribute__(arr[26]) * float(arr[27])
              + param.__getattribute__(arr[28]) * float(arr[29]) + param.__getattribute__(arr[30]) * float(arr[31])
              + param.__getattribute__(arr[32]) * float(arr[33]) + param.__getattribute__(arr[34]) * float(arr[35])
              + param.__getattribute__(arr[36]) * float(arr[37]) + param.__getattribute__(arr[38]) * float(arr[39])
              + param.__getattribute__(arr[40]) * float(arr[41]) + param.__getattribute__(arr[42]) * float(arr[43])
              + param.__getattribute__(arr[44]) * float(arr[45]) + param.__getattribute__(arr[46]) * float(arr[47])
              + param.__getattribute__(arr[48]) * float(arr[49]) + param.__getattribute__(arr[50]) * float(arr[51])
              + param.__getattribute__(arr[52]) * float(arr[53]) + param.__getattribute__(arr[54]) * float(arr[55])
              + param.__getattribute__(arr[56]) * float(arr[57]) + param.__getattribute__(arr[58]) * float(arr[59])
              + param.__getattribute__(arr[60]) * float(arr[61]) + param.__getattribute__(arr[62]) * float(arr[63])
              + param.__getattribute__(arr[64]) * float(arr[65]) + param.__getattribute__(arr[66]) * float(arr[67])
              + param.__getattribute__(arr[68]) * float(arr[69]) + param.__getattribute__(arr[70]) * float(arr[71])
              + param.__getattribute__(arr[72]) * float(arr[73]) + param.__getattribute__(arr[74]) * float(arr[75])
              + param.__getattribute__(arr[76]) * float(arr[77]) + param.__getattribute__(arr[78]) * float(arr[79])
              + param.__getattribute__(arr[80]) * float(arr[81]) + param.__getattribute__(arr[82]) * float(arr[83])
              + param.__getattribute__(arr[84]) * float(arr[85]) + param.__getattribute__(arr[86]) * float(arr[87])
              + param.__getattribute__(arr[88]) * float(arr[89]) + param.__getattribute__(arr[90]) * float(arr[91])
              + param.__getattribute__(arr[92]) * float(arr[93]) + param.__getattribute__(arr[94]) * float(arr[95])
              + param.__getattribute__(arr[96]) * float(arr[97]) + param.__getattribute__(arr[98]) * float(arr[99])
              + param.__getattribute__(arr[100]) * float(arr[101]) + param.__getattribute__(arr[102]) * float(arr[103])
              + param.__getattribute__(arr[104]) * float(arr[105]) + param.__getattribute__(arr[106]) * float(arr[107])
              + param.__getattribute__(arr[108]) * float(arr[109]) + param.__getattribute__(arr[110]) * float(arr[111])
              + param.__getattribute__(arr[112]) * float(arr[113]) + param.__getattribute__(arr[114]) * float(arr[115])
              + param.__getattribute__(arr[116]) * float(arr[117]) + param.__getattribute__(arr[118]) * float(arr[119])
              + param.__getattribute__(arr[120]) * float(arr[121]) + param.__getattribute__(arr[122]) * float(arr[123])
              + param.__getattribute__(arr[124]) * float(arr[125]) + param.__getattribute__(arr[126]) * float(arr[127])
              + param.__getattribute__(arr[128]) * float(arr[129]) + param.__getattribute__(arr[130]) * float(arr[131])
              + param.__getattribute__(arr[132]) * float(arr[133]) + param.__getattribute__(arr[134]) * float(arr[135])
              + param.__getattribute__(arr[136]) * float(arr[137]) + param.__getattribute__(arr[138]) * float(arr[139])
              + param.__getattribute__(arr[140]) * float(arr[141]) + param.__getattribute__(arr[142]) * float(arr[143])
              + param.__getattribute__(arr[144]) * float(arr[145]) + param.__getattribute__(arr[146]) * float(arr[147])
              + param.__getattribute__(arr[148]) * float(arr[149]) + param.__getattribute__(arr[150]) * float(arr[151])
              + param.__getattribute__(arr[152]) * float(arr[153]) + param.__getattribute__(arr[154]) * float(arr[155])
              + param.__getattribute__(arr[156]) * float(arr[157]) + param.__getattribute__(arr[158]) * float(arr[159])
              + param.__getattribute__(arr[160]) * float(arr[161])
              + float(arr[162]) == result)
file.close()  # 将文件关闭


# begin to check
print(s.check())
check_result = s.model()
print(check_result)


for item in check_result:
    print(item, ":", check_result[item])
# result_file.write(check_result)

result_file.close()



