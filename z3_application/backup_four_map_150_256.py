# -*- coding: utf-8 -*-

from z3 import *
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
import re

# set_option(":pp.decimal-precision", 50)
# set_option(":pp.decimal", True)
set_option(rational_to_decimal=True)


source_folder = p.file_base + "/conv_network_simulation/compute_process_replace_result_with_ae/"
source_file = source_folder + "compute_process_layer3_with_x_ae_all_idx=0.param"
# equation_lines = 64  # 文件中总共有多少行
# param_num_in_line = 150  # 参与计算的一个卷积里有多少个参数
input_layer_1 = 12  # 输入矩阵的大小，由此计算总的输入节点个数
input_layer_2 = 12
input_layer_3 = 6

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
# print(param.x_11_11_5)
# print(is_real(param.x_11_11_5))
# print("x_11_11_5".find(str(param.x_11_11_5)))
idx = 1
file = open(source_file , "r")  # 设置文件对象
for line in file.readlines():
    print(str(idx) + ":" + line)
    idx = idx + 1
    arr = re.split(r"\+|\*|=", line)
    # 0-299 is the number and  coefficient, 300 is biases, 301 is the result
    # 301 is the result, if it greater than 0 ,it is the real value, else it should be less than or equal to 0
    result = float(arr[301])
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
              + param.__getattribute__(arr[160]) * float(arr[161]) + param.__getattribute__(arr[162]) * float(arr[163])
              + param.__getattribute__(arr[164]) * float(arr[165]) + param.__getattribute__(arr[166]) * float(arr[167])
              + param.__getattribute__(arr[168]) * float(arr[169]) + param.__getattribute__(arr[170]) * float(arr[171])
              + param.__getattribute__(arr[172]) * float(arr[173]) + param.__getattribute__(arr[174]) * float(arr[175])
              + param.__getattribute__(arr[176]) * float(arr[177]) + param.__getattribute__(arr[178]) * float(arr[179])
              + param.__getattribute__(arr[180]) * float(arr[181]) + param.__getattribute__(arr[182]) * float(arr[183])
              + param.__getattribute__(arr[184]) * float(arr[185]) + param.__getattribute__(arr[186]) * float(arr[187])
              + param.__getattribute__(arr[188]) * float(arr[189]) + param.__getattribute__(arr[190]) * float(arr[191])
              + param.__getattribute__(arr[192]) * float(arr[193]) + param.__getattribute__(arr[194]) * float(arr[195])
              + param.__getattribute__(arr[196]) * float(arr[197]) + param.__getattribute__(arr[198]) * float(arr[199])
              + param.__getattribute__(arr[200]) * float(arr[201]) + param.__getattribute__(arr[202]) * float(arr[203])
              + param.__getattribute__(arr[204]) * float(arr[205]) + param.__getattribute__(arr[206]) * float(arr[207])
              + param.__getattribute__(arr[208]) * float(arr[209]) + param.__getattribute__(arr[210]) * float(arr[211])
              + param.__getattribute__(arr[212]) * float(arr[213]) + param.__getattribute__(arr[214]) * float(arr[215])
              + param.__getattribute__(arr[216]) * float(arr[217]) + param.__getattribute__(arr[218]) * float(arr[219])
              + param.__getattribute__(arr[220]) * float(arr[221]) + param.__getattribute__(arr[222]) * float(arr[223])
              + param.__getattribute__(arr[224]) * float(arr[225]) + param.__getattribute__(arr[226]) * float(arr[227])
              + param.__getattribute__(arr[228]) * float(arr[229]) + param.__getattribute__(arr[230]) * float(arr[231])
              + param.__getattribute__(arr[232]) * float(arr[233]) + param.__getattribute__(arr[234]) * float(arr[235])
              + param.__getattribute__(arr[236]) * float(arr[237]) + param.__getattribute__(arr[238]) * float(arr[239])
              + param.__getattribute__(arr[240]) * float(arr[241]) + param.__getattribute__(arr[242]) * float(arr[243])
              + param.__getattribute__(arr[244]) * float(arr[245]) + param.__getattribute__(arr[246]) * float(arr[247])
              + param.__getattribute__(arr[248]) * float(arr[249]) + param.__getattribute__(arr[250]) * float(arr[251])
              + param.__getattribute__(arr[252]) * float(arr[253]) + param.__getattribute__(arr[254]) * float(arr[255])
              + param.__getattribute__(arr[256]) * float(arr[257]) + param.__getattribute__(arr[258]) * float(arr[259])
              + param.__getattribute__(arr[260]) * float(arr[261]) + param.__getattribute__(arr[262]) * float(arr[263])
              + param.__getattribute__(arr[264]) * float(arr[265]) + param.__getattribute__(arr[266]) * float(arr[267])
              + param.__getattribute__(arr[268]) * float(arr[269]) + param.__getattribute__(arr[270]) * float(arr[271])
              + param.__getattribute__(arr[272]) * float(arr[273]) + param.__getattribute__(arr[274]) * float(arr[275])
              + param.__getattribute__(arr[276]) * float(arr[277]) + param.__getattribute__(arr[278]) * float(arr[279])
              + param.__getattribute__(arr[280]) * float(arr[281]) + param.__getattribute__(arr[282]) * float(arr[283])
              + param.__getattribute__(arr[284]) * float(arr[285]) + param.__getattribute__(arr[286]) * float(arr[287])
              + param.__getattribute__(arr[288]) * float(arr[289]) + param.__getattribute__(arr[290]) * float(arr[291])
              + param.__getattribute__(arr[292]) * float(arr[293]) + param.__getattribute__(arr[294]) * float(arr[295])
              + param.__getattribute__(arr[296]) * float(arr[297]) + param.__getattribute__(arr[298]) * float(arr[299])
              + float(arr[300]) <= 0)
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
              + param.__getattribute__(arr[160]) * float(arr[161]) + param.__getattribute__(arr[162]) * float(arr[163])
              + param.__getattribute__(arr[164]) * float(arr[165]) + param.__getattribute__(arr[166]) * float(arr[167])
              + param.__getattribute__(arr[168]) * float(arr[169]) + param.__getattribute__(arr[170]) * float(arr[171])
              + param.__getattribute__(arr[172]) * float(arr[173]) + param.__getattribute__(arr[174]) * float(arr[175])
              + param.__getattribute__(arr[176]) * float(arr[177]) + param.__getattribute__(arr[178]) * float(arr[179])
              + param.__getattribute__(arr[180]) * float(arr[181]) + param.__getattribute__(arr[182]) * float(arr[183])
              + param.__getattribute__(arr[184]) * float(arr[185]) + param.__getattribute__(arr[186]) * float(arr[187])
              + param.__getattribute__(arr[188]) * float(arr[189]) + param.__getattribute__(arr[190]) * float(arr[191])
              + param.__getattribute__(arr[192]) * float(arr[193]) + param.__getattribute__(arr[194]) * float(arr[195])
              + param.__getattribute__(arr[196]) * float(arr[197]) + param.__getattribute__(arr[198]) * float(arr[199])
              + param.__getattribute__(arr[200]) * float(arr[201]) + param.__getattribute__(arr[202]) * float(arr[203])
              + param.__getattribute__(arr[204]) * float(arr[205]) + param.__getattribute__(arr[206]) * float(arr[207])
              + param.__getattribute__(arr[208]) * float(arr[209]) + param.__getattribute__(arr[210]) * float(arr[211])
              + param.__getattribute__(arr[212]) * float(arr[213]) + param.__getattribute__(arr[214]) * float(arr[215])
              + param.__getattribute__(arr[216]) * float(arr[217]) + param.__getattribute__(arr[218]) * float(arr[219])
              + param.__getattribute__(arr[220]) * float(arr[221]) + param.__getattribute__(arr[222]) * float(arr[223])
              + param.__getattribute__(arr[224]) * float(arr[225]) + param.__getattribute__(arr[226]) * float(arr[227])
              + param.__getattribute__(arr[228]) * float(arr[229]) + param.__getattribute__(arr[230]) * float(arr[231])
              + param.__getattribute__(arr[232]) * float(arr[233]) + param.__getattribute__(arr[234]) * float(arr[235])
              + param.__getattribute__(arr[236]) * float(arr[237]) + param.__getattribute__(arr[238]) * float(arr[239])
              + param.__getattribute__(arr[240]) * float(arr[241]) + param.__getattribute__(arr[242]) * float(arr[243])
              + param.__getattribute__(arr[244]) * float(arr[245]) + param.__getattribute__(arr[246]) * float(arr[247])
              + param.__getattribute__(arr[248]) * float(arr[249]) + param.__getattribute__(arr[250]) * float(arr[251])
              + param.__getattribute__(arr[252]) * float(arr[253]) + param.__getattribute__(arr[254]) * float(arr[255])
              + param.__getattribute__(arr[256]) * float(arr[257]) + param.__getattribute__(arr[258]) * float(arr[259])
              + param.__getattribute__(arr[260]) * float(arr[261]) + param.__getattribute__(arr[262]) * float(arr[263])
              + param.__getattribute__(arr[264]) * float(arr[265]) + param.__getattribute__(arr[266]) * float(arr[267])
              + param.__getattribute__(arr[268]) * float(arr[269]) + param.__getattribute__(arr[270]) * float(arr[271])
              + param.__getattribute__(arr[272]) * float(arr[273]) + param.__getattribute__(arr[274]) * float(arr[275])
              + param.__getattribute__(arr[276]) * float(arr[277]) + param.__getattribute__(arr[278]) * float(arr[279])
              + param.__getattribute__(arr[280]) * float(arr[281]) + param.__getattribute__(arr[282]) * float(arr[283])
              + param.__getattribute__(arr[284]) * float(arr[285]) + param.__getattribute__(arr[286]) * float(arr[287])
              + param.__getattribute__(arr[288]) * float(arr[289]) + param.__getattribute__(arr[290]) * float(arr[291])
              + param.__getattribute__(arr[292]) * float(arr[293]) + param.__getattribute__(arr[294]) * float(arr[295])
              + param.__getattribute__(arr[296]) * float(arr[297]) + param.__getattribute__(arr[298]) * float(arr[299])
              + float(arr[300]) == result)
file.close()  # 将文件关闭


# begin to check
print(s.check())
check_result = s.model()
print(check_result)


for item in check_result:
    print(item, ":", check_result[item])
# result_file.write(check_result)

result_file.close()



