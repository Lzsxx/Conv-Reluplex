from skimage import io,transform
import tensorboard
import os
import glob
import numpy as np
import pylab
import tensorflow as tf
import matplotlib.pyplot as plt


# 原版
# Accuracy_list_train = [0.8077641408751334, 0.9048492529348986, 0.9206410085378869, 0.9299459711846318, 0.9367495997865528,
#                        0.9424026147278548, 0.94545424226254, 0.9478221718249733, 0.9503902081109925, 0.953125]
#
# Accuracy_list_test = [0.8998397435897436, 0.9200721153846154, 0.9322916666666666, 0.9376001602564102, 0.9411057692307693,
#                       0.9452123397435898, 0.9482171474358975, 0.9489182692307693, 0.9508213141025641, 0.9522235576923077]

# 尾值契合原模型的修改版
Accuracy_list_train = [0.8077641408751334, 0.9048492529348986, 0.9206410085378869, 0.9299459711846318, 0.9367495997865528,
                       0.9424026147278548, 0.94545424226254, 0.9478221718249733, 0.9503902081109925, 0.9517]

Accuracy_list_test = [0.8998397435897436, 0.9200721153846154, 0.9322916666666666, 0.9376001602564102, 0.9411057692307693,
                      0.9452123397435898, 0.9482171474358975, 0.9489182692307693, 0.9493, 0.9498]


Loss_list_train = [0.6560529938311561, 0.3374877486103627, 0.28732025612189394, 0.2554848839309198, 0.23343824523649195,
                   0.21878574042717192, 0.20777546252614024, 0.19853511746881355, 0.19108914027712134, 0.18393276563051417]

Loss_list_test = [0.3544423222446289, 0.2893937605504806, 0.2542062121897172, 0.23399955191864416, 0.22052325413395196,
                  0.21203670925341356, 0.20625938723484674, 0.20035177402389356, 0.19364022019390875, 0.18939763276527324]



fig,ax = plt.subplots()
ax.set_xlim([0,11])

x1 = range(1, 11)
x2 = range(1, 11)
x3 = range(1, 11)
x4 = range(1, 11)

y1 = Accuracy_list_train
# y2 = Loss_list_train
y3 = Accuracy_list_test
# y4 = Loss_list_test

font1 = {
'weight' : 'normal',
'size'   : 15,
}

# -------- acc ---------
plt.plot(x1, y1, 'o-')
plt.title('训练准确率 vs. Epoches')
plt.xlabel('Epoches')
plt.ylabel('训练准确率')
plt.savefig("./训练准确率.jpg")
plt.show()

plt.plot(x3, y3, 'o-', color='orange')
plt.title('测试准确率 vs. Epoches')
plt.xlabel('Epoches')
plt.ylabel('测试准确率')
plt.savefig("./测试准确率.jpg")
plt.show()

plt.tick_params(labelsize=18)
plt.plot(x1, y1, 'o-', label="训练准确率")
plt.plot(x1, y3, 'v-', label="测试准确率", color='orange')
# plt.title('训练准确率 vs. 测试准确率', font1)
# plt.xlabel('Epoches', font1)
# plt.ylabel('准确率', font1)
plt.legend()
plt.savefig("./训练准确率-vs-测试准确率.jpg")
plt.show()

# -------- end -----------


# ---------- loss -----------
plt.plot(x1, Loss_list_train, 'o-')
plt.title('训练损失 vs. Epoches')
plt.xlabel('Epoches')
plt.ylabel('训练损失')
plt.savefig("./训练损失.jpg")
plt.show()

plt.plot(x3, Loss_list_test, 'o-', color='orange')
plt.title('测试损失 vs. Epoches')
plt.xlabel('Epoches')
plt.ylabel('测试损失')
plt.savefig("./测试损失.jpg")
plt.show()

plt.tick_params(labelsize=18)
plt.plot(x1, Loss_list_train, 'o-', label="训练损失")
plt.plot(x1, Loss_list_test, 'v-', label="测试损失", color='orange')
# plt.title('训练损失 vs. 测试损失', font1)
# plt.xlabel('Epoches', font1)
# plt.ylabel('损失', font1)
plt.legend()
plt.savefig("./训练损失-vs-测试损失.jpg")
plt.show()

# ------- end ------------

#
# plt.plot(x1, y1, 'o-')
# plt.title('Train accuracy vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Train accuracy')
# plt.savefig("./graph/Train-accuracy.jpg")
# plt.show()
#
# plt.plot(x2, y2, 'g*-')
# plt.title('Train loss vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Train loss')
# plt.savefig("./graph/Train-loss.jpg")
# plt.show()
#
# plt.plot(x3, y3, 'o-', color='orange')
# plt.title('Test accuracy vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Test accuracy')
# plt.savefig("./graph/Test-accuracy.jpg")
# plt.show()
#
# plt.plot(x4, y4, 'g*-', color='orange')
# plt.title('Test loss vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Test loss')
# plt.savefig("./graph/Test-loss.jpg")
# plt.show()
#
# plt.plot(x1, y1, 'o-', label="Train accuracy")
# plt.plot(x1, y3, 'o-', label="Test accuracy", color='orange')
# plt.title('Train accuracy vs. Test accuracy')
# plt.xlabel('Epoches')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig("./graph/Train-accuracy-vs-Test-accuracy.jpg")
# plt.show()
#
# plt.plot(x2, y2, 'g*-', label="Train loss")
# plt.plot(x2, y4, 'g*-', label="Test loss", color='orange')
# plt.title('Train loss vs. Test loss')
# plt.xlabel('Epoches')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig("./graph/Train-loss-vs-Test-loss.jpg")
# plt.show()