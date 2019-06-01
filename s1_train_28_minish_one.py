# -*- coding: utf-8 -*-

from skimage import io,transform
import tensorboard
import os
import glob
import numpy as np
import pylab
import tensorflow as tf
import matplotlib.pyplot as plt
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置随机化种子，保持运算结果一致

#mnist数据集中训练数据和测试数据保存地址
train_path = '/home/lzs/Documents/my_image_net/mycode/data_set/mnist_data/train/'
# train_path = '/home/lzs/Documents/my_image_net/mycode/data_set/mnist_data/test/'
test_path = '/home/lzs/Documents/my_image_net/mycode/data_set/mnist_data/test/'

# 读取训练数据及测试数据
train_data,train_label = fs.read_image(train_path)
test_data,test_label = fs.read_image(test_path)

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

#搭建CNN
x = tf.placeholder(tf.float32, [None, fs.w, fs.h, fs.c], name='x')
y_ = tf.placeholder(tf.int32 ,[None], name='y_')

def inference(input_tensor,train,regularizer):

    tf.set_random_seed(10)

    #第一层：卷积层，过滤器的尺寸为9×9，深度为3,不使用全0补充，步长为1。
    #尺寸变化：28×28×1->24×24×6
    with tf.variable_scope('layer1-conv1'):
        # 5，5，1，6
        conv1_weights = tf.get_variable('weight', [p.layer1_conv_size, p.layer1_conv_size, fs.c, p.layer1_conv_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[p.layer1_conv_amount],initializer=tf.constant_initializer(0.0))  # 6
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        # -----------  添加tensorboard data ---------------
        tf.summary.histogram("layer1_conv_weights", conv1_weights)
        tf.summary.histogram("layer1_conv_biases", conv1_biases)
        # -----------  添加变量在预测的时候使用 ---------------
        tf.add_to_collection('layer1_conv_weights', conv1_weights)
        tf.add_to_collection('layer1_conv_biases', conv1_biases)
        tf.add_to_collection('layer1_conv_result', conv1)
        tf.add_to_collection('layer1_after_relu', relu1)

    #第二层：池化层，过滤器的尺寸为2×2，不使用全0补充(由于输入大小和过滤器可以除尽，设置用不用0补充都一样，最后都不补0)，步长为2。
    #尺寸变化：24×24×6->12×12×6
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        # -----------  添加tensorboard data ---------------
        tf.summary.histogram("layer2_pool", pool1)

        # -----------  添加变量在预测的时候使用 ---------------
        tf.add_to_collection('layer2_pool', pool1)

    # change to one map

    #第三层：卷积层，过滤器的尺寸为5×5，深度为3,不使用全0补充，步长为1。
    #尺寸变化：12×12×6->8×8×4
    # with tf.variable_scope('layer3-conv2'):
    #     # 5,5,6,4
    #     conv2_weights = tf.get_variable('weight',[p.layer3_conv_size,p.layer3_conv_size,p.layer1_conv_amount,p.layer3_conv_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv2_biases = tf.get_variable('bias',[p.layer3_conv_amount],initializer=tf.constant_initializer(0.0))  # 4
    #     conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='VALID')
    #     relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    #     # -----------  添加变量在预测的时候使用 ---------------
    #     tf.add_to_collection('layer3_conv_weights', conv2_weights)
    #     tf.add_to_collection('layer3_conv_biases', conv2_biases)
    #     tf.add_to_collection('layer3_conv_result', conv2)
    #     tf.add_to_collection('layer3_after_relu', relu2)
    #
    # #第四层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    # #尺寸变化：8×8×4->4×4×4
    # with tf.variable_scope('layer4-pool2'):
    #     pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    #     # -----------  添加变量在预测的时候使用 ---------------
    #     tf.add_to_collection('layer4_pool', pool2)

    #将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为4×4×16的矩阵，然而第五层全连接层需要的输入格式
    #为向量，所以我们需要把代表每张图片的尺寸为4×4×16的矩阵拉直成一个长度为4×4×16的向量。
    #举例说，每次训练64张图片，那么第四层池化层的输出的size为(64,4×4×16),拉直为向量，nodes=4×4×16=256,尺寸size变为(64,256)
    # pool_shape = pool2.get_shape().as_list()
    # nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    # reshaped = tf.reshape(pool2,[-1,nodes])

    pool_shape = pool1.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool1,[-1,nodes])

    # change end

    #第五层：全连接层，nodes=4×4×16=256，256->120的全连接
    #尺寸变化：比如一组训练样本为64，那么尺寸变化为64×256->64×120
    #训练时，引入dropout，dropout在训练时会随机将部分节点的输出改为0，dropout可以避免过拟合问题。
    #这和模型越简单越不容易过拟合思想一致，和正则化限制权重的大小，使得模型不能任意拟合训练数据中的随机噪声，以此达到避免过拟合思想一致。
    #本文最后训练时没有采用dropout，dropout项传入参数设置成了False，因为训练和测试写在了一起没有分离，不过大家可以尝试。
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,p.fc1_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[p.fc1_amount],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

        # -----------  添加tensorboard data ---------------
        tf.summary.histogram("fc1_weights", fc1_weights)
        tf.summary.histogram("fc1_biases", fc1_biases)
        tf.summary.histogram("fc1_after_relu", fc1)

        # -----------  添加变量在预测的时候使用 ---------------
        tf.add_to_collection('fc1_weights', fc1_weights)
        tf.add_to_collection('fc1_biases', fc1_biases)
        tf.add_to_collection('fc1_after_relu', fc1)

    #第六层：全连接层，120->84的全连接
    #尺寸变化：比如一组训练样本为64，那么尺寸变化为64×120->64×84
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight',[p.fc1_amount,p.fc2_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[p.fc2_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

        # -----------  添加tensorboard data ---------------
        tf.summary.histogram("fc2_weights", fc2_weights)
        tf.summary.histogram("fc2_biases", fc2_biases)
        tf.summary.histogram("fc2_after_relu", fc2)

        # -----------  添加变量在预测的时候使用 ---------------
        tf.add_to_collection('fc2_weights', fc2_weights)
        tf.add_to_collection('fc2_biases', fc2_biases)
        tf.add_to_collection('fc2_after_relu', fc2)


    #第七层：全连接层（近似表示），84->10的全连接
    #尺寸变化：比如一组训练样本为64，那么尺寸变化为64×84->64×10。最后，64×10的矩阵经过softmax之后就得出了64张图片分类于每种数字的概率，
    #即得到最后的分类结果。
    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable('weight',[p.fc2_amount,p.fc3_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[p.fc3_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

        # -----------  添加tensorboard data ---------------
        tf.summary.histogram("fc3_weights", fc3_weights)
        tf.summary.histogram("fc3_biases", fc3_biases)
        tf.summary.histogram("fc3_result", logit)

        # -----------  添加变量在预测的时候使用 ---------------
        tf.add_to_collection('fc3_weights', fc3_weights)
        tf.add_to_collection('fc3_biases', fc3_biases)
        tf.add_to_collection('fc3_result', logit)

    return logit

#正则化，交叉熵，平均交叉熵，损失函数，最小化损失函数，预测和实际equal比较，tf.equal函数会得到True或False，
#accuracy首先将tf.equal比较得到的布尔值转为float型，即True转为1.，False转为0，最后求平均值，即一组样本的正确率。
#比如：一组5个样本，tf.equal比较为[True False True False False],转化为float型为[1. 0 1. 0 0],准确率为2./5=40%。
regularizer = tf.contrib.layers.l2_regularizer(0.001)

# 这里得到的y是倒数第2层传到output层时，Relu计算后的值
y = inference(x, False, regularizer)

# 计算交叉熵
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_, name="cross_entropy")
# 求交叉熵的平均值，自动累加以后除以个数
cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
# 累计所有loss
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
# 实现了Adam算法的优化器
# ################################################################### 计算入口点
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
# 判断两个array对应的值是否相等，如果相等对应位置是True，否则是False
correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32), y_, name='correct_prediction')
# 将True和False转化为float32，然后累加，除以个数，计算平均正确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')

# -----------  添加tensorboard data ---------------
tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)

# -----------  添加变量在预测的时候使用 ---------------
tf.add_to_collection('x', x)
tf.add_to_collection('y', y)
tf.add_to_collection('y_', y_)
tf.add_to_collection('loss', loss)
tf.add_to_collection('train_op', train_op)

# 定义两个数组
Loss_list_train = []
Accuracy_list_train = []

Loss_list_test = []
Accuracy_list_test = []

#创建Session会话
with tf.Session() as sess:
    #初始化所有变量(权值，偏置等)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # -----------  添加tensorboard data ---------------
    merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
    writer_train = tf.summary.FileWriter("./temp-summary/train", sess.graph)
    writer_test = tf.summary.FileWriter("./temp-summary/test", sess.graph)


    #将所有样本训练10次，每次训练中以64个为一组训练完所有样本。
    #train_num可以设置大一些。
    train_num = 10
    batch_size = 64

    count_train = 0
    count_test = 0

    for i in range(train_num):

        train_loss,train_acc,batch_num,merged_summary = 0, 0, 0, 0
        for train_data_batch,train_label_batch in fs.get_batch(train_data,train_label,batch_size):
            _,err,acc,merged_summary = sess.run([train_op,loss,accuracy,merged],feed_dict={x:train_data_batch,y_:train_label_batch})
            train_loss+=err;train_acc+=acc;batch_num+=1
            count_train += 1


        print("in turn : ", i)
        print("train loss:",train_loss/batch_num)
        print("train acc:",train_acc/batch_num)

        Loss_list_train.append(train_loss/batch_num)
        Accuracy_list_train.append(train_acc/batch_num)

        # # -----------  添加tensorboard data ---------------
        writer_train.add_summary(merged_summary, i)


        test_loss,test_acc,batch_num,merged_test = 0, 0, 0, 0
        for test_data_batch,test_label_batch in fs.get_batch(test_data,test_label,batch_size):
            err,acc,merged_test = sess.run([loss,accuracy,merged],feed_dict={x:test_data_batch,y_:test_label_batch})
            test_loss+=err;test_acc+=acc;batch_num+=1
            count_test += 1


        print("in turn : ", i)
        print("test loss:",test_loss/batch_num)
        print("test acc:",test_acc/batch_num)

        Loss_list_test.append(test_loss/batch_num)
        Accuracy_list_test.append(test_acc/batch_num)

        # # -----------  添加tensorboard data ---------------
        writer_test.add_summary(merged_test, i)

    # 存储模型所有参数
    folder = "./temp-model/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    saver.save(sess, folder+"model.ckpt")


writer_train.close()
writer_test.close()

# draw
#我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上


fig,ax = plt.subplots()
ax.set_xlim([0,11])

x1 = range(1, 11)
x2 = range(1, 11)
x3 = range(1, 11)
x4 = range(1, 11)

y1 = Accuracy_list_train
y2 = Loss_list_train
y3 = Accuracy_list_test
y4 = Loss_list_test


plt.plot(x1, y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.xlabel('Epoches')
plt.ylabel('Train accuracy')
plt.savefig("./graph/Train-accuracy.jpg")
plt.show()

plt.plot(x2, y2, 'g*-')
plt.title('Train loss vs. epoches')
plt.xlabel('Epoches')
plt.ylabel('Train loss')
plt.savefig("./graph/Train-loss.jpg")
plt.show()

plt.plot(x3, y3, 'o-', color='orange')
plt.title('Test accuracy vs. epoches')
plt.xlabel('Epoches')
plt.ylabel('Test accuracy')
plt.savefig("./graph/Test-accuracy.jpg")
plt.show()

plt.plot(x4, y4, 'g*-', color='orange')
plt.title('Test loss vs. epoches')
plt.xlabel('Epoches')
plt.ylabel('Test loss')
plt.savefig("./graph/Test-loss.jpg")
plt.show()

plt.plot(x1, y1, 'o-', label="Train accuracy")
plt.plot(x1, y3, 'o-', label="Test accuracy", color='orange')
plt.title('Train accuracy vs. Test accuracy')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./graph/Train-accuracy-vs-Test-accuracy.jpg")
plt.show()

plt.plot(x2, y2, 'g*-', label="Train loss")
plt.plot(x2, y4, 'g*-', label="Test loss", color='orange')
plt.title('Train loss vs. Test loss')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.savefig("./graph/Train-loss-vs-Test-loss.jpg")
plt.show()




# plt.subplot(2, 3, 1)  #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
# plt.plot(x1, y1, 'o-')
# plt.title('Train accuracy vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Train accuracy')
#
# plt.subplot(2, 3, 2)
# plt.plot(x2, y2, 'g*-')
# plt.title('Train loss vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Train loss')
#
# plt.subplot(2, 3, 3)
# plt.plot(x3, y3, 'o-', color='orange')
# plt.title('Test accuracy vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Test accuracy')
#
# plt.subplot(2, 3, 4)
# plt.plot(x4, y4, 'g*-', color='orange')
# plt.title('Test loss vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Test loss')
#
# plt.subplot(2, 3, 5)
# plt.plot(x1, y1, 'o-')
# plt.plot(x1, y3, 'o-', color='orange')
# plt.title('Train accuracy vs. Test accuracy')
# plt.xlabel('Epoches')
# plt.ylabel('Accuracy')
#
# plt.subplot(2, 3, 6)
# plt.plot(x2, y2, 'g*-')
# plt.plot(x2, y4, 'g*-', color='orange')
# plt.title('Train loss vs. Test loss')
# plt.xlabel('Epoches')
# plt.ylabel('Loss')
#
#
# plt.savefig("./graph/accuracy_loss.jpg")
# plt.show()
