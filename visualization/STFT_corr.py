import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io as sio
from DATA.load_PSD_data10 import load_psd_data_pool1, load_psd_data_pool4, load_psd_data_pool3, load_psd_data_pool2
from DATA.load_single_data10 import raw_test_batch10
from MODELS.CNNLSTM import CNN_LSTM
"""
这里我们想做的是将每个人波段的光谱密度均值作为特征值即（10s 10个数据值作为一个变量）与神经网络处理后感受野对应下的结果值做相关
但这里两个样本容量不一致，没法做person相关，要不就是将神经网络处理后的数据变为10，或者将特征值放宽
2019年10月13日15:46:39 这里在matlab就按照神经网size变换的size得到的功谱率均值来做相关性
"""

def one_hot(labels, n_class = 7):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"
	return y
fliter = [4, 8, 16, 32, 128]
band = 'delta'
# 神经网络的参数
bat = [542, 1037, 731, 522, 428, 287, 601, 419, 845, 802, 778, 1060, 91, 433, 208]
Learning_Rate_Base = 0.00325
Learning_Rate_Decay = 0.99
Regularazition_Rate = 0.00325
Moving_Average_Decay =0.99
Model_Save_Path = "H:/SpaceWork/CNN-LSTM/MODELS/CNNLSTM_v10"
Model_Name = "model.ckpt"
def evaluate(num):
    # num 表示要取那个人的数据
    # return 第num 个人数据经过测试得到数据。
    with tf.name_scope("input"):
        input_x = tf.placeholder(tf.float32, [bat[num], 5000, 9, 1], name='EEG-input')  # 数据的输入，第一维表示一个batch中样例的个数
        input_y = tf.placeholder(tf.float32, [None, 7], name='EEG-lable')  # 一个batch里的lable
    regularlizer = tf.contrib.layers.l2_regularizer(Regularazition_Rate)#本来测试的时候不用加这个
    is_training = tf.cast(False, tf.bool)
    out = CNN_LSTM(input_x, is_training, None)
    y = out['logist']
    with tf.name_scope("test_acc"):
        correct_predection = tf.equal(tf.argmax(y,1),tf.argmax(input_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predection,tf.float32))
        tf.summary.scalar('test_acc', accuracy)
    variable_averages = tf.train.ExponentialMovingAverage(Moving_Average_Decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    with  tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(Model_Save_Path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            x, y = raw_test_batch10(num)#获取第x个人的数据
            reshape_xs = np.reshape(x,(-1,5000,9,1))
            ys = one_hot(y)
            conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4,lstm, acc_score =sess.run([out['conv1'], out['pool1'], out['conv2'], out['pool2'],
                                                                                         out['conv3'], out['pool3'], out['conv4'], out['pool4'],out['rnn'],
                                                                                        accuracy],feed_dict={input_x: reshape_xs, input_y: ys})
            pool4 = np.reshape(pool4,(-1,313,9,32))
            Lstm = np.reshape(lstm,(-1,313,1,128))

            print("Afer %s training step, test accuracy = %g" % (global_step,acc_score))
        else :
            print("No checkpoint file found")
    return  conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, Lstm
def get_corr_fliter(num,data,channle,fliter,pool):
    """
    :param num: 要处理的第几个人的数据
    :param data: 神经网络处理得到的数据
    :param channle: 神经网络下的通道数
    :param fliter: 该data下的滤波器数
    :param pool: 表示神经网络哪一层
    :return:
    """
    list = []
    # con = evaluate(num,9,'conv1',16,2712000)#返回各通道滤波器下的list evaluate(num,channel,name,fliter,size):
    # num 表示要取那个人的数据，channel 表示对那个隐藏层通道数据感兴趣，name表示是哪一层，fliter 表示神经网络中间层滤波器的数量 size隐藏层处理数据长度
    # return 第num 个人数据经过测试得到的在name层处理后的输出数据对应channel的值。
    for i in range(3,channle):
        temp = data[i] #得到第i通道下各滤波器数据
        if pool ==1:
            ga = load_psd_data_pool1(num,i,band) # 这里得到的是我们已经求得的对应频段下PSD 功谱率均值
        if pool ==3:
            ga = load_psd_data_pool2(num,i,band) # 这里得到的是我们已经求得的对应频段下PSD 功谱率均值
        if pool ==5:
            ga = load_psd_data_pool3(num,i,band) # 这里得到的是我们已经求得的对应频段下PSD 功谱率均值
        if pool ==7:
            ga = load_psd_data_pool4(num,i,band,bat[num]) # 这里得到的是我们已经求得的对应频段下PSD 功谱率均值
        cost = []
        for k in range(fliter):
            cor = pd.DataFrame({'raw': temp[k], 'gamma': ga})  # 构建相关性的数据型
            cost.append(cor.raw.corr(cor.gamma))  # 得到各滤波器与预处理数据的相关性值
        x = np.mean(cost)
        # x = max(cost,key=abs)
        print("神经网络第 %g通道后的数据与原通道 %g的相关性为：%g " % (i, i, x))
        list.append(x)  # 二维矩阵
    return list

def sum(num):
    conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, lstm = evaluate(num)
    name = [conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, lstm]
    channel = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7','x8', 'x9']
    data1 = []
    for i in range(8):#i表示处理后第几层数据
        #     print(fliter[int(i/2)])`
        data = name[i]
        list = []
        size = data.shape[0] * data.shape[1]  # 对应滤波alpha or gamma or ... 原始数据通道铺平的长度
        # print('size = %g'% (size))
        for j in range(9):  # 隐藏层各通道
            flag = []
            for k in range(fliter[int(i / 2)]):  # 神经网络fliter器数量
                temp = data[:, :, j, k]  #
                temp = np.reshape(temp, [size])
                flag.append(temp)  # 第j个通道下各滤波器下的值
            list.append(flag)  # 第i层神经网所有通道
        if i %2 == 1:
            print("第%g神经层数据相关性处理开始：" % (i))
            channel[i] = get_corr_fliter(num, list, 9, fliter[int(i / 2)], i)
            # channel[i] = np.array(channel[i])
            data1.append(channel[i])
    data1 = np.array(data1)
    data = np.reshape(data1, (-1, 6))#得到6个EEG通道的相关性值
    sio.savemat('D:/CNN_LSTM/newResult10sPSD/'+band+'/'+str(num)+'.mat', {'data':data})

def main(argv=None):
    for i in range(15):
        tf.reset_default_graph() # Python的控制台会保存上次运行结束的变量，需要将之前的结果清除
        sum(i)
    # sum(0) # 计算第num个人的数据分析

if __name__ == '__main__':
    tf.app.run()