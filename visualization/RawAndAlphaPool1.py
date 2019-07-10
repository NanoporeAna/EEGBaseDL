import tensorflow as tf
import numpy as np
import  pandas as pd
from data.load_single_data import test_batch, load_alpha_data, load_alpha_data_pool1, load_alpha_data_pool2, \
    load_alpha_data_pool3
from model.CNN_Model import BaseCNN
def one_hot(labels, n_class = 7):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"
	return y
fliter = [16, 32, 64, 128]
chan = [9, 5, 5, 3, 3, 2, 2]
# 神经网络的参数
bat = [5424, 10376, 7310, 5228, 4284, 2870, 6015, 4192, 8457, 8026, 7781, 10608, 916, 4338, 2089]
Batch_Size = 100
Learning_Rate_Base = 0.0005
Learning_Rate_Decay = 0.99
Regularazition_Rate = 0.0005
Training_Steps = 967
Moving_Average_Decay =0.99
Model_Save_Path = "H:/SpaceWork/EEG_Work/Model/CNN_BN_v12"
Model_Name = "model.ckpt"
def evaluate(num,channel,name,fliter):
    # num 表示要取那个人的数据，channel 表示对那个通道数据感兴趣，name表示是哪一层，fliter 表示神经网络中间层滤波器的数量
    # return 第num 个人数据经过测试得到的在name层处理后的输出数据对应channel的值。
    with tf.name_scope("input"):
        input_x = tf.placeholder(tf.float32, [5424, 500, 9, 1], name='EEG-input')  # 数据的输入，第一维表示一个batch中样例的个数
        input_y = tf.placeholder(tf.float32, [None, 7], name='EEG-lable')  # 一个batch里的lable
    regularlizer = tf.contrib.layers.l2_regularizer(Regularazition_Rate)#本来测试的时候不用加这个
    out = BaseCNN(input_x,False,regularlizer)
    y = out['logit']
    with tf.name_scope("test_acc"):
        correct_predection = tf.equal(tf.argmax(y,1),tf.argmax(input_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predection,tf.float32))
        tf.summary.scalar('test_acc', accuracy)
    variable_averages = tf.train.ExponentialMovingAverage(Moving_Average_Decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    raw_channel = []

    with  tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(Model_Save_Path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            x, y = test_batch(num)#获取第x个人的数据
                # x = x[100:200]
                # y = y[100:200]
                # xs = standardize(x)
            reshape_xs = np.reshape(x,(-1,500,9,1))
            ys = one_hot(y)
            data,acc_score =sess.run([out['pool4'],accuracy],feed_dict={input_x:reshape_xs,input_y:ys})

            #length = data[0]*data[1]
            for i in range(fliter):
                temp = data[:,:,channel,i]
                raw = np.reshape(temp,[173568])# 数据的总长度
                raw_channel.append(raw)
            print("Afer %s training step, test accuracy = %g" % (global_step,acc_score))
            print(np.shape(data))
        else :
            print("No checkpoint file found")
    return raw_channel
def get_corr_fliter():
    mean = []
    corr = []


    # for i in range(15):
    #     conv1 = evaluate(i)
    #     alpha = load_alpha_data(i,3)
    #     for j in range(fliter[0]):
    #         cor = pd.DataFrame({'raw': conv1[i], 'alpha': alpha})
    #         corr.append(cor.raw.corr(cor.alpha))
    #     mean.append(np.mean(corr))
    conv1 = evaluate(0,0,'pool3',128)#得到经神经网络conv1层后所有滤波器下结果值
    alpha = load_alpha_data_pool3(0,3)#得到预处理数据下3通道的数据值


    for i in range(128):
        cor = pd.DataFrame({'raw': conv1[i], 'alpha': alpha})#构建相关性的数据型
        corr.append(cor.raw.corr(cor.alpha))#得到各滤波器与预处理数据的相关性值
    mean.append(np.mean(corr))
    print(mean[0])
# get_corr_fliter()