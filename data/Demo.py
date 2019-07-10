import os
import time

import tensorflow as tf
import  numpy as np
from data.train_data import train_batch
from data.test_7500_data import test_batch
from datasets.DataSet import DataSet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
def one_hot(labels, n_class = 7):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"
	return y

def standardize(train):
	""" Standardize data """
	# Standardize data
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	return X_train

def BaseCNN(input_tensor,train, regularizer):
    #卷积网第一层架构 输入为500*9*32 的矩阵
    with tf.variable_scope('layer1-conv'):
        conv1_weights = tf.get_variable("weight",[3,3,1,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases',[16],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME',name='conv1')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    with tf.variable_scope('layer1-pool'):
        pool1 =tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    #卷积网第二层架构输入为250*5*64
    with tf.variable_scope('layer2-conv'):
        conv2_weights = tf.get_variable("weight",[3,3,16,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases',[32],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME',name='conv2')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.variable_scope('layer2-pool'):
        pool2 =tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    # 卷积网第二层架构输入为125*3*128
    with tf.variable_scope('layer3-conv'):
        conv3_weights = tf.get_variable("weight",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('biases',[64],initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2,conv3_weights,strides=[1,1,1,1],padding='SAME',name='conv3')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
    with tf.variable_scope('layer3-pool'):
        pool3 =tf.nn.max_pool(relu3,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool3')
    # 卷积网第二层架构输入为63*2*256
    with tf.variable_scope('layer4-conv'):
        conv4_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('biases',[128],initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3,conv4_weights,strides=[1,1,1,1],padding='SAME',name='conv4')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))
    with tf.variable_scope('layer4-pool'):
        pool4 =tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4')
    #将第四层池化层的输出转化为全连接层的输入格式，注意，因为每一层神经网络的输入输出都为一个batch的矩阵，
    # 所以这里得到的维度也包含一个batch的数据的个数
    pool_shape = pool4.get_shape().as_list()
    #计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽及深度的乘积，注意这里pool_shape[0]为一个batch中数据的个数
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    #将第四层的输出转变为一个batch的向量
    reshaped = tf.reshape(pool4,[pool_shape[0],nodes])
    with tf.variable_scope('layer5-fc1'):
        fc1_weights =tf.get_variable('weight',[nodes,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias",[512],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases,name='fc1')
        if train: fc1 = tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights  = tf.get_variable('weight',[512,7],initializer=tf.truncated_normal_initializer(0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('biase',[7],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases
        out ={
            'conv1':conv1,
            'pool1':pool1,
            'conv2':conv2,
            'pool2':pool2,
            'conv3':conv3,
            'pool3':pool3,
            'conv4':conv4,
            'pool4':pool4,
            'fc1':fc1,
            'logit':logit
        }

    return out


#配置神经网络参数
Batch_Size = 100
Learning_Rate_Base = 0.00001
Learning_Rate_Decay = 0.99
Regularazition_Rate = 0.0001
Training_Steps =1000
Moving_Average_Decay =0.99
Model_Save_Path = "CNN_Model"
Model_Name = "model.ckpt"
def evaluate():
    with tf.name_scope("input"):
        input_x = tf.placeholder(tf.float32, [10080, 500, 9, 1], name='EEG-input')  # 数据的输入，第一维表示一个batch中样例的个数
        input_y = tf.placeholder(tf.float32, [None, 7], name='EEG-lable')  # 一个batch里的lable
    regularlizer = tf.contrib.layers.l2_regularizer(Regularazition_Rate)#本来测试的时候不用加这个
    y = BaseCNN(input_x,False,regularlizer)['logits']
    conv1_ = y['conv1']
    with tf.name_scope("test_acc"):
        correct_predection = tf.equal(tf.argmax(y,1),tf.argmax(input_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predection,tf.float32))
        tf.summary.scalar('test_acc', accuracy)
    variable_averages = tf.train.ExponentialMovingAverage(Moving_Average_Decay)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    while True:
        with  tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(Model_Save_Path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                x, y = test_batch()
                # x = x[100:200]
                # y = y[100:200]
                xs = standardize(x)
                reshape_xs = np.reshape(xs,(-1,500,9,1))
                ys = one_hot(y)
                conv1,acc_score =sess.run([conv1_,accuracy],feed_dict={input_x:reshape_xs,input_y:ys})#提取神经网咯中conv1
                print("Afer %s training step, test accuracy = %g" % (global_step,acc_score))
                print(conv1[0])
            else :
                print("No checkpoint file found")
                return
            time.sleep(10)

def main(argv =None):
    # train()
    evaluate()
if __name__ == '__main__':
    tf.app.run()

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir logs