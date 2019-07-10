# import  scipy.io as sio
# import numpy as np
# from scipy import fftpack
#
# import  matplotlib.pyplot as plt
# from sklearn import preprocessing
#
# with open("C:/Users/Nick/PycharmProjects/EEG_Work/path.txt") as file_object:
#     lines = file_object.readlines()
# math_path = []
#
# for line in lines:
#     math_path.append(line.strip())
# with open("C:/Users/Nick/PycharmProjects/EEG_Work/lable.txt") as file_object:
#     lines = file_object.readlines()
# lable_value = []
# for line in lines:
#     lable_value.append(int(line.strip()))
#
# def build_single_object():
#     load_data = sio.loadmat(math_path[0])
#     load_maxtrix = load_data['data2']
#     shape = load_maxtrix.shape
#     len = shape[0]
#     pre_train = load_maxtrix[0:len-1,0]
#     pre_train = pre_train.reshape(-1, 1)
#     train_single_batch = preprocessing.MinMaxScaler().fit_transform(pre_train)
#     return train_single_batch
#
# data = build_single_object()#信号数据的准备
# hx = fftpack.hilbert(data)
# envelop = np.sqrt(data**2+hx**2)
# print(envelop)
# plt.figure(figsize=(10, 10))
# plt.plot(data,'b')
# plt.title(u"raw data")
# plt.show()
# plt.figure(figsize=(10,10))
# plt.plot(envelop,'r')
# plt.title(u"envelop")
# plt.show()
#
import numpy as np
import tensorflow as tf
from scipy import fftpack

from datasets.DataSet import DataSet
from data.load_data import build_batch
import matplotlib.pyplot as plt

def one_hot(labels, n_class = 6):
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



"""
学习率指数衰减，没有实现

LEARNING_RATE_BASE = 0.1 #最初学习率
LEARNING_RATE_DECAY = 0.99 #学习率衰减率
LEARNING_RATE_STEP = 1 #喂入多少轮BATCH_SIZE后，更新一此学习率，一般为：总样本数/BATCH_SIZE
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.data.exponential_decay(LEARNING_RATE_BASE,
    global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
"""
batch_size = 100      # Batch size 模仿visulation 那篇论文里大脑解码的size
seq_len = 500          # Number of steps 500HZ的数据
learning_rate = 0.000011
epochs = int(100000/batch_size)#train_batch 只有111749个
n_classes = 6
n_channels = 9
graph = tf.Graph()
# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None,n_classes],name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')


with graph.as_default():
    # (1, 500, 9) -> (32, 250, 5)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=32, kernel_size=2, strides=1,
        padding='same', activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=2, padding='same')
    # (32, 250, 5) -> (64, 125,3 )
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=2, strides=1,
    padding='same', activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=3, strides=2, padding='same')
    # (64, 125, 3) -> (128, 63, 2)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=128, kernel_size=2, strides=1,
    padding='same', activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=3, strides=2, padding='same')
    # (128, 63, 2) -> (256, 32, 1)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=256, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=3, strides=2, padding='same')


with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_4, (-1, 256*32*1))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    # Predictions
    logits = tf.layers.dense(flat, n_classes)
    # Cost function and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
        labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    # Loop over epochs
    #(x,y),(m,n) = build_batch()
    x,y =build_batch()
    # y = standardize(y)#行不通，标准化很慢，且最后一直出第一个batch的acc，loss为nan
    for it in range(iteration):
        for e in range(1):
            # Loop over batches
            batch_xs, batch_ys = DataSet(x, y, batch_size).next_batch(batch_size)
            batch_ys = one_hot(batch_ys)
            batch_xs = standardize(batch_xs)
            # Feed dictionary
            feed = {inputs_: batch_xs, labels_: batch_ys, keep_prob_: 0.5, learning_rate_: learning_rate}
            max1 = sess.run([max_pool_1], feed_dict=feed)#取第一个卷积层卷积池化后的输出值
            temp = max1[0]#只需要shape

            # hx = fftpack.hilbert(temp)
            # envelop = np.sqrt(temp** 2 + hx ** 2)
            print(temp[0])
            #
            # print(envelop)


