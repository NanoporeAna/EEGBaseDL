import scipy.io as sio
import numpy as np
path = 'D:/EEG_Data10/STFT/'

def load_psd_data_pool1(num,channel,name):
    """
    :param num: 第几个人的数据
    :param channel:原来9通道中选着哪一通道数据，我要的是后EEG 6通道的数据，即channel=3，表示EEG通道第一个通道的数据
    :param name: 要处理的数据是哪一个频带的数据，如name = 'alpha'
    :return:第num个人在name波段下channel-3 的eeg通道数据值
    """
    newpath = path + 'pool1/' + name + '/data'+ str(num+1)+'_'+name+'.mat'        #matlab数据处理后保存的路径
    load_data = sio.loadmat(newpath)  #
    load_matrix = load_data['data2']  #
    length = load_matrix.shape[0]
    temp = load_matrix[0:length,channel-3]
    data = np.array(temp)
    return data
def load_psd_data_pool2(num,channel,name):
    newpath = path + 'pool2/' + name + '/data'+ str(num+1)+'_'+name+'.mat'        #'data1_alpha.mat'
    load_data = sio.loadmat(newpath)  #
    load_matrix = load_data['data2']  #
    length = load_matrix.shape[0]
    temp = load_matrix[0:length,channel-3]
    data = np.array(temp)
    return data
def load_psd_data_pool3(num,channel,name):
    newpath = path + 'pool3/' + name + '/data'+ str(num+1)+'_'+name+'.mat'        #'data1_alpha.mat'
    load_data = sio.loadmat(newpath)  #
    load_matrix = load_data['data2']  #
    length = load_matrix.shape[0]
    temp = load_matrix[0:length,channel-3]
    data = np.array(temp)
    return data
def load_psd_data_pool4(num,channel,name,batchsize):
    newpath = path + 'pool4/' + name + '/data'+ str(num+1)+'_'+name+'.mat'        #'data1_alpha.mat'
    load_data = sio.loadmat(newpath)  #
    load_matrix = load_data['data2']  #
    length = load_matrix.shape[0]
    temp = load_matrix[0:length,channel-3]
    len = 0
    data = []
    # print(temp.shape)
    """
    这里要解决数据不匹配的问题
    """
    if batchsize%2==1:
        for i in range(batchsize-1):
            if i % 2 ==0:
                for j in range(len,len+313):
                    data.append(temp[j])
                len += 313
            if i% 2==1:
                len = len-1
                for j in range(len, len + 313):
                    data.append(temp[j])
                len += 313
        # print(len)
        for k in range(len,length):
            data.append(temp[k])
        data.append(temp[length-1])
    if batchsize%2==0:
        for i in range(batchsize):
            if i % 2 == 0:
                for j in range(len, len + 313):
                    data.append(temp[j])
                len += 313
            if i % 2 == 1:
                len = len - 1
                for j in range(len, len + 313):
                    data.append(temp[j])
                len += 313
    data = np.array(data)
    data = data.reshape(-1)
    return data
# da = load_psd_data_pool4(0,5,'alpha',542)
# print(da.shape)


