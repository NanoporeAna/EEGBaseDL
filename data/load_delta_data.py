import scipy.io as sio
import numpy as np
"""
alpha_data(num)表示去第num个经过alpha滤波的数据
return 对应mat数据集的batch
"""
## 本来是想将每个文件的路径放到一个txt文件中，通过列表提取每行赋的路径，但一直报错，'' 与""的问题--已解决
with open("H:/SpaceWork/EEG_Work/Delta_paths") as file_object:
    lines = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
delta_path = []
for line in lines:
    delta_path.append(line.strip())  # 将每行地址追加到一个数组里
#

###怎么将mat数据分成单个500*9的数据矩阵，将128*9矩阵放到一个batch里##
def delta_data(num):
    test_batch = []
    load_data0 = sio.loadmat(delta_path[num])  # 鍔犺浇mat鏂囦欢
    load_matrix = load_data0['data']  # 鎻愬彇鍑鸿鏁版嵁
    shape = load_matrix.shape[0]
    for i in range(0, int(shape/500)):#存储第一个人的数据，将其作为测试集
        batch = load_matrix[i * 500:(i + 1) * 500]  # 取500*9的数据矩阵
        test_batch.append(batch)  # 取得的矩阵追加到list里
    test_batch = np.array(test_batch)
    return test_batch

def load_delta_data(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    load_data = sio.loadmat(delta_path[path])
    load_matrix = load_data['data']
    shape = load_matrix.shape[0]
    delta_data = load_matrix[0:shape,count]
    data =np.array(delta_data)
    return data

def load_delta_data_pool1(path,count):
    #conv2也可以调用这个框架
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    load_data = sio.loadmat(delta_path[path])
    load_matrix = load_data['data']
    shape = load_matrix.shape[0]

    delta_data = load_matrix[0:shape,count]
    for i in range(0,int(shape/2)-1):
        tem = (delta_data[i*2]+delta_data[i*2+1]+delta_data[i*2+2])/3
        temp.append(tem)
    temp.append(delta_data[shape-1])
    data =np.array(temp)
    return data

def load_delta_data_pool2(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    load_data = sio.loadmat(delta_path[path])
    load_matrix = load_data['data']
    shape = load_matrix.shape[0]
    delta_data = load_matrix[0:shape,count]

    for i in range(0,int(shape/4)-1):
        tem = (delta_data[i*4]+delta_data[i*4+1]+delta_data[i*4+2]+delta_data[i*4+3]+delta_data[i*4+4])/5
        temp.append(tem)
    temp.append(delta_data[shape-1])
    data =np.array(temp)
    return data
def load_delta_data_pool3(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    load_data = sio.loadmat(delta_path[path])
    load_matrix = load_data['data']
    shape = load_matrix.shape[0]

    delta_data = load_matrix[0:shape,count]
    flg = int(shape / 500)
    for j in range(0, flg):
        for i in range(0,62):
            tem = (delta_data[j*500+i*8]+delta_data[j*500+i*8+1]+delta_data[j*500+i*8+2]+delta_data[j*500+i*8+3]+delta_data[j*500+i*8+4]+
            delta_data[j*500+i * 8+5] + delta_data[j*500+i * 8 + 6] + delta_data[j*500+i * 8 + 7] )/ 8
            temp.append(tem)
        flag = (delta_data[j * 500 + 496] + delta_data[j * 500 + 497] + delta_data[j * 500 + 498] + delta_data[
                j * 500 + 499]) / 4
        temp.append(flag)
    data =np.array(temp)
    return data

def load_delta_data_pool4(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    load_data = sio.loadmat(delta_path[path])
    load_matrix = load_data['data']
    shape = load_matrix.shape[0]
    delta_data = load_matrix[0:shape,count]
    flg = int(shape / 500)
    for j in range(0, flg):
        for i in range(0, 31):
            # tem = (delta_data[j * 500 + i * 16] + delta_data[j * 500 + i * 16 + 1] + delta_data[j * 500 + i * 16 + 2] +
            #        delta_data[j * 500 + i * 16 + 3] + delta_data[j * 500 + i * 16 + 4] +
            #        delta_data[j * 500 + i * 16 + 5] + delta_data[j * 500 + i * 16 + 6] + delta_data[
            #            j * 500 + i * 16 + 7] + delta_data[j * 500 + i * 16 + 8] + delta_data[j * 500 + i * 16 + 9] +
            #        delta_data[j * 500 + i * 16 + 10] + delta_data[j * 500 + i * 16 + 11] + delta_data[
            #            j * 500 + i * 16 + 12] +
            #        delta_data[j * 500 + i * 16 + 13] + delta_data[j * 500 + i * 16 + 14] + delta_data[
            #            j * 500 + i * 16 + 15]) / 16
            # temp.append(tem)
            tem = (delta_data[j * 500 + i * 16] + delta_data[j * 500 + i * 16 + 1] + delta_data[j * 500 + i * 16 + 2] +
                   delta_data[j * 500 + i * 16 + 3] + delta_data[j * 500 + i * 16 + 4] +
                   delta_data[j * 500 + i * 16 + 5] + delta_data[j * 500 + i * 16 + 6] + delta_data[
                       j * 500 + i * 16 + 7] + delta_data[j * 500 + i * 16+ 8] + delta_data[j * 500 + i * 16 + 9] +
                   delta_data[j * 500 + i * 16 + 10] + delta_data[j * 500 + i * 16 + 11] + delta_data[
                       j * 500 + i * 16 + 12] +
                   delta_data[j * 500 + i * 16 + 13] + delta_data[j * 500 + i * 16 + 14]+ delta_data[j * 500 + i * 16 + 15]) / 16
            temp.append(tem)
        flag = (delta_data[j * 500 + 496] + delta_data[j * 500 + 497] + delta_data[j * 500 + 498] + delta_data[
            j * 500 + 499]) / 4
        temp.append(flag)

    data = np.array(temp)
    return data







