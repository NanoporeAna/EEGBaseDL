import scipy.io as sio
import numpy as np

## 本来是想将每个文件的路径放到一个txt文件中，通过列表提取每行赋的路径，但一直报错，'' 与""的问题--已解决
with open("H:/SpaceWork/EEG_Work/path.txt") as file_object:
    lines = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
mat_path = []
for line in lines:
    mat_path.append(line.strip())  # 将每行地址追加到一个数组里
# print("ok")
with open("H:/SpaceWork/EEG_Work/raw_paths.txt") as file_object:
    line_alpha = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
raw_path=[]
for alpha in line_alpha:
    raw_path.append(alpha.strip())
with open("H:/SpaceWork/EEG_Work/lables.txt") as file_object:
    lines_lable = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
lable_value = []
for line in lines_lable:
    lable_value.append(int(line.strip()))  # 将每行标签值追加到一个数组里

mat_dictionary = {}  # 构造字典
for i in range(0, 15):
    mat_dictionary[mat_path[i]] = lable_value[i]  # 存入每个对应的值

def get_lable(load_path):  # 根据路径得到标签
    return mat_dictionary[load_path]

###怎么将mat数据分成单个500*9的数据矩阵，将128*9矩阵放到一个batch里##
def test_batch(num):
    #取num位实验者数据
    test_batch = []
    test_label = []
    load_data0 = sio.loadmat(mat_path[num])  # 鍔犺浇mat鏂囦欢
    load_matrix = load_data0['data2']  # 鎻愬彇鍑鸿鏁版嵁
    shape = load_matrix.shape[0]
    for i in range(0, int(shape/500)):#存储第一个人的数据，将其作为测试集
        batch = load_matrix[i * 500:(i + 1) * 500]  # 取500*9的数据矩阵
        label = get_lable(mat_path[num])
        test_batch.append(batch)  # 取得的矩阵追加到list里
        test_label.append(label)
    test_batch = np.array(test_batch)
    test_label = np.array(test_label)
    return test_batch,test_label

def load_object_batch(num):
    #num表示要取哪位患者的数据集
    test_batch = []
    test_label = []
    load_data0 = sio.loadmat(raw_path[num])  # 鍔犺浇mat鏂囦欢
    load_matrix = load_data0['data2']  # 鎻愬彇鍑鸿鏁版嵁
    shape = load_matrix.shape[0]
    for i in range(0, int(shape / 500)):  # 存储第一个人的数据，将其作为测试集
        batch = load_matrix[i * 500:(i + 1) * 500]  # 取500*9的数据矩阵
        label = get_lable(mat_path[num])
        test_batch.append(batch)  # 取得的矩阵追加到list里
        test_label.append(label)
    test_batch = np.array(test_batch)
    test_label = np.array(test_label)
    return test_batch, test_label
def load_raw_data(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    load_data = sio.loadmat(raw_path[path])
    load_matrix = load_data['data2']
    shape = load_matrix.shape[0]
    alpha_data = load_matrix[0:shape,count]
    data =np.array(alpha_data)
    return data

def load_raw_data_pool1(path,count):
    #conv2也可以调用这个框架
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    load_data = sio.loadmat(raw_path[path])
    load_matrix = load_data['data2']
    shape = load_matrix.shape[0]

    alpha_data = load_matrix[0:shape,count]
    for i in range(0,int(shape/2)-1):
        tem = (alpha_data[i*2]+alpha_data[i*2+1]+alpha_data[i*2+2])/3
        temp.append(tem)
    temp.append(alpha_data[shape-1])
    data =np.array(temp)
    return data

def load_raw_data_pool2(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    load_data = sio.loadmat(raw_path[path])
    load_matrix = load_data['data2']
    shape = load_matrix.shape[0]

    alpha_data = load_matrix[0:shape,count]

    for i in range(0,int(shape/4)-1):
        tem = (alpha_data[i*4]+alpha_data[i*4+1]+alpha_data[i*4+2]+alpha_data[i*4+3]+alpha_data[i*4+4])/5
        temp.append(tem)
    temp.append(alpha_data[shape-1])
    data =np.array(temp)
    return data
def load_raw_data_pool3(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    load_data = sio.loadmat(raw_path[path])
    load_matrix = load_data['data2']
    shape = load_matrix.shape[0]

    alpha_data = load_matrix[0:shape,count]
    flg = int(shape/500)
    for j in range(0, flg):
        for i in range(0,62):
            tem = (alpha_data[j*500+i*8]+alpha_data[j*500+i*8+1]+alpha_data[j*500+i*8+2]+alpha_data[j*500+i*8+3]+alpha_data[j*500+i*8+4]+
            alpha_data[j*500+i * 8+5] + alpha_data[j*500+i * 8 + 6] + alpha_data[j*500+i * 8 + 7]  )/8
            temp.append(tem)
        flag = (alpha_data[j*500+496]+alpha_data[j*500+497]+alpha_data[j*500+498]+alpha_data[j*500+499])/4
        temp.append(flag)
    data =np.array(temp)
    return data
def load_raw_data_pool4(path,count):
    #path 表示第几个实验者，count表示要处理的通道
    temp = []
    load_data = sio.loadmat(raw_path[path])
    load_matrix = load_data['data2']
    shape = load_matrix.shape[0]
    alpha_data = load_matrix[0:shape,count]
    flg = int(shape / 500)
    for j in range(0, flg):
        for i in range(0, 31):
            tem = (alpha_data[j * 500 + i * 16] + alpha_data[j * 500 + i * 16 + 1] + alpha_data[j * 500 + i * 16 + 2] +
                   alpha_data[j * 500 + i * 16 + 3] + alpha_data[j * 500 + i * 16 + 4] +
                   alpha_data[j * 500 + i * 16 + 5] + alpha_data[j * 500 + i * 16 + 6] + alpha_data[
                       j * 500 + i * 16 + 7] + alpha_data[j * 500 + i * 16+ 8] + alpha_data[j * 500 + i * 16 + 9] +
                   alpha_data[j * 500 + i * 16 + 10] + alpha_data[j * 500 + i * 16 + 11] + alpha_data[
                       j * 500 + i * 16 + 12] +
                   alpha_data[j * 500 + i * 16 + 13] + alpha_data[j * 500 + i * 16 + 14]+ alpha_data[j * 500 + i * 16 + 15]) / 16
            temp.append(tem)
        flag = (alpha_data[j * 500 + 496] + alpha_data[j * 500 + 497] + alpha_data[j * 500 + 498] + alpha_data[
            j * 500 + 499]) / 4
        temp.append(flag)

    data = np.array(temp)
    return data



