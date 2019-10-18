import scipy.io as sio
import numpy as np
with open("H:/SpaceWork/CNN-LSTM/DATA1min") as file_object:
    lines = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
mat_path = []
for line in lines:
    mat_path.append(line.strip())  # 将每行地址追加到一个数组里
raw_path=[]
with open("H:/SpaceWork/CNN-LSTM/lable.txt") as file_object:
    lines_lable = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
lable_value = []
for line in lines_lable:
    lable_value.append(int(line.strip()))  # 将每行标签值追加到一个数组里

mat_dictionary = {}  # 构造字典
for i in range(0, 15):
    mat_dictionary[mat_path[i]] = lable_value[i]  # 存入每个对应的值

def get_lable(load_path):  # 根据路径得到标签
    return mat_dictionary[load_path]
length = 15000
###怎么将mat数据分成单个500*9的数据矩阵，将128*9矩阵放到一个batch里##
def raw_test_batch1min(num):
    #取num位实验者数据
    test_batch = []
    test_label = []
    load_data0 = sio.loadmat(mat_path[num])  # 鍔犺浇mat鏂囦欢
    load_matrix = load_data0['data2']  # 鎻愬彇鍑鸿鏁版嵁
    shape = load_matrix.shape[0]
    for i in range(0, int(shape/length)):#存储第一个人的数据，将其作为测试集
        batch = load_matrix[i * length:(i + 1) * length]  # 取500*9的数据矩阵
        label = get_lable(mat_path[num])
        test_batch.append(batch)  # 取得的矩阵追加到list里
        test_label.append(label)
    test_batch = np.array(test_batch)
    test_label = np.array(test_label)
    return test_batch,test_label
# for i in range(15):
#     data,_ = raw_test_batch1min(i)
#     print(data.shape)