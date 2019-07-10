import scipy.io as sio
import numpy as np
#  本来是想将每个文件的路径放到一个txt文件中，通过列表提取每行赋的路径，但一直报错，'' 与""的问题--已解决
with open("H:/SpaceWork/EEG_Work/path.txt") as file_object:
    lines = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
mat_path = []
for line in lines:
    mat_path.append(line.strip())  # 将每行地址追加到一个数组里
# print("ok")
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

#  怎么将mat数据分成单个128*9的数据矩阵，将128*9矩阵放到一个batch里##
def test_batch():
    shape = []
    for i in range(15):  #  取mat数据的行数,影响速度，直接套用上面已经加载的load_mat:下一步尝试去做
        load = sio.loadmat(mat_path[i])
        load_shape = load['data2']
        l_d = load_shape.shape
        shape.append(l_d[0])
    test_batch = []
    test_label = []
    for i in range(15):
        lo = sio.loadmat(mat_path[i])
        load = lo['data2']
        for j in range(0, 500):  #  500*15 =7500 为测试集，剩余的取100000个为训练集
            batch = load[j * 500:(j + 1) * 500]  # 取128*9的数据矩阵
            label = get_lable(mat_path[i])
            test_batch.append(batch)
            test_label.append(label)

    test_batch = np.array(test_batch)
    test_label = np.array(test_label)
    return test_batch, test_label
# x,y = test_batch()
# x = np.reshape(x,(-1,500,9,1))
# re = standardize(x)
# print(re[0,:,:,0])







