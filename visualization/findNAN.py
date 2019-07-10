import scipy.io as sio
import numpy as np
import pandas as pd
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






def load_raw_data(path):
    #path 表示第几个实验者，count表示要处理的通道
    load_data = sio.loadmat(raw_path[path])
    load_matrix = load_data['data2']
    shape = load_matrix.shape[0]
    print(shape)
    alpha_data = load_matrix[:]
    data =np.array(alpha_data)
    np.where(np.isnan(data))
    return data
for i in range(15):
    load_raw_data(i)

