import pandas as pd
import matplotlib.pyplot as plot
from pandas import DataFrame
import scipy.io as sio

"""
这里是测试输入数据通道之间的相关性
"""
with open("C:/Users/Nick/PycharmProjects/EEG_Work/raw_data.txt") as file_object:
    lines = file_object.readlines()  # 从文件中读取每一行，将获取的内容放到list里
mat_path = []
for line in lines:
    mat_path.append(line.strip())  # 将每行地址追加到一个数组里
load_data0 = sio.loadmat(mat_path[0])  # 加载mat文件
load_matrix = load_data0['data2'] # 提取出该数据
n = load_matrix.shape[0]
corMat = [] #相关性矩阵
for i in range(0,9):
    for j in range (i,9):
        row2 = load_matrix[0:n,i]
        row3 = load_matrix[0:n,j]
        cor = DataFrame({''+str(i): row2, ''+str(j): row3})  # corr 求相关系数矩阵
        print(cor.corr())
        corMat.append(cor.corr())

# plot.pcolor(corMat)
# plot.show()
# #
