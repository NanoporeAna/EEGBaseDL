import scipy.io as sio
import numpy as np
'''
该python文件是用来得到影响通道最大的相关性值（从8个隐藏层中挑选出来），结果为1*9的矩阵
'''
path = 'D:/EEG_WorkSpace10/MRS/data_corr/delta_max/'
def delt():
	for i in range(0,15):

		newpath = path + str(i)+ '.mat'
		load_data = sio.loadmat(newpath)
		list = []

		load_matrix = load_data['data'] # 取不同隐藏层后的相关系数
		channel0 =load_matrix[0:9] #
		array0 = np.array(channel0)
		array0 = np.reshape(array0, (-1,9)) #
		temp = []
		for j in range(9):
			emp1 = array0[:, j] # 取第j列的所有数据
			emp1 = np.reshape(emp1, -1)
			max1 = max(emp1, key=abs) #取绝对值最大然后返回原来值
			temp.append(max1)


		temp = np.array(temp)
		temp = np.reshape(temp,(-1,9))

		sio.savemat('D:/EEG_WorkSpace10/MRS/data_corr/delta_map/'+str(i)+'.mat', {'data':temp })
delt()
# newpath = path + str(0)+ '.mat'
# load_data = sio.loadmat(newpath)
# load_matrix = load_data[CNN[0]]
# channel0 =load_matrix[:]
# array0 = np.array(channel0)
# array0 = np.reshape(array0, (-1,9))
# temp = []
# for i in range(9):
# 	emp1  = array0[:,i]
# 	emp1 = np.reshape(emp1,-1)
# 	print(emp1)
# 	max1 = max(emp1, key=abs)
# 	temp.append(max1)
# temp =np.array(temp)
# print(temp)





