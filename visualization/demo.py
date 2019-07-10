import scipy.io as sio
import numpy as np


CNN =['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4']
path = 'D://data_corr/delta/'
def delt():
	for i in range(0,15):

		newpath = path + str(i)+ '.mat'
		load_data = sio.loadmat(newpath)
		load_matrix = load_data[CNN[0]]
		channel0 =load_matrix[0:9]
		array0 = np.array(channel0)
		array0 = np.reshape(array0, (-1,9))
		array0[0,3:9] = None
		array0[1, 0] = None
		array0[1,4:9] = None
		array0[2, 0:2] = None
		array0[2, 5:9] = None
		array0[3, 0:3] = None
		array0[3, 6:9] = None
		array0[4, 0:4] = None
		array0[4, 7:9] = None
		array0[5, 0:5] = None
		array0[5, 8] = None
		array0[6, 0:6] = None
		array0[7, 0:7] = None
		array0[8, 0:8] = None

		load_matrix1 = load_data[CNN[1]]
		channel1 = load_matrix1[0:9]
		array1 = np.array(channel1)
		array1 = np.reshape(array1, (-1, 9))
		array1[0, 5:9] = None
		array1[1, 0:2] = None
		array1[1, 7:9] = None
		array1[2, 0:4] = None
		array1[3, 0:6] = None
		array1[4, 0:8] = None

		load_matrix2 = load_data[CNN[2]]
		channel2 = load_matrix2[0:9]
		array2 = np.array(channel2)
		array2 = np.reshape(array2, (-1, 9))
		array2[1, 0:2] = None
		array2[2, 0:4] = None
		array2[3, 0:6] = None
		array2[4, 0:8] = None

		load_matrix3 = load_data[CNN[3]]
		channel3 = load_matrix3[0:9]
		array3 = np.array(channel3)
		array3 = np.reshape(array3, (-1, 9))
		array3[1, 0:4] =None
		array3[2, 0:8] = None

		load_matrix4 = load_data[CNN[4]]
		channel4 = load_matrix4[0:9]
		array4 = np.array(channel4)
		array4 = np.reshape(array4, (-1, 9))
		array4[1, 0:4] = None
		array4[2, 0:8] = None

		load_matrix5 = load_data[CNN[5]]
		channel5 = load_matrix5[0:9]
		array5 = np.array(channel5)
		array5 = np.reshape(array5, (-1, 9))
		array5[1, 0:8] = None

		load_matrix6 = load_data[CNN[6]]
		channel6 = load_matrix6[0:9]
		array6 = np.array(channel6)
		array6 = np.reshape(array6, (-1, 9))
		array6[1, 0:8] = None
		sio.savemat('D://new_data_corr/delta/'+str(i)+'.mat', {CNN[0]:array0, CNN[1]:array1, CNN[2]:array2, CNN[3]:array3, CNN[4]:array4, CNN[5]:array5, CNN[6]:array6 })


delt()




