import scipy.io as sio
import numpy as np
"""
这个脚本是用来获取整个数据集的产生的5000*8 矩阵大小的 batch
return (11712, 5000, 8)的样本集和（11712）便签集
"""
with open("H:/SpaceWork/CNN-LSTM/raw_data8") as file_object:
	lines = file_object.readlines()  #
mat_path = []
for line in lines:
	mat_path.append(line.strip())
# print("ok")
with open("H:/SpaceWork/CNN-LSTM/lable.txt") as file_object:
	lines_lable = file_object.readlines()
lable_value = []
for line in lines_lable:
	lable_value.append(int(line.strip()))

mat_dictionary = {}
for i in range(0, 15):
	mat_dictionary[mat_path[i]] = lable_value[i]


def get_lable(load_path):
	return mat_dictionary[load_path]

channel = 8
EEG_length = 5000 #10s data

def tailor_train_batch():
	load_data0 = sio.loadmat(mat_path[0])
	load_matrix = load_data0['data2']
	load_data1 = sio.loadmat(mat_path[1])
	load_matrix1 = load_data1['data2']
	load_data2 = sio.loadmat(mat_path[2])
	load_matrix2 = load_data2['data2']
	load_data3 = sio.loadmat(mat_path[3])
	load_matrix3 = load_data3['data2']
	load_data4 = sio.loadmat(mat_path[4])
	load_matrix4 = load_data4['data2']
	load_data5 = sio.loadmat(mat_path[5])
	load_matrix5 = load_data5['data2']
	load_data6 = sio.loadmat(mat_path[6])
	load_matrix6 = load_data6['data2']
	load_data7 = sio.loadmat(mat_path[7])
	load_matrix7 = load_data7['data2']
	load_data8 = sio.loadmat(mat_path[8])
	load_matrix8 = load_data8['data2']
	load_data9 = sio.loadmat(mat_path[9])
	load_matrix9 = load_data9['data2']
	load_data10 = sio.loadmat(mat_path[10])
	load_matrix10 = load_data10['data2']
	load_data11 = sio.loadmat(mat_path[11])
	load_matrix11 = load_data11['data2']
	load_data12 = sio.loadmat(mat_path[12])
	load_matrix12 = load_data12['data2']
	load_data13 = sio.loadmat(mat_path[13])
	load_matrix13 = load_data13['data2']
	load_data14 = sio.loadmat(mat_path[14])
	load_matrix14 = load_data14['data2']
	shape = []
	for i in range(15):  # 取mat数据的行数,影响速度，直接套用上面已经加载的load_mat:下一步尝试去做
		load = sio.loadmat(mat_path[i])
		load_shape = load['data2']
		l_d = load_shape.shape
		shape.append(l_d[0])
	train_batch = []
	train_label = []

	for i in range(15):
		lo = sio.loadmat(mat_path[i])
		load = lo['data2']
		# print(shape[i]) #打印出每个的shape
		for j in range(50,int(shape[i] / EEG_length)):
			batchx = load[j * EEG_length:(j + 1) * EEG_length]  # 鍙?28*9鐨勬暟鎹煩闃?
			batch = np.reshape(batchx, (EEG_length, channel))
			label = get_lable(mat_path[i])
			train_batch.append(batch)
			train_label.append(label)
	# #  对1号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[0] / EEG_length /2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[0])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# #  对2号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[1] / EEG_length/2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix1[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[1])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# #  对3号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[2] / EEG_length/2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix2[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[2])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# #  对4号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[3] / EEG_length/2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix3[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[3])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# #  对5号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[4] / EEG_length/2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix4[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[4])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# #  对6号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[5] / EEG_length/2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix5[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[5])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# #  对7号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[6] / EEG_length/2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix6[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[6])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# #  对8号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[7] / EEG_length/2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix7[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[7])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# #  对9号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[8] / EEG_length/2) - 1):
	# 	for i in range(1, 6):
	# 		batchx = load_matrix8[j * EEG_length + i * 973*2:(j + 1) * EEG_length + i * 973*2]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[8])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	# 对10号样本进行裁剪增加40倍数据集，平衡数据
	for j in range(25,int(shape[9] / EEG_length/2) - 1):
		for i in range(1, 3):
			batchx = load_matrix9[j * EEG_length + i * 3379:(j + 1) * EEG_length + i * 3379]
			batch = np.reshape(batchx, (EEG_length, channel))
			label = get_lable(mat_path[9])
			train_batch.append(batch)
			train_label.append(label)
	# 对11号样本进行裁剪增加50倍数据集，平衡数据
	for j in range(25,int(shape[10] / EEG_length/2) - 1):
		for i in range(1, 5):
			batchx = load_matrix10[j * EEG_length + i * 2379:(j + 1) * EEG_length + i * 2379]
			batch = np.reshape(batchx, (EEG_length, channel))
			label = get_lable(mat_path[10])
			train_batch.append(batch)
			train_label.append(label)
	# #  对12号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[11] / EEG_length/2) - 1):
	# 	for i in range(1, 11):
	# 		batchx = load_matrix11[j * EEG_length + i * 973:(j + 1) * EEG_length + i * 973]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[11])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
	#  对13号样本进行裁剪增加40倍数据集，平衡数据
	for j in range(25,int(shape[12] / EEG_length/2) - 1):
		for i in range(1, 13):
			batchx = load_matrix12[j * EEG_length + i * 879:(j + 1) * EEG_length + i * 879]
			batch = np.reshape(batchx, (EEG_length, channel))
			label = get_lable(mat_path[12])
			train_batch.append(batch)
			train_label.append(label)
	# #  对14号样本进行裁剪增加10倍数据集，平衡数据
	# for j in range(int(shape[13] / EEG_length/2) - 1):
	# 	for i in range(1, 11):
	# 		batchx = load_matrix13[j * EEG_length + i * 973:(j + 1) * EEG_length + i * 973]
	# 		batch = np.reshape(batchx, (EEG_length, channel))
	# 		label = get_lable(mat_path[13])
	# 		train_batch.append(batch)
	# 		train_label.append(label)
#  对15号样本进行裁剪增加120倍数据集，平衡数据
	for j in range(25,int(shape[14] / EEG_length/2) - 1):
		for i in range(1, 17):
			batchx = load_matrix14[j * EEG_length + i * 678:(j + 1) * EEG_length + i * 678]
			batch = np.reshape(batchx, (EEG_length, channel))
			label = get_lable(mat_path[14])
			train_batch.append(batch)
			train_label.append(label)

	train_batch = np.array(train_batch)
	train_label = np.array(train_label)
	state = np.random.get_state()  # 打乱数据
	np.random.shuffle(train_batch)
	np.random.set_state(state)
	np.random.shuffle(train_label)

	return train_batch, train_label

# x, y = tailor_train_batch()
# print(np.shape(x))

