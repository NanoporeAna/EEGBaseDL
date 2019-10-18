import scipy.io as sio
import numpy as np

"""
这个脚本是用来获取整个数据集的产生的5000*8 矩阵大小的 batch
test_batch()
return (1857,15000,9)的样本集和（1857）便签集
"""
with open("H:/SpaceWork/CNN-LSTM/DATA1min") as file_object:
    lines = file_object.readlines()  # 浠庢枃浠朵腑璇诲彇姣忎竴琛岋紝灏嗚幏鍙栫殑鍐呭鏀惧埌list閲?
mat_path = []
for line in lines:
    mat_path.append(line.strip())  # 灏嗘瘡琛屽湴鍧€杩藉姞鍒颁竴涓暟缁勯噷
# print("ok")
with open("H:/SpaceWork/EEG_Work/lable.txt") as file_object:
    lines_lable = file_object.readlines()  # 浠庢枃浠朵腑璇诲彇姣忎竴琛岋紝灏嗚幏鍙栫殑鍐呭鏀惧埌list閲?
lable_value = []
for line in lines_lable:
    lable_value.append(int(line.strip()))  # 灏嗘瘡琛屾爣绛惧€艰拷鍔犲埌涓€涓暟缁勯噷

mat_dictionary = {}  # 鏋勯€犲瓧鍏?
for i in range(0, 15):
    mat_dictionary[mat_path[i]] = lable_value[i]  # 瀛樺叆姣忎釜瀵瑰簲鐨勫€?

def get_lable(load_path):  # 鏍规嵁璺緞寰楀埌鏍囩
    return mat_dictionary[load_path]

length = 15000
channel = 9
def tailor_test_batch():
    load_data0 = sio.loadmat(mat_path[0])  # 鍔犺浇mat鏂囦欢
    load_matrix = load_data0['data2'] # 鎻愬彇鍑鸿鏁版嵁
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
    for i in range(15):#取mat数据的行数,影响速度，直接套用上面已经加载的load_mat:下一步尝试去做
        load = sio.loadmat(mat_path[i])
        load_shape = load['data2']
        l_d =load_shape.shape
        shape.append(l_d[0])
    test_batch = []
    test_label = []
    # test_batch = []
    # test_lable = []

    for i in range(15):
        lo = sio.loadmat(mat_path[i])
        load =lo['data2']
        # print(shape[i]) #打印出每个的shape
        for j in range(0, 14):
            batchx = load[j * length:(j + 1) * length]  # 鍙?28*9鐨勬暟鎹煩闃?
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[i])
            test_batch.append(batch)
            test_label.append(label)

    #  对1号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[0] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix[j * length + i * 1879 * 4:(j + 1) * length + i * 1879 * 4]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[0])
            test_batch.append(batch)
            test_label.append(label)
    #  对2号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[1] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix1[j * length + i * 1879 * 4:(j + 1) * length + i * 1879 * 4]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[1])
            test_batch.append(batch)
            test_label.append(label)
    #  对3号样本进行裁剪增加1倍数据集，平衡数据
    for j in range(int(shape[2] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix2[j * length + i * 1879 * 4:(j + 1) * length + i * 1879 * 4]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[2])
            test_batch.append(batch)
            test_label.append(label)
    #  对4号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[3] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix3[j * length + i * 1879 * 4:(j + 1) * length + i * 1879 * 4]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[3])
            test_batch.append(batch)
            test_label.append(label)
    #  对5号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[4] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix4[j * length + i * 1879 * 4:(j + 1) * length + i * 1879 * 4]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[4])
            test_batch.append(batch)
            test_label.append(label)
    #  对6号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[5] / 30000) - 1):
        for i in range(1, 2):
            batchx = load_matrix5[j * length + i * 1879 * 4:(j + 1) * length + i * 1879 * 4]
            batch = np.reshape(batchx, (length, 9))
            label = get_lable(mat_path[5])
            test_batch.append(batch)
            test_label.append(label)
    #  对7号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[6] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix6[j * length + i * 1879 * 4:(j + 1) * length + i * 1879 * 4]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[6])
            test_batch.append(batch)
            test_label.append(label)
    #  对8号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[7] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix7[j * length + i * 1879 * 4:(j + 1) * length + i * 1879 * 4]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[7])
            test_batch.append(batch)
            test_label.append(label)
    #  对9号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[8] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix8[j * length + i * 3879 * 2:(j + 1) * length + i * 3879 * 2]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[8])
            test_batch.append(batch)
            test_label.append(label)
    # 对10号样本进行裁剪增加1倍数据集，平衡数据
    for j in range(int(shape[9] / length /2) - 1):
        for i in range(1, 3):
            batchx = load_matrix9[j * length + i * 1709 * 2:(j + 1) * length + i * 1709 * 2]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[9])
            test_batch.append(batch)
            test_label.append(label)
    # 对11号样本进行裁剪增加3倍数据集，平衡数据
    for j in range(int(shape[10] / length /2) - 1):
        for i in range(1, 7):
            batchx = load_matrix10[j * length + i * 1573 * 2:(j + 1) * length + i * 1573 * 2]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[10])
            test_batch.append(batch)
            test_label.append(label)
    #  对12号样本进行裁剪增加10倍数据集，平衡数据
    for j in range(int(shape[11] / length /2) - 1):
        for i in range(1, 6):
            batchx = load_matrix11[j * length + i * 2879 * 2:(j + 1) * length + i * 2879 * 2]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[11])
            test_batch.append(batch)
            test_label.append(label)
    #  对13号样本进行裁剪增加2倍数据集，平衡数据
    for j in range(int(shape[12] / length /2)):
        for i in range(1, 5):
            batchx = load_matrix12[j * length + i * 2709 * 2:(j + 1) * length + i * 2709 * 2]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[12])
            test_batch.append(batch)
            test_label.append(label)
    #  对14号样本进行裁剪增加0.5倍数据集，平衡数据
    for j in range(int(shape[13] / length /2) - 1):
        for i in range(1, 2):
            batchx = load_matrix13[j * length + i * 4879 * 2:(j + 1) * length + i * 4879 * 2]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[13])
            test_batch.append(batch)
            test_label.append(label)
    #  对15号样本进行裁剪增加6倍数据集，平衡数据
    for j in range(int(shape[14] / length /2)):
        for i in range(1, 13):
            batchx = load_matrix14[j * length + i * 1234:(j + 1) * length + i * 1234]
            batch = np.reshape(batchx, (length, channel))
            label = get_lable(mat_path[14])
            test_batch.append(batch)
            test_label.append(label)

    test_batch = np.array(test_batch)
    test_label = np.array(test_label)
    state = np.random.get_state()  # 打乱数据
    np.random.shuffle(test_batch)
    np.random.set_state(state)
    np.random.shuffle(test_label)
    return test_batch,test_label


x, y = tailor_test_batch()
print(x.shape)







