import numpy as np
class DataSet(object):

    def __init__(self,EEGData,lable):
        """
        初始化每次调用next_batch属性
        :param EEGData:
        :param lable:
        :param num_example:
        :return:
        """
        self._EEGData = EEGData
        self._lable = lable
        self._num_examples = EEGData.shape[0]
        self._epochs_completed = 0 #完成遍历的轮次数
        self._index_in_epoch = 0  #调用get_next_batch函数后记住上一次的位置
    def next_batch(self,batch_size,fake_data=False,shuffle=True):
        """
        获取下一个batch池块
        第一个epoch怎么处理,
        每个epoch的结尾连接下一个epoch的开头怎么处理，
        非第一个epoch&非结尾怎么处理。
        这样分开，主要是因为每个epoch的开头，都要shuffle index.即将所有数据顺序都打乱
        :param self:指向实例本身的引用，必须有，在开头
        :param batch_size:要取batch的数量
        :param fake_data:虚假数据
        :param shuffle:是否将128个batch打乱顺序
        :return:除去当前取得128个batch余下的数据集
        """
        start =self._index_in_epoch #self._index_in_epoch  所有的调用，总共用了多少个样本，相当于一个全局变量
        # start第一个batch为0，剩下的就和self._index_in_epoch一样，如果超过了一个epoch，在下面还会重新赋值
        #Shuffle for the first epoch 第一个epoch需要shuffle
        #epoch:迭代次数，1个epoch等于使用训练集中的全部样本训练一次；一个epoch = 所有训练样本的一个正向传递和一个反向传递
        if self._epochs_completed == 0 and start ==0 and shuffle:
            index0 = np.arange(self._num_examples)#生成的一个所有样本长度的np.array
            # print(index0)
            np.random.shuffle(index0) #将训练集的数据打乱顺序
            self._EEGData = np.array(self._EEGData)[index0]
            self._lable = np.array(self._lable)[index0]

        ## Go to the next epoch
        if start + batch_size > self._num_examples:#epoch的结尾和下一个epoch的开头
        # Finished epoch
            self._epochs_completed += 1
        # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start  # 最后不够一个batch还剩下几个
            EEGData_rest_part = self._EEGData[start:self._num_examples]
            lable_rest_part = self._lable[start:self._num_examples]
            #shuffle 数据
            if shuffle:
                index = np.arange(self._num_examples)
                np.random.shuffle(index)
                self._EEGData = self._EEGData[index]
                self._lable = self._lable[index]
        # Start next epoch
            start = 0
            self._index_in_epoch = batch_size-rest_num_examples
            end = self._index_in_epoch
            EEGData_new_part = self._EEGData[start:end]
            lable_new_part =self._lable[start:end]
            return np.concatenate((EEGData_rest_part,EEGData_new_part),axis=0),np.concatenate((lable_rest_part,lable_new_part),axis=0)
        else:# 除了第一个epoch，以及每个epoch的开头，剩下中间batch的处理方式
            self._index_in_epoch += batch_size   # start = index_in_epochs
            end =self._index_in_epoch            #end很简单，就是 index_in_epoch加上batch_size
            return self._EEGData[start:end],self._lable[start:end]

# input = ['a', 'b', '1', '2', '*', '3', 'c', '&', '#']
# output = ["Letter", "Letter", "Number", "Number", "Symbol", "Number", "Letter", "Symbol", "Symbol"]
# ds = DataSet(input, output)
# for i in range(3):
#     image_batch, label_batch = ds.next_batch(4)
#     print(image_batch)
#     print(label_batch)
