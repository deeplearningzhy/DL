#coding=utf-8
import tensorflow as tf
import os
import numpy as np

#parameters
IMG_HEIGHT=100
IMG_WIDTH=100
N_CLASSES=15
CHANNELS=1

#get the image_list&label_list
def read_images(new_dir,batch_size):
    img=[]
    lab=[]
    for filename in os.listdir(new_dir):
        img.append(new_dir+filename)#得到了imglist

        temp1=filename.split('_')[-1].split('.')[0]#_classID.jpg,we want to get the classID
        lab.append(temp1)#得到了lablist

    #shuffle
    temp=np.array( [img, lab] )#并成两排（2行4列的numpy数组）
    temp=temp.transpose()#矩阵求转置（现在变成了4行两列的数组，图像与相对应的标签在一行）

    np.random.shuffle(temp)#把上面这个4×2的举证的行打乱顺序
    image_list=list(temp[:, 0])#取出上面这个已经打乱顺序的矩阵的第一列，对应图形
    label_list=list(temp[:, 1])#取出上面这个已经打乱顺序的矩阵的第二列，对应标签
    label_list = [int(i) for i in label_list]
    # return image_list,label_list

#Create Batch
# def get_batch(image_list,label_list,batch_size):
    #Before create batch,you need do something.....
    #(1)convert to tensor
    image = tf.convert_to_tensor(image_list, dtype=tf.string)
    label = tf.convert_to_tensor(label_list, dtype=tf.int32)
    #(2)Build a TF Queue,把img&lab装进Queue,and shuffle data
    image,label=tf.train.slice_input_producer([image,label],shuffle=True)
    #(3)Read image from disk
    image=tf.read_file(image)
    image=tf.image.decode_jpeg(image,channels=CHANNELS)
    #(4)Resize image to a common size
    image=tf.image.resize_images(image,[IMG_HEIGHT,IMG_WIDTH])
    #(5)normalize
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # image=image*1.0/127.5-1.0
    #(6)standardlize
    image = tf.image.per_image_standardization(image)
    #=====Now,start create batch(调用tf.train.batch(),当然你还可以使用tf.train.shuffle_batch)
    image_batch,label_batch=tf.train.batch([image,label],
                                           batch_size=batch_size,
                                           capacity=2000,
                                           num_threads=100)
    #=====batch生成结束，当然你还可以在这里给标签来个one-hot编码======
    # image_batch, label_batch = tf.train.shuffle_batch([image, label],
    #                                                    batch_size=2,
    #                                                    num_threads=64,
    #                                                    capacity=2000,
    #                                                    min_after_dequeue=1000)

    # n_classes=2
    label_batch = tf.one_hot(label_batch, N_CLASSES)
    label_batch = tf.reshape(label_batch, [batch_size, N_CLASSES])
    return image_batch,label_batch


# def read_data():
#     # 数据目录
#     # data_dir = '/home/your_name/TensorFlow/DCGAN/data/mnist'
#
#     # 打开训练数据
#     fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
#     # 转化成 numpy 数组
#     loaded = np.fromfile(file=fd, dtype=np.uint8)
#     # 根据 mnist 官网描述的数据格式，图像像素从 16 字节开始
#     trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
#
#     # 训练 label
#     fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
#     loaded = np.fromfile(file=fd, dtype=np.uint8)
#     trY = loaded[8:].reshape((60000)).astype(np.float)
#
#     # 测试数据
#     fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
#     loaded = np.fromfile(file=fd, dtype=np.uint8)
#     teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
#
#     # 测试label
#     fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
#     loaded = np.fromfile(file=fd, dtype=np.uint8)
#     teY = loaded[8:].reshape((10000)).astype(np.float)
#
#     trY = np.asarray(trY)
#     teY = np.asarray(teY)
#
#     # 由于生成网络由服从某一分布的噪声生成图片，不需要测试集，
#     # 所以把训练和测试两部分数据合并
#     X = np.concatenate((trX, teX), axis=0)
#     y = np.concatenate((trY, teY), axis=0)
#
#     # 打乱排序
#     seed = 547
#     np.random.seed(seed)
#     np.random.shuffle(X)
#     np.random.seed(seed)
#     np.random.shuffle(y)
#
#     # 这里，y_vec 表示对网络所加的约束条件，这个条件是类别标签，
#     # 可以看到，y_vec 实际就是对 y 的one-hot编码，关于什么是one-hot编码，
#     # 请参考 http://www.cnblogs.com/Charles-Wan/p/6207039.html
#     y_vec = np.zeros((len(y), 10), dtype=np.float)
#     for i, label in enumerate(y):
#         y_vec[i,int(y[i]) ] = 1.0
#         # y = trY[:70000]
#         # y_vec = one_hot(y, 10)
#
#     return X / 255., y_vec

# 建立
# def one_hot(x,n):
#     if type(x) == list:
#         x = np.array(x)
#         x = x.flatten()
#     o_h = np.zeros((len(x),n))
#     o_h[np.arange(len(x)),x] = 1
#     return o_h
# # 然后将
# y_vec = np.zeros((len(y), 10), dtype=np.float)
# for i, label in enumerate(y):
# y_vec[i,y[i]] = 1.0
# 替换为
# y = trY[:70000]
# y_vec = one_hot(y, 10)
# 即可