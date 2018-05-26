#coding=utf-8
import tensorflow as tf
import os
import numpy as np
#parameters
IMG_HEIGHT=100
IMG_WIDTH=100
BATCH_SIZE=100#batch_size在生成batch的时候已经作为了函数参数
LR=1e-4
N_CLASSES=15
CHANNELS=1

#get the image_list&label_list
def read_images(new_dir):
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
    return image_list,label_list

#Create Batch
def get_batch(image_list,label_list,batch_size):
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
    label_batch = tf.one_hot(label_batch, N_CLASSES)
    label_batch = tf.reshape(label_batch, [batch_size, N_CLASSES])
    return image_batch,label_batch

#测试一下我们的代码是否有误
#new_dir="/home/lu/cs/CNN/data3/newtrain3/"
#img_list,lab_list=read_images(new_dir)
#img_batch,lab_batch=get_batch(img_list,lab_list,batch_size=BATCH_SIZE)
#print img_list
#print lab_list
#print img_batch
#print lab_batch
# Tensor("batch:0", shape=(10, 100, 100, 1), dtype=float32)
# Tensor("batch:1", shape=(10,), dtype=int32)

# Tensor("batch:0", shape=(10, 100, 100, 1), dtype=float32)
# Tensor("Cast_2:0", shape=(10, 2), dtype=int32)#one-hot
