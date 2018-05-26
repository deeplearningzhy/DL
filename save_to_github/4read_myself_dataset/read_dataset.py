#coding=utf-8
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
#Image Parameters
N_CLASSES=15
IMG_H=100
IMG_W=100
CHANNELS=1

#Reading the dataset
#Mode="folder"
#dataset_path=/home/lu/cs/DCASE_NEW/DATASET/
def read_images(dataset_path):

    img,lab=list(),list()

    label=0
    #list the directory
    classes=sorted(os.walk(dataset_path).next()[1])#==>['0','1','2',...]

    for c in classes:
        c_dir=os.path.join(dataset_path,c)
        walk=os.walk(c_dir).next()#==>(c_dir,[],[.../img,img])

        #add each image to the training set
        for sample in walk[2]:
            img.append(os.path.join(c_dir,sample))
            lab.append(label)
        label += 1

    ############# shuffle ################
    temp = np.array([img, lab])  # 并成两排（2行4列的numpy数组）
    temp = temp.transpose()  # 矩阵求转置（现在变成了4行两列的数组，图像与相对应的标签在一行）

    np.random.shuffle(temp)  # 把上面这个4×2的举证的行打乱顺序
    image_list = list(temp[:, 0])  # 取出上面这个已经打乱顺序的矩阵的第一列，对应图形
    label_list = list(temp[:, 1])  # 取出上面这个已经打乱顺序的矩阵的第二列，对应标签
    label_list = [int(i) for i in label_list]
    return image_list, label_list

##################################################
# Create Batch
def get_batch(image_list, label_list, batch_size):
    # Before create batch,you need do something.....
    # (1)convert to tensor
    image = tf.convert_to_tensor(image_list, dtype=tf.string)
    label = tf.convert_to_tensor(label_list, dtype=tf.int32)
    # (2)Build a TF Queue,把img&lab装进Queue,and shuffle data
    image, label = tf.train.slice_input_producer([image, label], shuffle=True)
    # (3)Read image from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    # (4)Resize image to a common size
    image = tf.image.resize_images(image, [IMG_H, IMG_W])
    # (5)normalize
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image=image*1.0/127.5-1.0
    # (6)standardlize
    image = tf.image.per_image_standardization(image)
    # =====Now,start create batch(调用tf.train.batch(),当然你还可以使用tf.train.shuffle_batch)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              capacity=2000,
                                              num_threads=100)




    return image_batch, label_batch

###测试代码可行性
batch_size=64
dataset_path="/home/lu/cs/DCASE_NEW/DATASET/"
img_list,lab_list=read_images(dataset_path)
batch_x,batch_y=get_batch(img_list,lab_list,batch_size)
# print(img_list)
# print(lab_list)
print(batch_x)
print(batch_y)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
############################################################
batches=int(len(img_list)/batch_size)
for start, end in zip(range(0, len(img_list), batch_size),
                      range(batch_size, len(img_list), batch_size)
                     ):

    for step in range(batches):
        batch_x=batch_x[start:end]
        batch_y=batch_y[start:end]
        img_batch,lab_batch=sess.run([batch_x,batch_y])
        print(img_batch)
        print(lab_batch)


# for step in range(1,10):
#     img_batch,lab_batch=sess.run([batch_x,batch_y])
#     print(img_batch)
#     print(lab_batch)