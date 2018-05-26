#coding=utf-8
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
#Image Parameters
N_CLASSES=15
IMG_H=100
IMG_W=100
CHANNELS=1

#Reading the dataset
#Mode="folder"
#img_dir=/home/lu/cs/DCASE_NEW/DATASET/0/
def read_images(img_dir):
    label=0
    img_list=[]
    lab_list=[]
    for imgfile in os.listdir(img_dir):
        img_list.append(img_dir+imgfile)
        lab_list.append(label)
    return img_list,lab_list

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
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # image=image*1.0/127.5-1.0
    # (6)standardlize
#     image = tf.image.per_image_standardization(image)
    # =====Now,start create batch(调用tf.train.batch(),当然你还可以使用tf.train.shuffle_batch)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              capacity=2000,
                                              num_threads=100)

    return image_batch, label_batch

#测试代码
batch_size=1
img_dir="D:\\deep_keyan\\homework\\2018_3_28\\img_test\\"
img_list,lab_list=read_images(img_dir)
image_batch, label_batch=get_batch(img_list,lab_list,batch_size)
with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i<1:

            img, label = sess.run([image_batch, label_batch])
            img1=img.transpose((0,2,1,3))
            img2=np.reshape(img1,(batch_size,100,100))
            print(img1)
            print (img2)
            # just test one batch
            for j in np.arange(batch_size):
                print('label: %d' % label[j])
                plt.imshow(img2[j,:,:],cmap ='gray')
                # plt.show()
                plt.savefig('aa.jpg')
            i+=1

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
