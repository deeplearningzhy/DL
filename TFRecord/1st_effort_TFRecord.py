#!/usr/bin/env python
#coding=utf-8
import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


tfrecords_filename = '/home/user/zhyproject/shiyan/data3/train.tfrecords'


# 制作TFRecord格式
def createTFRecord(filename, mapfile):
    tfrecords_filename = '/home/user/zhyproject/shiyan/data3/train.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for name in os.listdir(filename):
        img_path=filename+name
        img = Image.open(img_path)
        img = img.resize((100, 100))
        img_raw = img.tobytes()  # 将图片转化成二进制格式


        label_list = []
        with open(mapfile) as f:
            lst1 = f.readlines()
            for i in lst1:
                m = i.split()
                n=int(m[1])
                label_list.append(n)
            # print label_list

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(n),
            'image_raw': _bytes_feature(img_raw)#括号里面要填图片
        }))
        writer.write(example.SerializeToString())
    writer.close()

# 读取train.tfrecord中的数据
def read_and_decode(tfrecords_filename):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer([tfrecords_filename], shuffle=False, num_epochs=1)
    # 从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
    _, serialized_example = reader.read(filename_queue)

    # 解析读入的一个样例，如果需要解析多个，可以用parse_example
    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([], tf.string), })
    # 将字符串解析成图像对应的像素数组
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [100, 100, 1])  # reshape为128*128*3通道图片
    img = tf.image.per_image_standardization(img)
    labels = tf.cast(features['label'], tf.int32)
    return img, labels

#将图片几个一打包，形成一个batch
def createBatch(tfrecords_filename, batchsize):
    images, labels = read_and_decode(tfrecords_filename)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batchsize

    image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                          batch_size=batchsize,
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue
                                                          )

    # label_batch = tf.one_hot(label_batch, depth=2)
    return image_batch, label_batch



if __name__ == "__main__":
    # 训练图片两张为一个batch,进行训练，测试图片一起进行测试
    mapfile = '/home/user/zhyproject/shiyan/data3/record.txt/train.record.txt'
    train_filename = '/home/user/zhyproject/shiyan/data3/train/'
    tfrecords_filename = '/home/user/zhyproject/shiyan/data3/train.tfrecords'

    createTFRecord(train_filename,mapfile)
    # test_filename = "/home/wc/DataSet/traffic/testTFRecord/test.tfrecords"
    #     createTFRecord(test_filename,mapfile)
    image_batch, label_batch = createBatch(tfrecords_filename, batchsize=100)
    # test_images, test_labels = createBatch(filename=test_filename, batchsize=20)
    with tf.Session() as sess:
        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(initop)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while 1:
                _image_batch, _label_batch = sess.run([image_batch, label_batch])
                step += 1
                print step
                print (_label_batch)
        except tf.errors.OutOfRangeError:
            print (" trainData done!")

        # try:
        #     step = 0
        #     while 1:
        #         _test_images, _test_labels = sess.run([image_batch, label_batch])
        #         step += 1
        #         print step
        #         #                 print _image_batch.shape
        #         print (_test_labels)
        # except tf.errors.OutOfRangeError:
        #     print (" TEST done!")
        finally:
            coord.request_stop()
            coord.join(threads)

