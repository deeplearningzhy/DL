#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# from scipy import ndimage
import skimage.io as io
from PIL import Image





def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    # bus = []
    # label_bus = []
    # beach = []
    # label_beach = []

    image_list=[]
    label_list=[]
    for file in os.listdir(file_dir):
        image_list.append(file_dir+file)
    # print image_list
    txtfile="/home/user/zhyproject/shiyan/data3/record.txt/train.record.txt"
    with open(txtfile) as f:
        lst1=f.readlines()
        for i in lst1:
            m=i.split()
            label_list.append(int(m[1]))
        # print label_list




    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# %%

def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)

    # if np.shape(images)[0] != n_samples:
    if images.shape[0] != n_samples:# 我修改了一下
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i])  # type(image) must be array!
            # image=Image.open(images[i])#我把上面的改成了这个
            image_raw = image.tostring()
            # image_raw=image.tobytes()#我将上面这个也改了，将图片转化成二进制格式
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(label),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' % e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')


# %%

def read_and_decode(tfrecords_file, batch_size):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 1D tensor - [batch_size]
    '''
    # make an input queue from the tfrecord file
    #创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    #创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer([tfrecords_file])
    #从文件中读出一个样例，也可以使用read_up_to一次读取多个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例，如果需要解析多个，可以用parse_example
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.

    image = tf.reshape(image, [100, 100])
    image=tf.cast(image,tf.float32)
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=2000)


    label_batch=tf.reshape(label_batch, [batch_size])
    ##==iadd
    # image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch




# #Convert data to TFRecord
# #=======================start===============================
# #下面是测试上面的代码是否可行
#
# BATCH_SIZE = 10
# file_dir="/home/user/zhyproject/shiyan/data2/train/"
# save_dir="/home/user/zhyproject/shiyan/data2/"
#
#
# name = 'train'
# images, labels = get_files(file_dir)
# convert_to_tfrecord(images, labels, save_dir, name)
#
#
# # %% TO test train.tfrecord file
#
# def plot_images(images, labels):
#     '''plot one batch size
#     '''
#     for i in np.arange(0, BATCH_SIZE):
#         plt.subplot(5, 5, i + 1)
#         plt.axis('off')
#         plt.title(chr(ord('A') + labels[i] - 1), fontsize=14)
#         plt.subplots_adjust(top=1.5)
#         plt.imshow(images[i])
#     plt.show()
#
#
#
# tfrecords_file='/home/user/zhyproject/shiyan/data2/train.tfrecords'
# image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
#
# with tf.Session()  as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         while not coord.should_stop() and i < 1:
#             # just plot one batch size
#             image, label = sess.run([image_batch, label_batch])
#             plot_images(image, label)
#             i += 1
#
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)
# #
# # #==================================end====================================

