#coding=utf-8
import os
import time
from PIL import Image

import tensorflow as tf

# 将图片裁剪为 128 x 128
OUTPUT_SIZE = 128
# 图片通道数，3 表示彩色
DEPTH = 3


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_path, name):
    """
    Converts s dataset to tfrecords
    """

    rows = 64
    cols = 64
    depth = DEPTH
    # 循环 12 次，产生 12 个 .tfrecords 文件
    for ii in range(12):
        writer = tf.python_io.TFRecordWriter(name + str(ii) + '.tfrecords')
        # 每个 tfrecord 文件有 16384 个图片
        for img_name in os.listdir(data_path)[ii * 16384: (ii + 1) * 16384]:
            # 打开图片
            img_path = data_path + img_name
            img = Image.open(img_path)
            # 设置裁剪参数
            h, w = img.size[:2]
            j, k = (h - OUTPUT_SIZE) / 2, (w - OUTPUT_SIZE) / 2
            box = (j, k, j + OUTPUT_SIZE, k + OUTPUT_SIZE)
            # 裁剪图片
            img = img.crop(box=box)
            # image resize
            img = img.resize((rows, cols))
            # 转化为字节
            img_raw = img.tobytes()
            # 写入到 Example
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'image_raw': _bytes_feature(img_raw)}))
            writer.write(example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    # current_dir = os.getcwd()
    # data_path = current_dir + '/data/img_align_celeba/'
    data_path='/home/lu/cs/DCGAN/img_align_celeba/'
    # name = current_dir + '/data/img_align_celeba_tfrecords/train'
    name='/home/lu/cs/DCGAN/img_align_celeba_tfrecords/train'
    start_time = time.time()

    print('Convert start')
    print('\n' * 2)

    convert_to(data_path, name)

    print('\n' * 2)
    print('Convert done, take %.2f seconds' % (time.time() - start_time))