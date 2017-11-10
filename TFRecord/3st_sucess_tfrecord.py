#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage.io as io
from PIL import Image

def get_files(image_dir,txt_dir):
    #首先定义两个空列表用来存放读取出来的image_list和label_list
    image_list=[]
    label_list=[]
    #===获取image_list,start===
    for file in os.listdir(image_dir):#@1:总结os.listdir()&os.walk()两者的区别
        image_list.append(image_dir+file)#@2.这里一定要用加号，否则会报错（总结list的一些操作，原地修改，不可赋值什么的有点忘了）
    #===end===
    #===获取label_list,start===
    with open(txt_dir) as f:
        lst=f.readlines()#@3.总结文件读的三种操作的区别
        for i in lst:
            m=i.split()#@4.总结序列的split操作
            label_list.append(int(m[1]))

    #===end===
    #以上操作就得到了装着每一张图片的列表imagelist和装着相应图片对应的标签labellist
    #为了shufful它们，我们我们把这两个list做了如下操作

    temp=np.array( [image_list, label_list] )#并成两排（2行4列的numpy数组）
    temp=temp.transpose()#矩阵求转置（现在变成了4行两列的数组，图像与相对应的标签在一行）
    np.random.shuffle(temp)#把上面这个4×2的举证的行打乱顺序
    image_list=list(temp[:, 0])#取出上面这个已经打乱顺序的矩阵的第一列，对应图形
    label_list=list(temp[:, 1])#取出上面这个已经打乱顺序的矩阵的第二列，对应标签
    label_list = [int(i) for i in label_list]
    return image_list, label_list

#生成整数型的属性
def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#生成字符串型的属性
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_tfrecord(images,labels,tfrecord_file):
    # tfrecord_file = "/home/user/zhyproject/shiyan/data2/train.tfrecords/"
    n_samples = len(labels)

    # if images.shape[0] != n_samples:  # 这样会报错list has no shape attribute
    # if len(image_list) != n_samples:#这样也会报错
    # if np.shape(images)[0] != n_samples:#原为这样[]显示不正常，有错误，直接注释掉
    #     raise ValueError('Images size %d does not match label size %d.' % (len(image_list), n_samples))

# wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = io.imread(images[i]) #type(image) must be array!
            image_raw = image.tostring()
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

def read_and_decode(tfrecord_file,batch_size):
    #创建一个reader来读取TFrecord文件中的样例
    reader=tf.TFRecordReader()
    #创建一个队列来维护输入文件列表
    filename_queue=tf.train.string_input_producer([tfrecord_file])#注意这里的[]
    #从文件中读出一个样例。也可以使用read_up_to函数一次性读取多个样例
    _,serialized_example=reader.read(filename_queue)
    #解析读入的一个样例。如果需要解析多个样例，可以用parse_example函数
    img_features=tf.parse_single_example(
        serialized_example,
        features={
                #tensorflow提供两种不同的属性解析方法。
                #一种是tf.FixedLenfeature,这种方法解析得到的结果为一个tensor.
                #另一种是tf.VarLenfeature,这种方法解析得到的结果为SparseTensor,用于处理稀疏数据。
                #这里解析数据的格式需要和上面程序写入数据的格式一致
                'label':tf.FixedLenFeature( [],tf.int64 ),
                'image_raw': tf.FixedLenFeature( [],tf.string)
        }
    )
    #tf.decode_raw可以将字符串解析为图像对应的像素数组
    image=tf.decode_raw(img_features['image_raw'],tf.uint8)
    image=tf.reshape(image,[100,100])
    image=tf.cast(image,tf.float32)#这句话是我自己加的
    label=tf.cast(img_features['label'],tf.int32)#和上面那个tf.decode_raw相对应
    image_batch,label_batch=tf.train.batch([image,label],
                                            batch_size=batch_size,
                                            num_threads=64,#这个数的设置是不是要和batch_size一致啊
                                            capacity=2000)#这个数的设置又要有什么规格呢)
    #为什么要给label_batch来一个reshape呢
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch



 # =============抽取一个batch================
    # plot one batch size
def plot_images(images, labels):
    for i in np.arange(0, batch_size):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(chr(ord('A') + labels[i] - 1), fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

if __name__ == "__main__":
    # 网络参数
    batch_size = 16#注意这里的batch_size的范围在[1,num_sample]之间，超过报错
    # 各种路径
    image_dir = "/home/user/zhyproject/shiyan/data2/train/"
    label_dir = "/home/user/zhyproject/shiyan/data2/txt/train.txt"
    tfrecord_file = "/home/user/zhyproject/shiyan/data2/train.tfrecords/"
    # 开始工作
    # @1
    image_list, label_list = get_files(image_dir, label_dir)
    # @2(记住它无返回值，是的，别人写的代码也是没有)
    convert_to_tfrecord(image_list, label_list,tfrecord_file)
    # @3
    image_batch, label_batch = read_and_decode(tfrecord_file, batch_size)




# 启动会话
    with tf.Session() as sess:
        i = 0
        # 下面这两个经常连在一起
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i < 1:
                #just plot one batch size
                # 下面这句话非常重要
                image, label = sess.run([image_batch, label_batch])
                plot_images(image, label)
                i += 1
        except tf.errors.OutOfRangeError:
            print("done!")
        finally:
            coord.request_stop()
        coord.join(threads)
