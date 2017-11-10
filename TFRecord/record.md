record_input.py
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
import skimage.io as io
from PIL import Image


把训练数据转化成tfrecord形式--学习总结
主要分为三步
step1：把trainset里面的image_list和label_list得到。这主要通过get_files()函数实现
step2：把image_list和label_list转化成tfrecord的形式。这主要通过convert_to_tfrecord()函数实现
step3：解析tfrecord文件，并生成image_batch和label_batch
下面主要介绍上面三个函数的实现。
（1）首先介绍下我的train数据集
我选取的是提取了mfs特征的100×100×1的语谱图，共有17750张。
data3里面包含train test txt（train.txt,test.txt）
train,test里面都是图片，txt是对应图片的标签
def get_files(file_dir)
函数参数：
file_dir：dataset file directory
返回参数：
list of images and list of labels
代码实现如下：(我自己写的代码)
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
            label_list.append(int(m[1])
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

#===举个例子,start===
[['a' 'b' 'c' 'd']
 ['1' '2' '3' '4']]

[['a' '1']
 ['b' '2']
 ['c' '3']
 ['d' '4']]

[['d' '4']
 ['a' '1']
 ['c' '3']
 ['b' '2']]

['d', 'a', 'c', 'b']

['4', '1', '3', '2']

[3, 4, 1, 2]
#===end===

(2)在将数据转换成tfrecord之前先介绍一下tfrecord的格式
它将解码前的图片存为字符串string，对应的标签存为整数列表int
故下面这两个函数直接照搬 tensorflow官网
#生成整数型的属性
def _int64_feature(value):
    return tf.tarin.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.ByteList(value=[value]))
#========start：开始将imagelist和labellist里面的东西转化成tfrecord形式===============
#convert all images and labels to one tfrecord file
def convert_to_tfrecord(images,labels,save_dir,name):
函数参数：
image：list of image directories,string type
labels:list of labels,int type
save_dir:the directory to save tfrecord file,eg:'/home/folder1/'
name:the name of tfrecord file,string type,eg:'train'
函数返回值：
No return
convert need some time,be patient...
def convert_to_tfrecord(images,labels,save_dir,name):
    filename=os.path.jion(save_dir,name,+".tfrecords")#这就是tfrecord文件要保存的位置，用os.path.jion(字符串的拼接)
    n_samples=len(labels)#数出样本一共有多少个

    #if np.shape(images)[0]!=n_samples#这个有点问题，尝试下如果不要这句话会不会有影响
    if images.shape[0] != n_samples:# 我修改了一下
        raise valueError('Images size %d does not match label size %d.') % (images.shape[0],n_samples)

    #(wait some time here,transforing,transforing need some time based on the size of the time)
    #定义一个writer，好把下面这些东西存到给定的filename（也就是tfrecord文件中）
    writer=tf.python_io.TFRecordWriter(filename)
    print("\nTrainsform start......")
    for i in np.arange(0,n_samples):
        try:
            #===这个过程就是把图像转化成像素数组,start===
            image=io.imread(images[i])#type(image) must be array
            image_raw=image.tostring#将图片转化成一个字符串
            label=int(labels[i])
            #===end===
            #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
            example=tf.train.Example(features=tf.train.Features(features={
                'label'=int64_feature(label),
                'image_raw':bytes_feature(image_raw)

            }))
            #将这个EXample写入TFrecord文件
            writer.write(example.SerializeToString())
        except IOError as e:
            print("Could not read:",images[i])
            print('error: %s' % e)
            print('Skip it!\n')
    write.close()
    print("Transform done!")
以上程序可以将数据集中的所有训练数据存储到一个TFRecord 文件中。
当数据量较大时，也可以将数据写入多个TFrecord文件。
tensorflow对从文件列表中读取数据提供了很好的支持

****************************************************************************
这个函数我来改造一下
def convert_to_tfrecord(images,labels,save_dir,name):
我来把这个函数的参数改一下
#尝试1:def convert_to_tfrecord(images,labels,tfrecord_file):
#尝试2:def convert_to_tfrecord(images,labels):
images还是图像列表，
labels还是标签列表，
tfrecord_file是tfrecord格式文件存放的路径
filename = os.path.join(save_dir, name + '.tfrecords')
上面这句话直接改成下面这样：
tfrecord_file="/这里填你的tfrecord文件想要存放的地方/"

n_samples = len(labels)
# if np.shape(images)[0] != n_samples:#这段话是否可以不要
if images.shape[0] != n_samples:# 我修改了一下上面的这句话
    raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

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
****************************************************************************************************************


(3)下面这个步骤给出了如何读取TFrecord文件中的数据,并生成batch
#读取数据集（读取二进制数据）
def read_and_decode(tfrecords_file,batch_size)
#read and decode tfrecord file.generate(image,label)batches
函数参数：
tfrecord_file:the directory of tfrecord file
batch_size:number of images in each batch
返回值：
image：4D tensor-[batch_size,width,height,channel]
label:1D tensor-[batch_size]

def read_and_decode(tfrecords_file,batch_size)
    #创建一个reader来读取TFrecord文件中的样例
    reader=tf.TFRecordReader()
    #创建一个队列来维护输入文件列表
    filename_queue=tf.train.string_input_producer([tfrecords_filename])#注意这里的[]
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
                'label':tf.FixedLenFeature( [],tf.int64 )
                'image_raw': tf.FixedLenfeature( [],tf.string)
        }
    )
    #tf.decode_raw可以将字符串解析为图像对应的像素数组
    image=tf.decode_raw(img_features['image_raw'],tf.uint8)
    image=tf.reshape(image,[100,100])
    image=tf.cast(image,tf.float32)#这句话是我自己加的
    label=tf.cast(img_features['label'],tf.int32)#和上面那个tf.decode_raw相对应
    image_batch,label_batch=tf.train.batch([image,label],
                                            batch_size=batch_size,
                                            num_threads=64#这个数的设置是不是要和batch_size一致啊
                                            capacity=2000#这个数的设置又要有什么规格呢)
    #为什么要给label_batch来一个reshape呢
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
    
至此，tensorflow高效从文件读取数据差不多完结了。但是我们通常会在这个文件这里做一些测试，以来间样上面的代码是否可行。
#=====================test=========================
if __name__=="__main__":
    #网络参数
    batch_size=100
    #各种路径
    image_dir="/这里是存放训练集图片的地址/"
    label_dir="/这里是存放训练图片标签的地址/"
    tfrecord_file="/这里是存放tfrecord图片的地址/"
    #开始工作
    #@1
    image_list,label_list=get_files(image_dir,label_dir)
    #@2(记住它无返回值，是的，别人写的代码也是没有)
    convert_to_tfrecord(image_list,label_list)
    #@3
    image_batch,label_batch=read_and_decode(tfrecord_file,batch_size)
    #=============抽取一个batch================
    #plot one batch size
    def plot_images(images,labels):
        for i in np.arange(0, BATCH_SIZE=batch_size):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(chr(ord('A') + labels[i] - 1), fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()
    
#启动会话
with tf,session() as sess:
    i=0
    #下面这两个经常连在一起
    coord=tf.train.Coodinator()
    threads=tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i<1:
        #just plot one batch size
        #下面这句话非常重要
        image,label=sess.run([image_batch,label_batch])
        plot_images(image,label)
        i+=1
    except tf.error.OutOfRangeError:
        print("done!")
    finally:
        coord.request_stop()
    coord.join(threads)
    
    










