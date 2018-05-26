#coding=utf-8
import tensorflow as tf
import numpy as np
import read0
import scipy.misc as sm#与图像处理有关的模块
import matplotlib.pyplot as plt
##Training parameter
BATCH_SIZE=100
EPOCHS=100
lr=0.002


##Network paras
noise_dim=100
IMG_H=100
IMG_W=100
CHANNEL=1
is_training=tf.placeholder(tf.bool)
##################Build network#####################
def input():
    with tf.variable_scope("input"):
        noise_input=tf.placeholder(tf.float32,[BATCH_SIZE,noise_dim],name="noise_input")
        real_image_input=tf.placeholder(tf.float32,[BATCH_SIZE,IMG_H,IMG_W,CHANNEL],name="real_image_input")
        label_input=tf.placeholder(tf.float32,[BATCH_SIZE,15])
        return noise_input,real_image_input,label_input

##LeakyRelu Activation
def leakyrelu(x,alpha=0.2):
    with tf.variable_scope("relu"):
        return 0.5*(1+alpha)*x+0.5*(1-alpha)*abs(x)

##Generator Network:input noise,output image
#注意：BN在训练和测试阶段有不同的表现，所以我们要设置is_training参数，指出是训练还是测试
#reuse???
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=25 * 25 * 128)#7*7*128是mnist图片
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 25, 25, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding='same')
        # Apply tanh for better stability - clip values to [-1, 1].
        x = tf.nn.tanh(x)
        return x

# Discriminator Network判决器网络（两层【卷积网络】）
# Input: Image, Output: Prediction Real/Fake Image（输入的是图片，输出的是这张图片属于真、假的概率）
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        # Flatten
        dim = int(np.prod(x.get_shape()[1:]))
        x = tf.reshape(x, shape=[-1, dim])
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    return x


#保存图片函数-用来保存采样器采样后的图片
def save_images(images,size,path):
    """
    Save the samples images
    :param images:原始sample图片大小,[img_h,img_w,channels]
    :param size: 画布的小格子
    :param path: 画布保存的路径
    :return: 无
    The best size number is
      max(  int(sqrt(image.shape[0])),  int(sqrt(image.shape[1]))+1  )
    Example:
    The batch_size=64,then the size is recommended [8,8]
    The batch_size=32,then the size is recommended [6,6]

    """
    #图片归一化，主要用于生成器输出是tanh形式的归一化
    img=(images+1.0)/2.0
    #原始图片的高和宽
    h,w=img.shape[1],img.shape[2]
    #产生一个大画布，用来保存生成的【batch】个图像
    merge_img=np.zeros( (h*size[0],w*size[1],3) )

    #循环使得画布特定的地方值为某一副图像的值
    for index,a_image in enumerate(images):
        i=index % size[1]
        j=index / size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w, :]=a_image
    #保存画布
    return sm.imsave(path,merge_img)

