#coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from vae_data import *

#use the default graph
tf.reset_default_graph()
batch_size = 64
n_latent = 100


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image[:,:,0]

    return img

def Encoder(image_flat,is_training, reuse=False):

    with tf.variable_scope("encoder", reuse=reuse):

        x = tf.reshape(image_flat, shape=[-1, 100, 100, 1])
        x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='same',name='en_conv1')
        x = lrelu(x)

        x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same',name='en_conv2')
        x=tf.layers.batch_normalization(x,training=is_training,name='en_bn1')
        x = lrelu(x)

        x = tf.contrib.layers.flatten(x)

        x=tf.layers.dense(x,units=1024,name='en_fc1')
        x=tf.layers.batch_normalization(x,training=is_training,name='en_bn2')
        x=lrelu(x)

        z_mu = tf.layers.dense(x, units=n_latent,name='en_fc2_mu')
        z_sig = tf.layers.dense(x, units=n_latent,name='en_fc2_sig')

        return z_mu, z_sig


def Decoder(z_in,is_training,reuse=False):

    with tf.variable_scope("decoder", reuse=reuse):
        x=tf.layers.dense(z_in,units=1024,name='de_fc1')
        x=tf.layers.batch_normalization(x,training=is_training,name='de_bn1')
        x=tf.nn.relu(x)

        x = tf.layers.dense(x, units=25*25*128,name='de_fc2')
        x=tf.layers.batch_normalization(x,training=is_training,name='de_bn2')
        x=tf.nn.relu(x)

        x = tf.reshape(x,[-1,25,25,128])

        x=tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='same',name='de_deconv1')
        x=tf.layers.batch_normalization(x,training=is_training,name='de_bn3')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=5, strides=2, padding='same',name='de_deconv2')
        x=tf.nn.sigmoid(x)
        return x

#train
def build_model():
    """ Graph Input """
    IMG=tf.placeholder(tf.float32,[None,100*100])
    is_training=tf.placeholder(tf.bool)#1

    # train data
    imgpath = '/home/lu/cs/DCASE_NEW/DATA/data1/newtrain1/'
    img_arr, lab_arr = GetFlatPix(imgpath)  # (17750,10000),(17750,15)
    n_train = img_arr.shape[0]

    '''encoding'''
    # mu, sigma = Encoder(IMG, is_training=True,reuse=False)#1
    mu, sigma = Encoder(IMG, is_training=is_training,reuse=False)#2

    '''sampling by re-parameterization technique'''
    eps = tf.random_normal(shape=tf.shape(mu))#defaut 0-1,float32
    z = mu + tf.exp(sigma / 2) * eps

    '''decoding'''
    # out_img = Decoder(z,is_training=True,reuse=False)#1
    out_img = Decoder(z, is_training=is_training, reuse=False)#2
    out_flat=tf.reshape(out_img,[-1,100*100])
    '''vae_loss'''
    recon_loss = tf.reduce_sum(tf.squared_difference(out_flat, IMG), 1)#均方差loss
    # recon_loss = -tf.reduce_sum(IMG * tf.log(1e-8 + out_flat)         #交叉熵误差
    #             + (1 - IMG) * tf.log(1e-8 + 1 - out_flat), axis=1)


    kl_loss = 0.5 * tf.reduce_sum(tf.exp(sigma)- 1.- sigma +tf.square(mu), 1)
    vae_loss = tf.reduce_mean(recon_loss + kl_loss)

    """ Training """
    # optimizers
    t_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer= tf.train.AdamOptimizer(1e-3).minimize(vae_loss, var_list=t_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 如果当前的路径下不存在out这个文件夹，就创建它
    if not os.path.exists('results_bn/'):
        os.makedirs('results_bn/')


    for step in range(100000):

        start = (step * batch_size) % n_train
        end = min(start + batch_size, n_train)
        _, total_loss, RE_loss, KL_loss = sess.run([optimizer, vae_loss, recon_loss, kl_loss],
         feed_dict={IMG: img_arr[start:end],
                    is_training:True})


        if step % 1000 == 0:  # train
            print('step:%d,Total_Loss:%f,RE_loss:%f,KL_loss:%f' %
                  (step, total_loss, np.mean(RE_loss), np.mean(KL_loss)))

            # test
            """" Testing """
            # # Sampling from random z 训练好了之后，把decoder单独拿出来，把z丢给他
            ZN = tf.placeholder(tf.float32, [None, n_latent])
            # fake_images = Decoder(ZN,is_training=False,reuse=True)#1
            fake_images = Decoder(ZN, is_training=is_training, reuse=True)#2

            z_batch=np.random.normal(0., 1, (batch_size, n_latent)).astype(np.float32)
            samples = sess.run(fake_images, feed_dict={ZN: z_batch,is_training:False})
            samples_img=np.reshape(samples,(-1,100,100,1))
            from scipy.misc import imsave as ims
            # 显示生成的图片
            ims("./results_bn/" + str(step) + ".jpg", merge(samples_img[0:batch_size], [8, 8]))


build_model()
