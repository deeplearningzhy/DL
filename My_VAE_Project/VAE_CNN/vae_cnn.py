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

def Encoder(image_flat, reuse=False):

    with tf.variable_scope("Encoder", reuse=reuse):

        x = tf.reshape(image_flat, shape=[-1, 100, 100, 1])
        x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='same')
        x = lrelu(x)
        x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same')
        x = lrelu(x)

        x = tf.contrib.layers.flatten(x)

        z_mu = tf.layers.dense(x, units=n_latent)
        z_sig = tf.layers.dense(x, units=n_latent)

        return z_mu, z_sig


def Decoder(z_in,reuse=False):

    with tf.variable_scope("Decoder", reuse=reuse):


        x = tf.layers.dense(z_in, units=25*25*128)
        x = tf.reshape(x,[-1,25,25,128])
        x=tf.nn.relu(x)

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=5, strides=2, padding='same')
        x=tf.nn.sigmoid(x)
        return x

#train
def build_model():
    """ Graph Input """
    IMG=tf.placeholder(tf.float32,[None,100*100])

    # train data
    imgpath = '/home/lu/cs/DCASE_NEW/DATA/data1/newtrain1/'
    img_arr, lab_arr = GetFlatPix(imgpath)  # (17750,10000),(17750,15)
    n_train = img_arr.shape[0]

    '''encoding'''
    mu, sigma = Encoder(IMG, reuse=False)
    '''sampling by re-parameterization technique'''
    eps = tf.random_normal(shape=tf.shape(mu))#defaut 0-1,float32
    z = mu + tf.exp(sigma / 2) * eps

    '''decoding'''
    out_img = Decoder(z,reuse=False)
    out_flat=tf.reshape(out_img,[-1,100*100])
    '''vae_loss'''
    recon_loss = tf.reduce_sum(tf.squared_difference(out_flat, IMG), 1)
    kl_loss = 0.5 * tf.reduce_sum(tf.exp(sigma)- 1.- sigma +tf.square(mu), 1)
    vae_loss = tf.reduce_mean(recon_loss + kl_loss)

    """ Training """
    # optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(vae_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 如果当前的路径下不存在out这个文件夹，就创建它
    if not os.path.exists('results/'):
        os.makedirs('results/')


    for step in range(100000):

        start = (step * batch_size) % n_train
        end = min(start + batch_size, n_train)
        _, total_loss, RE_loss, KL_loss = sess.run([optimizer, vae_loss, recon_loss, kl_loss],
         feed_dict={IMG: img_arr[start:end]})


        if step % 1000 == 0:  # train
            print('step:%d,Total_Loss:%f,RE_loss:%f,KL_loss:%f' %
                  (step, total_loss, np.mean(RE_loss), np.mean(KL_loss)))

            # test
            """" Testing """
            # # Sampling from random z 训练好了之后，把decoder单独拿出来，把z丢给他
            ZN = tf.placeholder(tf.float32, [None, n_latent])
            fake_images = Decoder(ZN,reuse=True)

            z_batch=np.random.normal(0., 1, (batch_size, n_latent)).astype(np.float32)
            samples = sess.run(fake_images, feed_dict={ZN: z_batch})
            samples_img=np.reshape(samples,(-1,100,100,1))
            from scipy.misc import imsave as ims
            # 显示生成的图片
            ims("./results/" + str(step) + ".jpg", merge(samples_img[0:batch_size], [8, 8]))


build_model()
