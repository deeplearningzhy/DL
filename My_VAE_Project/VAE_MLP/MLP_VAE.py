#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
import os
from vae_input import *

#超参数
batch_size=64
n_latent=100
img_dim=10000
#train data
imgpath='/home/lu/cs/DCASE_NEW/DATA/data1/newtrain1/'
img_arr,lab_arr=GetFlatPix(imgpath)
n_train=img_arr.shape[0]

# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_latent, keep_prob):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer:x.get_shape()[1]:img_dim
        w0 = tf.get_variable('w0',
                             [x.get_shape()[1], n_hidden],#[img_dim,512]
                             initializer=w_init)

        b0 = tf.get_variable('b0',
                             [n_hidden],#[512]
                             initializer=b_init)

        h0 = tf.matmul(x, w0) + b0

        h0 = tf.nn.elu(h0)#先relu,再dropout
        h0 = tf.nn.dropout(h0, keep_prob)#[512]

        # 2nd hidden layer
        w1 = tf.get_variable('w1',
                             [h0.get_shape()[1], n_hidden],#[512,512]
                             initializer=w_init)

        b1 = tf.get_variable('b1',
                             [n_hidden],#[512]
                             initializer=b_init)

        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        w_out = tf.get_variable('w_out',
                                [h1.get_shape()[1], n_latent],#[512,100]
                                initializer=w_init)

        b_out = tf.get_variable('b_out',
                                [n_latent],#[100]
                                initializer=b_init)
#——————————————方式1————————————————————————————————————
    #     gaussian_params = tf.matmul(h1, w_out) + b_out
    #     # The mean parameter is unconstrained
    #     mean = gaussian_params[:, :n_latent]
    #     # The standard deviation must be positive. Parametrize with a softplus and
    #     # add a small epsilon for numerical stability
    #     stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_latent:])
    # return mean, stddev
#————————————方式2————————————————————————————————————————



        mean = tf.matmul(h1, w_out) + b_out
        variance = tf.matmul(h1, w_out) + b_out



    return mean, variance

# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):

    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0',
                             [z.get_shape()[1], n_hidden],#[100,512]
                             initializer=w_init)

        b0 = tf.get_variable('b0',
                             [n_hidden],#[512]
                             initializer=b_init)

        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1',
                             [h0.get_shape()[1], n_hidden],#[512,512]
                             initializer=w_init)

        b1 = tf.get_variable('b1',
                             [n_hidden],#[512]
                             initializer=b_init)

        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo',
                             [h1.get_shape()[1], n_output],#[512,img_dim]
                             initializer=w_init)

        bo = tf.get_variable('bo',
                             [n_output],#img_dim?
                             initializer=b_init)

        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

#用于训练完之后 测试生成图片
def decoder(z, dim_img, n_hidden):

    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)

    return y

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image[:,:,0]

    return img

# Gateway出入口
def train():

    X_p=tf.placeholder(tf.float32,[None,img_dim])
    # kp=tf.placeholder(tf.float32)
    # encoding(img(x_hat)=>encoder=>mean,variance)
    mu, sigma = gaussian_MLP_encoder(X_p, 512, 100, keep_prob=0.5)

    # sampling by re-parameterization technique
    eps= tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z_composed = mu + tf.exp(sigma/2) * eps

    # decoding(compose_z=>decoder=>img'(y))  #x is real img
    y = bernoulli_MLP_decoder(z_composed, 512, 10000, keep_prob=0.5)#sigmoid出来的y
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)#avoid loss NAN

    # loss
    recon_loss =-tf.reduce_sum(X_p * tf.log(y) + (1 - X_p) * tf.log(1 - y), 1)
    kl_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    recon_loss = tf.reduce_mean(recon_loss)
    kl_loss = tf.reduce_mean(kl_loss)
    loss_op = recon_loss+kl_loss

    #optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss_op)

    #start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists('mlp_result/'):
        os.makedirs('mlp_result/')

    for step in range(100000):

        start = (step * batch_size) % n_train
        end = min(start + batch_size, n_train)
        _, total_loss, RE_loss, KL_loss = sess.run([train_op, loss_op, recon_loss, kl_loss],
                                                   feed_dict={X_p: img_arr[start:end]})

        if step % 1000 == 0:  # train
            print('step:%d,Total_Loss:%f,RE_loss:%f,KL_loss:%f' %
                  (step, total_loss, RE_loss, KL_loss))

            # test
            """" Testing """
            # # Sampling from random z 训练好了之后，把decoder单独拿出来，把z丢给他
            ZN = tf.placeholder(tf.float32, [None, n_latent])
            fake_images = decoder(ZN, 10000,512)

            z_batch = np.random.normal(0., 1, (batch_size, n_latent)).astype(np.float32)
            samples = sess.run(fake_images, feed_dict={ZN: z_batch})
            samples_img = np.reshape(samples, (-1, 100, 100, 1))
            from scipy.misc import imsave as ims
            # 显示生成的图片
            ims("./mlp_result/" + str(step) + ".jpg", merge(samples_img[0:batch_size], [8, 8]))


train()



