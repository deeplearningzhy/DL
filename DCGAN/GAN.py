#coding=utf-8
import tensorflow as tf
import os
import time
import numpy as np
from dcgan_data import *

#超参数
batch_size=100
z_dim=100
MODEL_DIRECTORY = "dcgan_model/model.ckpt"
LOGS_DIRECTORY = "dcgan_model/train"


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image[:,:,0]

    return img

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def discriminator(img, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    with tf.variable_scope("discriminator", reuse=reuse):
        #conv1
        # x=tf.reshape(img_flat,[-1,100,100,1])#[batch,h,w,c]
        x = tf.layers.conv2d(img, filters=64, kernel_size=4, strides=2, padding='same', name='dis_conv1')
        x = lrelu(x)
        # conv2
        x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same', name='dis_conv2')
        x = tf.layers.batch_normalization(x, training=is_training, name='dis_bn1')
        x = lrelu(x)
        #flatten
        x = tf.contrib.layers.flatten(x)
        #fc1
        x =tf.layers.dense(x,units=1024,name='dis_fc1')
        x=tf.layers.batch_normalization(x,training=is_training,name='dis_bn2')
        x=lrelu(x)
        #fc2
        x_out=tf.layers.dense(x,units=1,name='dis_fc2')
        x_pred=tf.nn.sigmoid(x_out)
    return x_out,x_pred


def generator(z_in, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope("generator", reuse=reuse):
        #FC1
        z=tf.layers.dense(z_in,units=1024,name='gen_fc1')
        z=tf.layers.batch_normalization(z,training=is_training,name='gen_bn1')
        z=tf.nn.relu(z)
        #FC2
        z=tf.layers.dense(z,units=25*25*128,name='gen_fc2')
        z=tf.layers.batch_normalization(z,training=is_training,name='gen_bn2')
        z=tf.nn.relu(z)
        #reshape
        z=tf.reshape(z,[-1,25,25,128])
        #deconv1
        z=tf.layers.conv2d_transpose(z,filters=64,kernel_size=4,strides=2,padding='same',name='gen_deconv1')
        z=tf.layers.batch_normalization(z,training=is_training,name='gen_bn3')
        z=tf.nn.relu(z)
        #deconv2
        z=tf.layers.conv2d_transpose(z,filters=1,kernel_size=4,strides=2,padding='same',name='gen_deconv2')
        z=tf.nn.sigmoid(z)
    return z




def build_model():
    """ Graph Input """
    IMG=tf.placeholder(tf.float32,[None,10000],name='real_img')
    IMG_matrix=tf.reshape(IMG,[-1,100,100,1],name='img_matrix')
    ZN=tf.placeholder(tf.float32,[None,100],name='noise')

    '''train data'''
    imgpath = '/home/lu/cs/DCASE_NEW/DATA/data1/newtrain1/'
    img_arr, lab_arr = GetFlatPix(imgpath)  # (17750,10000),(17750,15)
    n_train = img_arr.shape[0]

    """ Loss Function """
    G_sample=generator(ZN,is_training=True,reuse=False)#

    D_real_out,D_real_pred=discriminator(IMG_matrix,is_training=True,reuse=False)

    D_fake_out,D_fake_pred=discriminator(G_sample,is_training=True,reuse=True)

    # get loss for discriminator(这里用的是sigmoid function)
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_out, labels=tf.ones_like(D_real_out)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_out, labels=tf.zeros_like(D_fake_out)))
    d_loss = d_loss_real + d_loss_fake

    # get loss for generator
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_out, labels=tf.ones_like(D_fake_out)))

    """ Training """
    #——————————————————方式1——————————————————————————————————————————
    # # divide trainable variables into a group for D and a group for G
    # t_vars = tf.trainable_variables()
    # d_vars = [var for var in t_vars if 'dis_' in var.name]
    # g_vars = [var for var in t_vars if 'gen_' in var.name]
    #
    # # optimizers
    # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #     d_optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5) \
    #               .minimize(d_loss, var_list=d_vars)
    #     g_optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5) \
    #               .minimize(g_loss, var_list=g_vars)

    #————————————————————————————————————————————————————————————————————


    #————————————————————————方式2————————————————————————————————————————
    gen_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
    dis_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')

    gen_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='generator')
    with tf.control_dependencies(gen_update_ops):
        train_gen=tf.train.AdamOptimizer(1e-3,beta1=0.5).minimize(g_loss,var_list=gen_vars)

    dis_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope='discriminator')
    with tf.control_dependencies(dis_update_ops):
        train_dis=tf.train.AdamOptimizer(1e-3,beta1=0.5).minimize(d_loss,var_list=dis_vars)

    # """" Testing """
    # # for test
    # fake_images = generator(ZN, is_training=False, reuse=True)#测试用

    """ Summary """
    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)

    # final summary operations
    g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    #——————————————————start train————————————————————————————————
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # saver to save model
    saver = tf.train.Saver()
    # summary writer
    writer = tf.summary.FileWriter(logdir=LOGS_DIRECTORY, graph=sess.graph)#——————————改

    if not os.path.exists('result_DCGAN/'):
        os.makedirs('result_DCGAN/')

    '''loop for train'''
    start_time = time.time()
    for step in range(100000):

        start = (step * batch_size) % n_train
        end = min(start + batch_size, n_train)

        img_batch=img_arr[start:end]
        #generator noise to feed to the generator
        noise_batch= np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)

        #(1)先固定G开始Train D,D要看来自样本的img和来自generator的img
        _,summary_str,dl=sess.run([train_dis,d_sum,d_loss],
                      feed_dict={IMG:img_batch,
                                 ZN:noise_batch
                                 })
        writer.add_summary(summary_str, step)

        #(2)固定D，train G（G只需要feed noise即可）
        _,summary_str,gl=sess.run([train_gen,g_sum,g_loss],
                                  feed_dict={ZN:noise_batch})
        writer.add_summary(summary_str, step)

        if step%1000==0:
            print('step:%d,discriminator_loss:%f,generator_loss:%f'%(step,dl,gl))


            """" Testing """
            fake_images = generator(ZN, is_training=False, reuse=True)  # 测试用
            final_samples=sess.run(fake_images,feed_dict={ZN:noise_batch})
            final_samples = np.reshape(final_samples, (-1, 100, 100, 1))
            from scipy.misc import imsave as ims
            # 显示生成的图片
            ims("./result_DCGAN/" + str(step) + ".jpg", merge(final_samples[0:batch_size], [8, 8]))

    # save model for final step
    save_path = saver.save(sess, MODEL_DIRECTORY)
    print("Model saved in file: %s" % save_path)


build_model()

















