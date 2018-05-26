#coding=utf-8
import tensorflow as tf
import numpy as np
import os
from vae_data import *
os.environ['CUDA_VISIBLE_DEVICES']='2,3'
MODEL_DIRECTORY = "CVAE_model/model.ckpt"
LOGS_DIRECTORY = "CVAE_model/train"
#use the default graph
tf.reset_default_graph()
batch_size = 100
n_latent = 500
train_epoch=200
num_sample=100
Mylog=open('VAE.txt','w+')#输出重定向
def sample_Z(m, n):
    return np.random.uniform(-1, 1, size=(m,n)).astype(np.float32)

def concat(z,y):
    return tf.concat([z,y],1)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x = tf.reshape(x, shape=[batch_size, 100, 100, 1])
    y=tf.reshape(y,[batch_size,1,1,15])
    return tf.concat([x, y*tf.ones([batch_size,100,100,15])], 3)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image[:,:,0]

    return img

def Encoder(x,is_training=True, reuse=False):

    with tf.variable_scope("encoder", reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        x = tf.layers.conv2d(x, filters=64, kernel_size=5, strides=2, padding='same',
                             kernel_initializer=w_init,
                             bias_initializer=b_init,
                             name='en_conv1')
        x = lrelu(x)

        x = tf.layers.conv2d(x, filters=128, kernel_size=5, strides=2, padding='same',
                             kernel_initializer=w_init,
                             bias_initializer=b_init,
                             name='en_conv2')
        x=tf.layers.batch_normalization(x,training=is_training,name='en_bn1')
        x = lrelu(x)

        x = tf.contrib.layers.flatten(x)

        x=tf.layers.dense(x,units=1024,
                          kernel_initializer=w_init,
                          bias_initializer=b_init,
                          name='en_fc1')
        x=tf.layers.batch_normalization(x,training=is_training,name='en_bn2')
        x=lrelu(x)

        z_mu = tf.layers.dense(x, units=n_latent,
                               kernel_initializer=w_init,
                               bias_initializer=b_init,
                               name='en_fc2_mu')
        z_sig = tf.layers.dense(x, units=n_latent,
                                kernel_initializer=w_init,
                                bias_initializer=b_init,
                                name='en_fc2_sig')

        return z_mu, z_sig


def Decoder(z_in,is_training=True,reuse=False):

    with tf.variable_scope("decoder", reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        x=tf.layers.dense(z_in,units=1024,
                          kernel_initializer=w_init,
                          bias_initializer=b_init,
                          name='de_fc1')
        x=tf.layers.batch_normalization(x,training=is_training,name='de_bn1')
        x=tf.nn.relu(x)

        x = tf.layers.dense(x, units=25*25*128,
                            kernel_initializer=w_init,
                            bias_initializer=b_init,
                            name='de_fc2')
        x=tf.layers.batch_normalization(x,training=is_training,name='de_bn2')
        x=tf.nn.relu(x)

        x = tf.reshape(x,[-1,25,25,128])

        x=tf.layers.conv2d_transpose(x, filters=64, kernel_size=5, strides=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     name='de_deconv1')
        x=tf.layers.batch_normalization(x,training=is_training,name='de_bn3')
        x = tf.nn.relu(x)

        x = tf.layers.conv2d_transpose(x, filters=1, kernel_size=5, strides=2, padding='same',
                                       kernel_initializer=w_init,
                                       bias_initializer=b_init,
                                       name='de_deconv2')
        x=tf.nn.sigmoid(x)
        return x

#train
def build_model():
    """ Graph Input """
    X=tf.placeholder(tf.float32,[None,100*100])
    Y=tf.placeholder(tf.float32,[None,15])
    Z=tf.placeholder(tf.float32,[None,n_latent])

    '''Train data'''
    imgpath = '/home/lu/cs/DCASE_NEW/DATA/data1/newfull/'
    img_arr, lab_arr = GetFlatPix(imgpath)  # (23400,10000),(23400,15)
    n_train = img_arr.shape[0]

    '''encoding'''
    mu, sigma = Encoder(conv_cond_concat(X,Y), is_training=True,reuse=False)#2

    '''sampling by re-parameterization technique'''
    eps = tf.random_normal(shape=tf.shape(mu))#defaut 0-1,float32
    z = mu + tf.exp(sigma / 2) * eps

    '''decoding'''
    out_img = Decoder(concat(z,Y), is_training=True, reuse=False)
    out_flat=tf.reshape(out_img,[-1,100*100])

    '''vae_loss'''
    recon_loss = tf.reduce_sum(tf.squared_difference(out_flat,X), 1)#均方差loss
    recon_loss_mean=tf.reduce_mean(recon_loss)
    # recon_loss = -tf.reduce_sum(IMG * tf.log(1e-8 + out_flat)         #交叉熵误差
    #             + (1 - IMG) * tf.log(1e-8 + 1 - out_flat), axis=1)

    kl_loss = 0.5 * tf.reduce_sum(tf.exp(sigma)- 1.- sigma +tf.square(mu), 1)
    kl_loss_mean=tf.reduce_mean(kl_loss)
    vae_loss = recon_loss_mean + kl_loss_mean
    # vae_loss = tf.reduce_mean(recon_loss + kl_loss)

    """ Training """
    # optimizers
    t_vars = tf.trainable_variables()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer= tf.train.AdamOptimizer(1e-3).minimize(vae_loss, var_list=t_vars)
    '''fake img'''
    fake_images = Decoder(concat(Z,Y), is_training=False, reuse=True)

    """ Summary """
    tf.summary.scalar("recon_loss", recon_loss_mean )#average loss! sum loss will error
    tf.summary.scalar("kl_loss", kl_loss_mean)
    tf.summary.scalar("vae_loss", vae_loss)

    # final summary operations
    merged_summary_op = tf.summary.merge_all()

    '''start a session'''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # saver to save model
    saver = tf.train.Saver()
    # summary writer
    writer = tf.summary.FileWriter(logdir=LOGS_DIRECTORY, graph=sess.graph)  # ——————————改

    # 如果当前的路径下不存在out这个文件夹，就创建它
    if not os.path.exists('results_CVAE/'):
        os.makedirs('results_CVAE/')

    '''loop for train'''
    import time
    import random
    start_time = time.time()
    for epoch in range(train_epoch):
        shuffle_idx = random.sample(range(0, n_train), n_train)
        shuffled_set = img_arr[shuffle_idx]
        shuffle_label = lab_arr[shuffle_idx]
        total_batch = n_train // batch_size  # -1最好能整除，不然会出dimension不匹配的error
        for iter in range(total_batch):
            x = shuffled_set[iter * batch_size:(iter + 1) * batch_size]
            y = shuffle_label[iter * batch_size:(iter + 1) * batch_size]

            _,summary_str,Tot_loss, Re_loss, KL_loss = sess.run(
                [optimizer,merged_summary_op, vae_loss, recon_loss, kl_loss],
                feed_dict={X: x,
                           Y: y,
                           })
            writer.add_summary(summary_str, epoch * total_batch + iter)

            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, total_loss: %.8f, rec_loss: %.8f,kl_loss:%.8f" \
                  % (epoch, iter, total_batch, time.time() - start_time, Tot_loss, np.mean(Re_loss), np.mean(KL_loss)))
            Mylog.write(
                str("Epoch: [%2d] [%4d/%4d] time: %4.4f, total_loss: %.8f, rec_loss: %.8f,kl_loss:%.8f" \
                    % (epoch, iter, total_batch, time.time() - start_time, Tot_loss, np.mean(Re_loss), np.mean(KL_loss)) + '\n'))
            Mylog.flush()

        # display each epoch
        """" Testing(specified condition,random noise) """
        # ————————从0-14逐个生成：one-hot——————————
        for d in range(15):
            y = np.zeros(batch_size, dtype=np.int64) + d
            y_one_hot = np.zeros((batch_size, 15))
            y_one_hot[np.arange(batch_size), y] = 1

            """" Testing """
            noise = sample_Z(num_sample, n_latent)

            final_samples = sess.run(fake_images,
                                     feed_dict={Z: noise,
                                                Y: y_one_hot
                                                })
            final_samples = np.reshape(final_samples, (-1, 100, 100, 1))
            from scipy.misc import imsave as ims

            ims("./results_CVAE/" + '_epoch%d' % epoch + '_test_class_%d.jpg' % d,
                merge(final_samples[0:batch_size], [10, 10]))

        if epoch % 50 == 0 or (epoch + 1) == train_epoch:
            save_path = saver.save(sess, MODEL_DIRECTORY)
            print("Model updated and saved in file: %s" % save_path)


build_model()
