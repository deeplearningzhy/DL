#coding=utf-8
import tensorflow as tf
import numpy as np
from .ops import batch_normal, de_conv, conv2d, fully_connect, lrelu

#encoder
def Encode(x):

    with tf.variable_scope('encoder') as scope:

        conv1 = tf.nn.relu(batch_normal(conv2d(x, output_dim=64, name='e_c1'), scope='e_bn1'))
        conv2 = tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='e_c2'), scope='e_bn2'))
        conv3 = tf.nn.relu(batch_normal(conv2d(conv2 , output_dim=256, name='e_c3'), scope='e_bn3'))
        conv3 = tf.reshape(conv3, [batch_size, 256 * 8 * 8])
        fc1 = tf.nn.relu(batch_normal(fully_connect(conv3, output_size=1024, scope='e_f1'), scope='e_bn4'))
        z_mean = fully_connect(fc1 , output_size=128, scope='e_f2')
        z_sigma = fully_connect(fc1, output_size=128, scope='e_f3')

    return z_mean, z_sigma


def discriminate(self, x_var, reuse=False):

    with tf.variable_scope("discriminator") as scope:

        if reuse:
            scope.reuse_variables()

        conv1 = tf.nn.relu(conv2d(x_var, output_dim=32, name='dis_conv1'))
        conv2= tf.nn.relu(batch_normal(conv2d(conv1, output_dim=128, name='dis_conv2'), scope='dis_bn1', reuse=reuse))
        conv3= tf.nn.relu(batch_normal(conv2d(conv2, output_dim=256, name='dis_conv3'), scope='dis_bn2', reuse=reuse))
        conv4 = conv2d(conv3, output_dim=256, name='dis_conv4')
        middle_conv = conv4
        conv4= tf.nn.relu(batch_normal(conv4, scope='dis_bn3', reuse=reuse))
        conv4= tf.reshape(conv4, [self.batch_size, -1])

        fl = tf.nn.relu(batch_normal(fully_connect(conv4, output_size=256, scope='dis_fully1'), scope='dis_bn4', reuse=reuse))
        output = fully_connect(fl , output_size=1, scope='dis_fully2')

        return middle_conv, output

def generate(self, z_var, reuse=False):

    with tf.variable_scope('generator') as scope:

        if reuse == True:
            scope.reuse_variables()

        d1 = tf.nn.relu(batch_normal(fully_connect(z_var , output_size=8*8*256, scope='gen_fully1'), scope='gen_bn1', reuse=reuse))
        d2 = tf.reshape(d1, [self.batch_size, 8, 8, 256])
        d2 = tf.nn.relu(batch_normal(de_conv(d2 , output_shape=[self.batch_size, 16, 16, 256], name='gen_deconv2'), scope='gen_bn2', reuse=reuse))
        d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, 32, 32, 128], name='gen_deconv3'), scope='gen_bn3', reuse=reuse))
        d4 = tf.nn.relu(batch_normal(de_conv(d3, output_shape=[self.batch_size, 64, 64, 32], name='gen_deconv4'), scope='gen_bn4', reuse=reuse))
        d5 = de_conv(d4, output_shape=[self.batch_size, 64, 64, 3], name='gen_deconv5', d_h=1, d_w=1)

        return tf.nn.tanh(d5)

