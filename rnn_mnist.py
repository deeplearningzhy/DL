#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf
from tensorflow.contrib import rnn

#import mnist dataset导入数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./MNIST/",one_hot=True)

#define constants定义常量
#unrolled through 28 time steps以28步数展开rnn网络
time_steps=28

#hidden LSTM units每个单元格的隐藏节点个数
num_units=128

#rows of 28 pixels（每一行有多少个像素）
n_input=28

#learning rate for adam（学习率）
learning_rate=0.001

#mnist is meant to be classified in 10 classes(0-9).
n_classes=10

#size of batch
batch_size=128

#现在设置占位、权重以及偏置变量
#（用于将输出的形态从 [batch_size,num_units] 转换为 [batch_size,n_classes]），
# 从而可以预测正确的类别。

# weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# defining placeholders
# input image placeholder
x = tf.placeholder("float", [None, time_steps, n_input])
# input label placeholder
y = tf.placeholder("float", [None, n_classes])

#processing the input tensor from [batch_size,n_steps,n_input]
#to "time_steps" number of [batch_size,n_input] tensors
input = tf.unstack(x, time_steps, 1)

# defining the network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

#converting last output of dimension [batch_size,num_units]
#to[batch_size,n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# loss_function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# model evaluation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while iter < 800:
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
        batch_x = batch_x.reshape((batch_size, time_steps, n_input))
        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter % 10 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("For iter ", iter)
            print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")

        iter = iter + 1

    #calculating test accuracy
    test_data =mnist.test.images[:128].reshape((-1,time_steps,n_input))
    test_label =mnist.test.labels[:128]
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x:test_data,y:test_label}))

